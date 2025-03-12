module TestSuite.Test.Convolution.CUDNN (tests) where

import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Foreign
import Foreign.C
import System.IO.Unsafe (unsafePerformIO)
import Test.Tasty
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck

import Test.Tensor (Tensor(..))
import Test.Tensor qualified as Tensor
import Test.Tensor.TestValue

import TestSuite.Test.Convolution.Examples3B1B
import TestSuite.Util.TestKernel

{-------------------------------------------------------------------------------
  Lists of tests
-------------------------------------------------------------------------------}

tests :: TestTree
tests = testGroup "TestSuite.Test.Convolution.CUDNN" [
      testGroup "Sanity" [
            testCase "bindingVersion" test_bindingVersion
          , testCase "libraryVersion" test_libraryVersion
        ]
    , testGroup "Examples" [
          testCase "weightedMovingAverage" example_weightedMovingAverage
        ]
    , testGroup "Properties" [
          testGroup "matchesModel" [
              testGroup "1d" [
                  testProperty "kernelSize2" $ prop_matchesModel_1d @Nat2
                , testProperty "kernelSize3" $ prop_matchesModel_1d @Nat3
                , testProperty "kernelSize4" $ prop_matchesModel_1d @Nat4
                ]
            , testProperty "4d" prop_matchesModel
            ]
        ]
        , testProperty "mode" prop_mode
    ]

{-------------------------------------------------------------------------------
  Sanity checks
-------------------------------------------------------------------------------}

-- | Confirm that basic FFI interaction works as expected
test_bindingVersion :: Assertion
test_bindingVersion =
    assertEqual "" 1 $
      c_test_cudnn_binding_version

-- | Confirm cuDNN version (we expect at least version 9.0)
test_libraryVersion :: Assertion
test_libraryVersion =
    if c_test_cudnn_library_version >= 90000
      then return ()
      else assertFailure "Expect cuDNN version 9.0 or higher"

{-------------------------------------------------------------------------------
  Examples
-------------------------------------------------------------------------------}

example_weightedMovingAverage :: Assertion
example_weightedMovingAverage =
    assertEqual "" (Tensor.dim1 $ movingWeightedAverageResult @TestValue) $
      convolveCUDNN_1d
        (Tensor.dim1 movingWeightedAverageKernel)
        (Tensor.padWith 0 2 $ Tensor.dim1 $ movingAverageInput @TestValue)

{-------------------------------------------------------------------------------
  Properties

  NOTE: cuDNN does not like it when the size of the image is smaller than
  the size of the kernel.
-------------------------------------------------------------------------------}

-- | Compare our implementation against cuDNN, 1D case
prop_matchesModel_1d :: forall w.
     SNatI w
  => TestKernel '[w] TestValue  -- ^ Kernel
  -> Tensor Nat1 TestValue      -- ^ Input
  -> Property
prop_matchesModel_1d (testKernel -> kernel) input =
    Tensor.sizeAtLeast (minWidth ::: VNil) input ==>
          convolveCUDNN_1d kernel input
      === Tensor.convolve kernel input
  where
    minWidth :: Int
    minWidth = fromIntegral $ snatToNatural (snat @w)

-- | Compare our implementation against cuDNN, general case
prop_matchesModel :: ConvolutionParams TestValue -> Property
prop_matchesModel params =
        convolveCUDNN c_mode_cross_correlation stride kernels input
    === convolve_cuDNN_style params
  where
    ConvolutionParams{stride, input, kernels} = params

prop_mode :: ConvolutionParams TestValue -> Property
prop_mode params =
        convolveCUDNN
          c_mode_cross_correlation
          stride
          kernels
          input
    === convolveCUDNN
          c_mode_convolution
          stride
          ( Tensor.foreach kernels $ \outputFeature ->
              Tensor.foreach outputFeature $ \inputFeature ->
                Tensor.rotate inputFeature
          )
          input
  where
    ConvolutionParams{stride, input, kernels} = params

{-------------------------------------------------------------------------------
  Model
-------------------------------------------------------------------------------}

-- | cuDNN-style convolutions, but using our implementation
convolve_cuDNN_style :: Real a => ConvolutionParams a -> Tensor Nat4 a
convolve_cuDNN_style params =
    Tensor.foreach input $ \channels -> Tensor [
        -- Both the input and the kernel have 3 channels, so the result must
        -- be a singleton "channel".
        case Tensor.convolveWithStride stride' inputFeatures channels of
          Tensor [result] -> result
          _otherwise -> error "unexpected result"
      | inputFeatures <- Tensor.getTensor kernels
      ]
  where
    ConvolutionParams{stride = (sv, sh), input, kernels} = params

    stride' :: Vec Nat3 Int
    stride' = 1 ::: sv ::: sh ::: VNil

-- | Convolution parameters
--
-- Although both the input and the output are 4D tensors, their structure is
-- different:
--
-- * The input is NCHW:
--   - N images
--   - each image has C channels
--   - height H and width W
--
-- * The output is KCRS:
--   - K "output features"
--   - C "input features"
--   - height R and width S
--
-- For every input image we compute K output images ("channels"); each output
-- image results from applying C 2D kernels to each channel, adding up the
-- results. The result is an N*K*H*W tensor.
data ConvolutionParams a = ConvolutionParams {
      stride  :: (Int, Int)
    , input   :: Tensor Nat4 a
    , kernels :: Tensor Nat4 a
    }
  deriving stock (Show)

instance (Arbitrary a, Num a, Eq a) => Arbitrary (ConvolutionParams a) where
  arbitrary = sized $ \n -> do
      numImages      <- choose (1, max 1 n)
      inputFeatures  <- choose (1, 3)
      outputFeatures <- choose (1, 3)
      kernelHeight   <- choose (1, 5)
      kernelWidth    <- choose (1, 5)
      inputHeight    <- choose (kernelHeight, max kernelHeight n)
      inputWidth     <- choose (kernelWidth,  max kernelWidth  n)

      let inputSize :: Tensor.Size Nat4
          inputSize = numImages
                  ::: inputFeatures
                  ::: inputHeight
                  ::: inputWidth
                  ::: VNil

      let kernelSize :: Tensor.Size Nat4
          kernelSize = outputFeatures
                   ::: inputFeatures
                   ::: kernelHeight
                   ::: kernelWidth
                   ::: VNil

      stride  <- (,) <$> choose (1, 5) <*> choose (1, 5)
      input   <- Tensor.arbitraryOfSize inputSize arbitrary
      kernels <- Tensor.arbitraryOfSize kernelSize arbitrary

      return ConvolutionParams {stride, input, kernels}

  -- Shrinking is a bit complicated, because we need to maintain consistency
  -- between the kernels and the input
  shrink params = concat [
        -- Shrink stride
        [ params{stride = (sv', sh')}
        | (sv', sh') <- shrink stride
        , sv' > 0
        , sh' > 0
        ]

        -- Shrink input size
      , [ params{input = input', kernels = kernels'}
        | axe <- Tensor.allAxes (Tensor.size input)
        , let input'   = Tensor.axeWith axe input
        , let kernels' = adjustKernels axe kernels

          -- Image should not be smaller than the kernel
        , let (_ ::: _ ::: ih ::: iw ::: VNil) = Tensor.size input'
        , let (_ ::: _ ::: kh ::: kw ::: VNil) = Tensor.size kernels'
        , ih >= kh
        , iw >= kw
        ]

        -- Shrink the kernel
      , [ params{input = input', kernels = kernels'}
        | axe <- Tensor.allAxes (Tensor.size kernels)
        , let kernels' = Tensor.axeWith axe kernels
        , let input'   = adjustInput axe input
        ]

        -- Shrink input elements
      , [ params{input = images'}
        | images' <- Tensor.shrinkElem (Just Tensor.zero) shrink input
        ]

        -- Shrink kernel element
      , [ params{kernels = outputFeatures'}
        | outputFeatures' <- Tensor.shrinkElem Nothing shrink kernels
        ]
      ]
    where
      ConvolutionParams{stride, input, kernels} = params

      -- Adjust each kernel after we axe some part of the input
      adjustKernels :: Tensor.Axe Nat4 -> Tensor Nat4 a -> Tensor Nat4 a
      adjustKernels (Tensor.AxeHere _) =
          -- We dropped some images; kernel is unaffected
          id
      adjustKernels axe@(Tensor.AxeNested (Tensor.AxeHere _)) =
          -- We dropped some input channels; also drop the corresponding
          -- input features from the kernel
          Tensor.axeWith axe
      adjustKernels _otherwise =
          -- We reduced image height or width; kernel is unaffected
          -- (though we must check that the image is large enough now)
          id

      -- Adjust the input after we axe some of the kernels
      adjustInput :: Tensor.Axe Nat4 -> Tensor Nat4 a -> Tensor Nat4 a
      adjustInput (Tensor.AxeHere _) =
          -- We dropped some output features; input is unaffected
          id
      adjustInput axe@(Tensor.AxeNested (Tensor.AxeHere _)) =
          -- We dropped some input features; drop the corresponding channels
          Tensor.axeWith axe
      adjustInput _otherwise =
          -- We shrunk the kernel size (height or width), input is unaffected
          id

{-------------------------------------------------------------------------------
  Compute convolution using cuDNN
-------------------------------------------------------------------------------}

convolveCUDNN_1d :: forall a.
     (Fractional a, Real a)
  => Tensor Nat1 a -> Tensor Nat1 a -> Tensor Nat1 a
convolveCUDNN_1d kernel input = extract1d $
    convolveCUDNN
      c_mode_cross_correlation
      (1, 1)
      (Tensor [Tensor [Tensor [kernel]]])
      (Tensor [Tensor [Tensor [input]]])
  where
    extract1d :: Tensor Nat4 a -> Tensor Nat1 a
    extract1d (Tensor [Tensor [Tensor [output]]]) = output
    extract1d _ = error "convolveCUDNN_1d: unexpected output"

convolveCUDNN ::
     (Fractional a, Real a)
  => CudnnConvolutionMode
  -> (Int, Int)    -- ^ vertical and horizontal stride
  -> Tensor Nat4 a -- ^ kernel
  -> Tensor Nat4 a -- ^ input
  -> Tensor Nat4 a
convolveCUDNN mode (sv, sh) kernels input = unsafePerformIO $
    Tensor.unsafeWithCArray (realToFrac <$> kernels) $ \kernelsPtr ->
    Tensor.unsafeWithCArray (realToFrac <$> input)   $ \inputPtr   ->
    alloca $ \outputHeightPtr ->
    alloca $ \outputWidthPtr  -> do
      outputPtr <-
        c_test_cudnn_convolve
          mode
          (fromIntegral sv)
          (fromIntegral sh)
          (fromIntegral k)
          (fromIntegral kh)
          (fromIntegral kw)
          kernelsPtr
          (fromIntegral n)
          (fromIntegral c)
          (fromIntegral ih)
          (fromIntegral iw)
          inputPtr
          outputHeightPtr
          outputWidthPtr
      oh <- fromIntegral <$> peek outputHeightPtr
      ow <- fromIntegral <$> peek outputWidthPtr
      outputFPtr <- newForeignPtr finalizerFree outputPtr
      let outputSize = n ::: k ::: oh ::: ow ::: VNil
      return $ realToFrac <$> Tensor.unsafeFromCArray outputSize outputFPtr
  where
    n ::: c ::: ih ::: iw ::: VNil = Tensor.size input
    k ::: _ ::: kh ::: kw ::: VNil = Tensor.size kernels

{-------------------------------------------------------------------------------
  FFI imports
-------------------------------------------------------------------------------}

type CudnnConvolutionMode = CInt

foreign import capi unsafe "test-cudnn.h test_cudnn_binding_version"
  c_test_cudnn_binding_version :: Int

foreign import capi unsafe "test-cudnn.h test_cudnn_library_version"
  c_test_cudnn_library_version :: Int

foreign import capi unsafe "cudnn.h value CUDNN_CONVOLUTION"
  c_mode_convolution :: CudnnConvolutionMode

foreign import capi unsafe "cudnn.h value CUDNN_CROSS_CORRELATION"
  c_mode_cross_correlation :: CudnnConvolutionMode

foreign import capi unsafe "test-cudnn.h test_cudnn_convolve"
  c_test_cudnn_convolve ::
       CudnnConvolutionMode
    -> CInt       -- ^ vertical_stride
    -> CInt       -- ^ horizontal_stride
    -> CInt       -- ^ num_kernels
    -> CInt       -- ^ kernel_height
    -> CInt       -- ^ kernel_width
    -> Ptr Float  -- ^ kernel
    -> CInt       -- ^ num_images
    -> CInt       -- ^ input_channels
    -> CInt       -- ^ input_height
    -> CInt       -- ^ input_width
    -> Ptr Float  -- ^ input
    -> Ptr CInt   -- ^ output_height
    -> Ptr CInt   -- ^ output_width
    -> IO (Ptr Float)
