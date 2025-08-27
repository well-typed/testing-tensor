-- | Tensors (n-dimensional arrays)
--
-- This is an implementation of tensors that emphasizes simplicify above all; it
-- is meant for use in QuickCheck tests.
--
-- Intended for qualified import.
--
-- > import Test.Tensor (Tensor)
-- > import Test.Tensor qualified as Tensor
module Test.Tensor (
    -- * Definition
    Tensor(..)
  , getScalar
  , getTensor
    -- ** Convenience constructors
  , scalar
  , dim1
  , dim2
  , dim3
  , dim4
  , dim5
  , dim6
  , dim7
  , dim8
  , dim9
    -- * Size
  , Size
  , size
  , sizeAtLeast
    -- * Standard operations
  , zipWith
  , replicate
  , rotate
  , distrib
  , transpose
  , foreach
  , foreachWith
    -- * Subtensors
  , subs
  , convolve
  , convolveWithStride
  , padWith
  , padWith'
    -- * Conversions
  , Lists
  , toLists
  , fromLists
  , fromList
    -- * QuickCheck support
    -- ** Generation
  , arbitraryOfSize
    -- ** Shrinking
  , shrinkWith
  , shrinkWith'
  , shrinkElem
    -- *** Axes
  , Axe(..)
  , allAxes
  , axeWith
  , axeSize
    -- *** Zeroing
  , Zero(..)
  , zero
  , zeroWith
    -- * FFI
  , toStorable
  , fromStorable
  , unsafeWithCArray
  , unsafeFromCArray
  , unsafeFromPrealloc
  , unsafeFromPrealloc_
  ) where

import Prelude hiding (zipWith, replicate)

import Control.Monad.Trans.State (StateT(..), evalStateT)
import Data.Bifunctor
import Data.Foldable (foldl')
import Data.Foldable qualified as Foldable
import Data.List qualified as L
import Data.Maybe (catMaybes)
import Data.Ord
import Data.Proxy
import Data.Type.Nat
import Data.Vec.Lazy (Vec(..))
import Data.Vec.Lazy qualified as Vec
import Data.Vector.Storable qualified as Storable (Vector)
import Data.Vector.Storable qualified as Vector
import Foreign hiding (rotate)
import GHC.Show (appPrec1, showSpace)
import GHC.Stack
import Numeric.Natural
import Test.QuickCheck (Arbitrary(..), Arbitrary1(..), Gen)
import Test.QuickCheck qualified as QC

{-------------------------------------------------------------------------------
  Definition
-------------------------------------------------------------------------------}

-- | N-dimensional tensor
--
-- Invariants:
--
-- * The dimension must be strictly positive (zero is not allowed)
-- * Tensors must be rectangular
--
-- (These invariants could in principle be enforced by using more precise types,
-- but at the cost of much more complex code.)
data Tensor n a where
  Scalar :: a -> Tensor Z a
  Tensor :: [Tensor n a] -> Tensor (S n) a

deriving stock instance Eq a => Eq (Tensor n a)

deriving stock instance Functor     (Tensor n)
deriving stock instance Traversable (Tensor n)
deriving stock instance Foldable    (Tensor n)

getScalar :: Tensor Z a -> a
getScalar (Scalar x) = x

getTensor :: Tensor (S n) a -> [Tensor n a]
getTensor (Tensor xs) = xs

{-------------------------------------------------------------------------------
  Size
-------------------------------------------------------------------------------}

type Size n = Vec n Int

-- | Analogue of 'List.length'
size :: Tensor n a -> Size n
size (Scalar _)  = VNil
size (Tensor xs) = L.length xs ::: size (L.head xs)

-- | Check that each dimension has at least the specified size
sizeAtLeast :: Size n -> Tensor n a -> Bool
sizeAtLeast sz = and . Foldable.toList . Vec.zipWith (<=) sz . size

{-------------------------------------------------------------------------------
  Standard operations
-------------------------------------------------------------------------------}

-- | Analogue of 'List.zipWith'
zipWith :: (a -> b -> c) -> Tensor n a -> Tensor n b -> Tensor n c
zipWith f (Scalar a)  (Scalar b)  = Scalar (f a b)
zipWith f (Tensor as) (Tensor bs) = Tensor $ L.zipWith (zipWith f) as bs

-- | Analogue of 'List.replicate'
replicate :: Size n -> a -> Tensor n a
replicate VNil       x = Scalar x
replicate (n ::: ns) x = Tensor $ L.replicate n (replicate ns x)

-- | Analogue of 'List.reverse'
--
-- This amounts to a 180 degrees rotation of the tensor.
rotate :: Tensor n a -> Tensor n a
rotate (Scalar x)  = Scalar x
rotate (Tensor xs) = Tensor $ map rotate (L.reverse xs)

-- | Analogue of @distribute@ (@distributive@ package)
--
-- Since we don't track the complete size of the tensor at the type level, we
-- must be told how large the resulting tensor is going to be.
distrib :: Functor f => Size n -> f (Tensor n a) -> Tensor n (f a)
distrib VNil       = Scalar . fmap getScalar
distrib (n ::: ns) = Tensor . fmap (distrib ns) . distribList n . fmap getTensor

-- | Transpose
--
-- This is essentially a special case of 'distrib'.
transpose :: Tensor Nat2 a -> Tensor Nat2 a
transpose = fromLists . L.transpose . toLists

-- | Map element over the first dimension of the tensor
foreach :: Tensor (S n) a -> (Tensor n a -> Tensor m b) -> Tensor (S m) b
foreach (Tensor as) f = Tensor (Prelude.map f as)

-- | Variation of 'foreach' with an auxiliary list
foreachWith ::
    Tensor (S n) a
 -> [x]
 -> (Tensor n a -> x -> Tensor m b)
 -> Tensor (S m) b
foreachWith (Tensor as) xs f = Tensor (L.zipWith f as xs)

{-------------------------------------------------------------------------------
  Subtensors
-------------------------------------------------------------------------------}

-- | Compute number of subtensors
--
-- Internal auxiliary.
numSubs ::
     Size n  -- ^ Kernel size
  -> Size n  -- ^ Input size
  -> Size n  -- ^ Output size
numSubs VNil       VNil       = VNil
numSubs (k ::: ks) (i ::: is) = (i - k + 1) ::: numSubs ks is

-- | Subtensors of the specified size
subs :: Size n -> Tensor n a -> Tensor n (Tensor n a)
subs = \kernelSize input ->
    go (numSubs kernelSize (size input)) kernelSize input
  where
    go :: Size n -> Size n -> Tensor n a -> Tensor n (Tensor n a)
    go VNil       VNil       (Scalar x)  = Scalar (Scalar x)
    go (r ::: rs) (n ::: ns) (Tensor xs) = Tensor [
          Tensor <$> distrib rs selected
        | selected <- consecutive r n (map (go rs ns) xs)
        ]

-- | Apply stride.
--
-- This is the N-dimensional equivalent of 'everyNth'.
--
-- Internal auxiliary.
applyStride :: Vec n Int -> Tensor n a -> Tensor n a
applyStride VNil       (Scalar x)  = Scalar x
applyStride (s ::: ss) (Tensor xs) = Tensor $
    everyNth s (map (applyStride ss) xs)

-- | Convolution
--
-- See 'padWith' for adjusting boundary conditions.
convolve ::
     (SNatI n, Num a)
  => Tensor n a  -- ^ Kernel
  -> Tensor n a  -- ^ Input
  -> Tensor n a
convolve = convolveWithStride (pure 1)

-- | Generalization of 'convolve' when using a non-default stride
convolveWithStride :: forall n a.
     Num a
  => Vec n Int   -- ^ Stride
  -> Tensor n a  -- ^ Kernel
  -> Tensor n a  -- ^ Input
  -> Tensor n a
convolveWithStride stride kernel input =
    aux <$> applyStride stride (subs (size kernel) input)
  where
    aux :: Tensor n a -> a
    aux = foldl' (+) 0 . zipWith (*) kernel

{-------------------------------------------------------------------------------
  Padding
-------------------------------------------------------------------------------}

-- | Add uniform padding
padWith :: SNatI n => a -> Int -> Tensor n a -> Tensor n a
padWith padding n = padWith' padding (pure (n, n))

-- | Generalization of 'padWith' with different padding per dimension
padWith' :: forall n a. a -> Vec n (Int, Int) -> Tensor n a -> Tensor n a
padWith' padding paddingSize tensor =
    go paddingSize newSize tensor
  where
    newSize :: Size n
    newSize = Vec.zipWith (\(b, a) n -> n + b + a) paddingSize (size tensor)

    go :: forall m. Vec m (Int, Int) -> Size m -> Tensor m a -> Tensor m a
    go VNil                     VNil       (Scalar x)  = Scalar x
    go ((before, after) ::: ps) (_ ::: ns) (Tensor xs) = Tensor $ concat [
          L.replicate before $ replicate ns padding
        , map (go ps ns) xs
        , L.replicate after $ replicate ns padding
        ]

{-------------------------------------------------------------------------------
  QuickCheck support
-------------------------------------------------------------------------------}

arbitraryOfSize :: Size n -> Gen a -> Gen (Tensor n a)
arbitraryOfSize sz = sequence . replicate sz

data Axe (n :: Nat) where
  -- | Axe some elements from the current dimension
  --
  -- We record which elements to drop as an @(offset, length)@ pair.
  AxeHere :: (Int, Int) -> Axe (S n)

  -- | Axe some elements from a nested dimension
  --
  -- In order to keep the tensor square, we must apply the same axe for every
  -- element of the /current/ dimension
  AxeNested :: Axe n -> Axe (S n)

deriving instance Show (Axe n)

-- | How many elements are removed by this axe?
--
-- Examples:
--
-- > axeSize (2 ::: 100 ::: VNil) (AxeHere (0, 1))               == 100
-- > axeSize (2 ::: 100 ::: VNil) (AxeNested (AxeHere (0, 99)))  == 198
axeSize :: Size n -> Axe n -> Int
axeSize = flip go
  where
    go ::  Axe n -> Size n -> Int
    go (AxeHere (_, len)) (_ ::: ns) = len * L.foldl' (*) 1 ns
    go (AxeNested axe)    (n ::: ns) = n * go axe ns

-- | All possible ways to axe some elements
--
-- This is adopted from the implementation of 'shrinkList' (in a way, an 'Axe'
-- is an explanation of the decisions made by 'shrinkList', generalized to
-- multiple dimensions).
--
-- Axes are sorted to remove as many elements as early as possible.
allAxes :: Size n -> [Axe n]
allAxes = \sz ->
    L.sortBy (flip $ comparing (axeSize sz)) $ go sz
  where
    go :: Size n -> [Axe n]
    go VNil       = []
    go (n ::: ns) = concat [
          concat [
              L.map AxeHere (removes 0 k n)
            | k <- takeWhile (> 0) (iterate (`div` 2) (n `div` 2))
            ]
        , L.map AxeNested (go ns)
        ]

    removes :: Int -> Int -> Int -> [(Int, Int)]
    removes offset k n
      | k > n     = []
      | otherwise = (offset, k) : removes (offset + k) k (n - k)

-- | Remove elements from the tensor (shrink dimensions)
axeWith :: Axe n -> Tensor n a -> Tensor n a
axeWith (AxeHere (offset, len)) (Tensor xss) = Tensor $
    before <> after
  where
    (before, dropFrom) = L.splitAt offset xss
    (_dropped, after)  = L.splitAt len dropFrom
axeWith (AxeNested axe) (Tensor xss) = Tensor $
    L.map (axeWith axe) xss

-- | Zero element
data Zero a where
  Zero :: Eq a => a -> Zero a

-- | Default 'Zero'
zero :: (Num a, Eq a) => Zero a
zero = Zero 0

-- | Zero elements in the tensor (leaving dimensions the same)
--
-- Returns 'Nothing' if the specified region was already zero everywhere.
zeroWith :: forall n a. Zero a -> Axe n -> Tensor n a -> Maybe (Tensor n a)
zeroWith (Zero z) = \axe tensor ->
    case go axe (size tensor) tensor of
      (_, False)      -> Nothing
      (tensor', True) -> Just tensor'
  where
    -- Additionally returns if anything changed
    go :: forall n'. Axe n' -> Size n' -> Tensor n' a -> (Tensor n' a, Bool)
    go (AxeHere (offset, len)) (_ ::: ns) (Tensor xss) = (
          Tensor $ before <> L.replicate len (replicate ns z) <> after
        , any (/= z) (Tensor dropped)
        )
      where
         (before, dropFrom) = L.splitAt offset xss
         (dropped, after)   = L.splitAt len dropFrom
    go (AxeNested axe) (_ ::: ns) (Tensor xss) =
        bimap Tensor or $ L.unzip $ L.map (go axe ns) xss

-- | Shrink tensor
shrinkWith ::
     Maybe (Zero a)  -- ^ Optional zero element (see 'shrinkElem')
  -> (a -> [a])      -- ^ Shrink individual elements
  -> Tensor n a -> [Tensor n a]
shrinkWith mZero f xs = shrinkWith' (allAxes (size xs)) mZero f xs

-- | Generalization of 'shrinkWith'
shrinkWith' :: forall n a.
     [Axe n]         -- ^ Shrink the size of the tensor (see 'allAxes')
  -> Maybe (Zero a)  -- ^ Optional zero element (see 'shrinkElem')
  -> (a -> [a])      -- ^ Shrink elements of the tensor
  -> Tensor n a -> [Tensor n a]
shrinkWith' axes mZero f xss = concat [
      [axeWith axe xss | axe <- axes]
    , shrinkElem mZero f xss
    ]

-- | Shrink an element of the tensor, leaving the size of the tensor unchanged
--
-- If a zero element is specified, we will first try to replace entire regions
-- of the tensor by zeroes; this can dramatically speed up shrinking.
shrinkElem :: forall n a.
     Maybe (Zero a)  -- ^ Optional zero element
  -> (a -> [a])      -- ^ Shrink individual elements
  -> Tensor n a -> [Tensor n a]
shrinkElem mZero f tensor = concat [
      case mZero of
        Nothing -> []
        Just z  -> catMaybes [
            zeroWith z axe tensor
          | axe <- allAxes overallSize
          , axeSize overallSize axe > 1
          ]
    , shrinkOne tensor
    ]
  where
    overallSize :: Size n
    overallSize = size tensor

    shrinkOne :: forall n'. Tensor n' a -> [Tensor n' a]
    shrinkOne (Scalar x)   = Scalar <$> f x
    shrinkOne (Tensor xss) = [
          Tensor $ before ++ [xs'] ++ after
        | (before, xs, after) <- pickOne xss
        , xs' <- shrinkOne xs
        ]

instance (SNatI n, Arbitrary a, Num a, Eq a) => Arbitrary (Tensor n a) where
  arbitrary = liftArbitrary arbitrary
  shrink    = shrinkWith (Just (Zero 0)) shrink

-- | Lift generators and shrinkers
--
-- NOTE: Since we cannot put any constraints on the type of the elements here,
-- we cannot use any zero elements. Using 'shrink' (or 'shrinkWith' directly)
-- might result in faster shrinking.
instance SNatI n => Arbitrary1 (Tensor n) where
  liftArbitrary g = QC.sized $ \n -> do
      sz :: Size n <- liftArbitrary $ QC.choose (1, 1 + n)
      arbitraryOfSize sz g

  liftShrink f = shrinkWith Nothing f

{-------------------------------------------------------------------------------
  FFI
-------------------------------------------------------------------------------}

-- | Translate to storable vector
--
-- The tensor is laid out in order specified (outer dimensions before inner).
toStorable :: Storable a => Tensor n a -> Storable.Vector a
toStorable = Vector.fromList . Foldable.toList

-- | Translate from storable vector
--
-- Throws an exception if the vector does not contain enough elements.
fromStorable ::
     (HasCallStack, Storable a)
  => Size n -> Storable.Vector a -> Tensor n a
fromStorable sz = fromList sz . Vector.toList

-- | Get pointer to elements of the tensor
--
-- See 'toStorable' for discussion of the layout.
--
-- The data should not be modified through the pointer, and the pointer should
-- not be used outside its scope.
unsafeWithCArray :: Storable a => Tensor n a -> (Ptr a -> IO r) -> IO r
unsafeWithCArray tensor = Vector.unsafeWith (toStorable tensor)

-- | Construct tensor from C array
--
-- The data should not be modified through the pointer after the tensor has
-- been constructed.
unsafeFromCArray :: Storable a => Size n -> ForeignPtr a -> Tensor n a
unsafeFromCArray sz fptr =
    fromStorable sz $ Vector.unsafeFromForeignPtr0 fptr n
  where
    n :: Int
    n = L.foldl' (*) 1 sz

-- | Construct tensor from preallocated C array
--
-- Allocates sufficient memory to hold the elements of the tensor; writing more
-- data will result in invalid memory access. The pointer should not be used
-- outside its scope.
unsafeFromPrealloc ::
     Storable a
  => Size n -> (Ptr a -> IO r) -> IO (Tensor n a, r)
unsafeFromPrealloc sz k = do
    fptr <- mallocForeignPtrArray n
    res  <- withForeignPtr fptr k
    return (unsafeFromCArray sz fptr, res)
  where
    n :: Int
    n = L.foldl' (*) 1 sz

-- | Like 'unsafeFromPrealloc' but without an additional return value
unsafeFromPrealloc_ ::
     Storable a
  => Size n -> (Ptr a -> IO ()) -> IO (Tensor n a)
unsafeFromPrealloc_ sz = fmap fst . unsafeFromPrealloc sz

{-------------------------------------------------------------------------------
  Convenience constructors
-------------------------------------------------------------------------------}

scalar :: a -> Tensor Nat0 a
scalar = fromLists

dim1 :: [a] -> Tensor Nat1 a
dim1 = fromLists

dim2 :: [[a]] -> Tensor Nat2 a
dim2 = fromLists

dim3 :: [[[a]]] -> Tensor Nat3 a
dim3 = fromLists

dim4 :: [[[[a]]]] -> Tensor Nat4 a
dim4 = fromLists

dim5 :: [[[[[a]]]]] -> Tensor Nat5 a
dim5 = fromLists

dim6 :: [[[[[[a]]]]]] -> Tensor Nat6 a
dim6 = fromLists

dim7 :: [[[[[[[a]]]]]]] -> Tensor Nat7 a
dim7 = fromLists

dim8 :: [[[[[[[[a]]]]]]]] -> Tensor Nat8 a
dim8 = fromLists

dim9 :: [[[[[[[[[a]]]]]]]]] -> Tensor Nat9 a
dim9 = fromLists

{-------------------------------------------------------------------------------
  Conversions

  This is primarily useful for specify tensor constants.
-------------------------------------------------------------------------------}

type family Lists n a where
  Lists Z     a = a
  Lists (S n) a = [Lists n a]

toLists :: Tensor n a -> Lists n a
toLists (Scalar x)  = x
toLists (Tensor xs) = map toLists xs

fromLists :: SNatI n => Lists n a -> Tensor n a
fromLists = go snat
  where
    go :: SNat n -> Lists n a -> Tensor n a
    go SZ = Scalar
    go SS = Tensor . map (go snat)

-- | Inverse to 'Foldable.toList'
--
-- Throws a pure exception if the list does not contain enough elements.
fromList :: forall n a. HasCallStack => Size n -> [a] -> Tensor n a
fromList sz xs =
    checkEnoughElems . flip evalStateT xs $ sequenceA (replicate sz genElem)
  where
    genElem :: StateT [a] Maybe a
    genElem = StateT L.uncons

    checkEnoughElems :: Maybe (Tensor n a) -> Tensor n a
    checkEnoughElems Nothing  = error "fromList: insufficient elements"
    checkEnoughElems (Just t) = t

{-------------------------------------------------------------------------------
  Show instance
-------------------------------------------------------------------------------}

showLists :: Show a => Proxy a -> SNat n -> (Show (Lists n a) => r) -> r
showLists _ SZ      k = k
showLists p (SS' n) k = showLists p n k

showConstructor :: Int -> SNat n -> ShowS
showConstructor p sn
  | n' == 0            = showString "scalar"
  | 1 <= n' && n' <= 9 = showString "dim" . shows n'
  | otherwise          = showString "fromLists @"
                       . explicitShowsPrec p (snatToNat sn)
  where
    n' :: Natural
    n' = snatToNatural sn

instance Show a => Show (Tensor n a) where
  showsPrec p tensor = showLists (Proxy @a) (tensorSNat tensor) $
      showParen (p >= appPrec1) $
          showConstructor appPrec1 (tensorSNat tensor)
        . showSpace
        . showsPrec appPrec1 (toLists tensor)

{-------------------------------------------------------------------------------
  Internal auxiliary: SNat
-------------------------------------------------------------------------------}

tensorSNatI :: Tensor n a -> (SNatI n => r) -> r
tensorSNatI (Scalar _)  k = k
tensorSNatI (Tensor xs) k = tensorSNatI (L.head xs) k

tensorSNat :: Tensor n a -> SNat n
tensorSNat tensor = tensorSNatI tensor snat

{-------------------------------------------------------------------------------
  Internal auxiliary: lists
-------------------------------------------------------------------------------}

-- | The first @r@ sublists of length @n@
--
-- >    consecutive 4 3 [1..6]
-- > == [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
consecutive :: Int -> Int -> [a] -> [[a]]
consecutive r n = L.take r . L.map (L.take n) . L.tails

-- | Every nth element of the list
--
-- Examples
--
-- > everyNth 1 [0..9] == [0,2,3,4,5,6,7,8,9]
-- > everyNth 2 [0..9] == [0,2,4,6,8]
-- > everyNth 3 [0..9] == [0,3,6,9]
everyNth :: forall a. Int -> [a] -> [a]
everyNth n = \xs ->
    if n > 0
      then go xs
      else error "everyNth: n should be strictly positive"
  where
    go :: [a] -> [a]
    go []     = []
    go (x:xs) = x : go (drop (n - 1) xs)

-- | Single out an element from the list
--
-- >    pickOne [1..4]
-- > == [ ( []      , 1 , [2,3,4] )
-- >    , ( [1]     , 2 , [3,4]   )
-- >    , ( [1,2]   , 3 , [4]     )
-- >    , ( [1,2,3] , 4 , []      )
-- >    ]
pickOne :: forall a. [a] -> [([a], a, [a])]
pickOne = \case
    []   -> error "pickOne: empty list"
    x:xs -> go [] x xs
  where
    go :: [a] -> a -> [a] -> [([a], a, [a])]
    go acc x []     = [(reverse acc, x, [])]
    go acc x (y:ys) = (reverse acc, x, (y:ys)) : go (x:acc) y ys

-- | Distribute @f@ over @[]@
distribList :: Functor f => Int -> f [a] -> [f a]
distribList 0 _   = []
distribList n fxs = (head <$> fxs) : distribList (n - 1) (tail <$> fxs)
