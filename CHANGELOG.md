# Revision history for tmp

## 0.2.0 -- 2025-08-27

* Improve docs
* Make tensors with empty dimensions `Show`able (such tensors are invalid, but
  if they are not showable, it's harder to debug code that generates them)
* Add some `HasCallStack` constraints
* Add `zipEach`
* Rename `foreach`/`foreachWith` to `forEach`/`forEachWith` for consistency

## 0.1.2 -- 2025-08-27

* Add `subsWithStride`

## 0.1.1 -- 2025-08-27

* Add missing instances for `TestValue`
* Document `Tensor` invariants (#11)

## 0.1.0 -- 2025-02-15

* First release
