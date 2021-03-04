## 1.1.2

* Fix error when decode hook decodes interface implementation into interface
  type. [GH-140]

## 1.1.1

* Fix panic that can happen in `decodePtr`

## 1.1.0

* Added `StringToIPHookFunc` to convert `string` to `net.IP` and `net.IPNet` [GH-133]
* Support struct to struct decoding [GH-137]
* If source map value is nil, then destination map value is nil (instead of empty)
* If source slice value is nil, then destination slice value is nil (instead of empty)
* If source pointer is nil, then destination pointer is set to nil (instead of
  allocated zero value of type)

## 1.0.0

* Initial tagged stable release.
