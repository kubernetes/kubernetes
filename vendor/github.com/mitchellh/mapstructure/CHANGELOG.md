## 1.4.3

* Fix cases where `json.Number` didn't decode properly [GH-261]

## 1.4.2

* Custom name matchers to support any sort of casing, formatting, etc. for
  field names. [GH-250]
* Fix possible panic in ComposeDecodeHookFunc [GH-251]

## 1.4.1

* Fix regression where `*time.Time` value would be set to empty and not be sent
  to decode hooks properly [GH-232]

## 1.4.0

* A new decode hook type `DecodeHookFuncValue` has been added that has
  access to the full values. [GH-183]
* Squash is now supported with embedded fields that are struct pointers [GH-205]
* Empty strings will convert to 0 for all numeric types when weakly decoding [GH-206]

## 1.3.3

* Decoding maps from maps creates a settable value for decode hooks [GH-203]

## 1.3.2

* Decode into interface type with a struct value is supported [GH-187]

## 1.3.1

* Squash should only squash embedded structs. [GH-194]

## 1.3.0

* Added `",omitempty"` support. This will ignore zero values in the source
  structure when encoding. [GH-145]

## 1.2.3

* Fix duplicate entries in Keys list with pointer values. [GH-185]

## 1.2.2

* Do not add unsettable (unexported) values to the unused metadata key
  or "remain" value. [GH-150]

## 1.2.1

* Go modules checksum mismatch fix

## 1.2.0

* Added support to capture unused values in a field using the `",remain"` value
  in the mapstructure tag. There is an example to showcase usage.
* Added `DecoderConfig` option to always squash embedded structs
* `json.Number` can decode into `uint` types
* Empty slices are preserved and not replaced with nil slices
* Fix panic that can occur in when decoding a map into a nil slice of structs
* Improved package documentation for godoc

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
