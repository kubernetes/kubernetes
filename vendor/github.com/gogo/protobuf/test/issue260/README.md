# The Bug

If in a message the following options are set:

* `typedecl` `false`
* `go_getters` `false`
* `marshaller` `true`

And one of the fields is using the `stdtime` and `nullable` `false` extension (to
use `time.Time` instead of the protobuf type), then an import to the _time_ package
is added even if it is not needed.
