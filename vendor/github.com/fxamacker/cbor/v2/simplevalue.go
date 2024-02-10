package cbor

import "reflect"

// SimpleValue represents CBOR simple value.
// CBOR simple value is:
// * an extension point like CBOR tag.
// * a subset of CBOR major type 7 that isn't floating-point.
// * "identified by a number between 0 and 255, but distinct from that number itself".
//   For example, "a simple value 2 is not equivalent to an integer 2" as a CBOR map key.
// CBOR simple values identified by 20..23 are: "false", "true" , "null", and "undefined".
// Other CBOR simple values are currently unassigned/reserved by IANA.
type SimpleValue uint8

var (
	typeSimpleValue = reflect.TypeOf(SimpleValue(0))
)
