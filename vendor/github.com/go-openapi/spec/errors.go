package spec

import "errors"

// Error codes
var (
	// ErrUnknownTypeForReference indicates that a resolved reference was found in an unsupported container type
	ErrUnknownTypeForReference = errors.New("unknown type for the resolved reference")

	// ErrResolveRefNeedsAPointer indicates that a $ref target must be a valid JSON pointer
	ErrResolveRefNeedsAPointer = errors.New("resolve ref: target needs to be a pointer")

	// ErrDerefUnsupportedType indicates that a resolved reference was found in an unsupported container type.
	// At the moment, $ref are supported only inside: schemas, parameters, responses, path items
	ErrDerefUnsupportedType = errors.New("deref: unsupported type")

	// ErrExpandUnsupportedType indicates that $ref expansion is attempted on some invalid type
	ErrExpandUnsupportedType = errors.New("expand: unsupported type. Input should be of type *Parameter or *Response")
)
