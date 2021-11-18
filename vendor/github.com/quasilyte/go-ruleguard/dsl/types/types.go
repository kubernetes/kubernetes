// Package types mimics the https://golang.org/pkg/go/types/ package.
// It also contains some extra utility functions, they're defined in ext.go file.
package types

// Implements reports whether a given type implements the specified interface.
func Implements(typ Type, iface *Interface) bool { return false }

// Identical reports whether x and y are identical types. Receivers of Signature types are ignored.
func Identical(x, y Type) bool { return false }

// A Type represents a type of Go. All types implement the Type interface.
type Type interface {
	// Underlying returns the underlying type of a type.
	Underlying() Type

	// String returns a string representation of a type.
	String() string
}

type (
	// An Array represents an array type.
	Array struct{}

	// A Slice represents a slice type.
	Slice struct{}

	// A Pointer represents a pointer type.
	Pointer struct{}

	// An Interface represents an interface type.
	Interface struct{}
)

// NewArray returns a new array type for the given element type and length.
// A negative length indicates an unknown length.
func NewArray(elem Type, len int) *Array { return nil }

// Elem returns element type of array.
func (*Array) Elem() Type { return nil }

// NewSlice returns a new slice type for the given element type.
func NewSlice(elem Type) *Slice { return nil }

// Elem returns element type of slice.
func (*Slice) Elem() Type { return nil }

// Len returns the length of array.
// A negative result indicates an unknown length.
func (*Array) Len() int { return 0 }

// NewPointer returns a new pointer type for the given element (base) type.
func NewPointer(elem Type) *Pointer { return nil }

// Elem returns the element type for the given pointer.
func (*Pointer) Elem() Type { return nil }
