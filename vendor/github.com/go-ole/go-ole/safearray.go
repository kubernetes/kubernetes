// Package is meant to retrieve and process safe array data returned from COM.

package ole

// SafeArrayBound defines the SafeArray boundaries.
type SafeArrayBound struct {
	Elements   uint32
	LowerBound int32
}

// SafeArray is how COM handles arrays.
type SafeArray struct {
	Dimensions   uint16
	FeaturesFlag uint16
	ElementsSize uint32
	LocksAmount  uint32
	Data         uint32
	Bounds       [16]byte
}

// SAFEARRAY is obsolete, exists for backwards compatibility.
// Use SafeArray
type SAFEARRAY SafeArray

// SAFEARRAYBOUND is obsolete, exists for backwards compatibility.
// Use SafeArrayBound
type SAFEARRAYBOUND SafeArrayBound
