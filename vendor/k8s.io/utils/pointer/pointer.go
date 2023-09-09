/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Deprecated: Use functions in k8s.io/utils/ptr instead: ptr.To to obtain
// a pointer, ptr.Deref to dereference a pointer, ptr.Equal to compare
// dereferenced pointers.
package pointer

import (
	"time"

	"k8s.io/utils/ptr"
)

// AllPtrFieldsNil tests whether all pointer fields in a struct are nil.  This is useful when,
// for example, an API struct is handled by plugins which need to distinguish
// "no plugin accepted this spec" from "this spec is empty".
//
// This function is only valid for structs and pointers to structs.  Any other
// type will cause a panic.  Passing a typed nil pointer will return true.
//
// Deprecated: Use ptr.AllPtrFieldsNil instead.
var AllPtrFieldsNil = ptr.AllPtrFieldsNil

// Int returns a pointer to an int.
var Int = ptr.To[int]

// IntPtr is a function variable referring to Int.
//
// Deprecated: Use ptr.To instead.
var IntPtr = Int // for back-compat

// IntDeref dereferences the int ptr and returns it if not nil, or else
// returns def.
var IntDeref = ptr.Deref[int]

// IntPtrDerefOr is a function variable referring to IntDeref.
//
// Deprecated: Use ptr.Deref instead.
var IntPtrDerefOr = IntDeref // for back-compat

// Int32 returns a pointer to an int32.
var Int32 = ptr.To[int32]

// Int32Ptr is a function variable referring to Int32.
//
// Deprecated: Use ptr.To instead.
var Int32Ptr = Int32 // for back-compat

// Int32Deref dereferences the int32 ptr and returns it if not nil, or else
// returns def.
var Int32Deref = ptr.Deref[int32]

// Int32PtrDerefOr is a function variable referring to Int32Deref.
//
// Deprecated: Use ptr.Deref instead.
var Int32PtrDerefOr = Int32Deref // for back-compat

// Int32Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Int32Equal = ptr.Equal[int32]

// Uint returns a pointer to an uint
var Uint = ptr.To[uint]

// UintPtr is a function variable referring to Uint.
//
// Deprecated: Use ptr.To instead.
var UintPtr = Uint // for back-compat

// UintDeref dereferences the uint ptr and returns it if not nil, or else
// returns def.
var UintDeref = ptr.Deref[uint]

// UintPtrDerefOr is a function variable referring to UintDeref.
//
// Deprecated: Use ptr.Deref instead.
var UintPtrDerefOr = UintDeref // for back-compat

// Uint32 returns a pointer to an uint32.
var Uint32 = ptr.To[uint32]

// Uint32Ptr is a function variable referring to Uint32.
//
// Deprecated: Use ptr.To instead.
var Uint32Ptr = Uint32 // for back-compat

// Uint32Deref dereferences the uint32 ptr and returns it if not nil, or else
// returns def.
var Uint32Deref = ptr.Deref[uint32]

// Uint32PtrDerefOr is a function variable referring to Uint32Deref.
//
// Deprecated: Use ptr.Deref instead.
var Uint32PtrDerefOr = Uint32Deref // for back-compat

// Uint32Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Uint32Equal = ptr.Equal[uint32]

// Int64 returns a pointer to an int64.
var Int64 = ptr.To[int64]

// Int64Ptr is a function variable referring to Int64.
//
// Deprecated: Use ptr.To instead.
var Int64Ptr = Int64 // for back-compat

// Int64Deref dereferences the int64 ptr and returns it if not nil, or else
// returns def.
var Int64Deref = ptr.Deref[int64]

// Int64PtrDerefOr is a function variable referring to Int64Deref.
//
// Deprecated: Use ptr.Deref instead.
var Int64PtrDerefOr = Int64Deref // for back-compat

// Int64Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Int64Equal = ptr.Equal[int64]

// Uint64 returns a pointer to an uint64.
var Uint64 = ptr.To[uint64]

// Uint64Ptr is a function variable referring to Uint64.
//
// Deprecated: Use ptr.To instead.
var Uint64Ptr = Uint64 // for back-compat

// Uint64Deref dereferences the uint64 ptr and returns it if not nil, or else
// returns def.
var Uint64Deref = ptr.Deref[uint64]

// Uint64PtrDerefOr is a function variable referring to Uint64Deref.
//
// Deprecated: Use ptr.Deref instead.
var Uint64PtrDerefOr = Uint64Deref // for back-compat

// Uint64Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Uint64Equal = ptr.Equal[uint64]

// Bool returns a pointer to a bool.
var Bool = ptr.To[bool]

// BoolPtr is a function variable referring to Bool.
//
// Deprecated: Use ptr.To instead.
var BoolPtr = Bool // for back-compat

// BoolDeref dereferences the bool ptr and returns it if not nil, or else
// returns def.
var BoolDeref = ptr.Deref[bool]

// BoolPtrDerefOr is a function variable referring to BoolDeref.
//
// Deprecated: Use ptr.Deref instead.
var BoolPtrDerefOr = BoolDeref // for back-compat

// BoolEqual returns true if both arguments are nil or both arguments
// dereference to the same value.
var BoolEqual = ptr.Equal[bool]

// String returns a pointer to a string.
var String = ptr.To[string]

// StringPtr is a function variable referring to String.
//
// Deprecated: Use ptr.To instead.
var StringPtr = String // for back-compat

// StringDeref dereferences the string ptr and returns it if not nil, or else
// returns def.
var StringDeref = ptr.Deref[string]

// StringPtrDerefOr is a function variable referring to StringDeref.
//
// Deprecated: Use ptr.Deref instead.
var StringPtrDerefOr = StringDeref // for back-compat

// StringEqual returns true if both arguments are nil or both arguments
// dereference to the same value.
var StringEqual = ptr.Equal[string]

// Float32 returns a pointer to a float32.
var Float32 = ptr.To[float32]

// Float32Ptr is a function variable referring to Float32.
//
// Deprecated: Use ptr.To instead.
var Float32Ptr = Float32

// Float32Deref dereferences the float32 ptr and returns it if not nil, or else
// returns def.
var Float32Deref = ptr.Deref[float32]

// Float32PtrDerefOr is a function variable referring to Float32Deref.
//
// Deprecated: Use ptr.Deref instead.
var Float32PtrDerefOr = Float32Deref // for back-compat

// Float32Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Float32Equal = ptr.Equal[float32]

// Float64 returns a pointer to a float64.
var Float64 = ptr.To[float64]

// Float64Ptr is a function variable referring to Float64.
//
// Deprecated: Use ptr.To instead.
var Float64Ptr = Float64

// Float64Deref dereferences the float64 ptr and returns it if not nil, or else
// returns def.
var Float64Deref = ptr.Deref[float64]

// Float64PtrDerefOr is a function variable referring to Float64Deref.
//
// Deprecated: Use ptr.Deref instead.
var Float64PtrDerefOr = Float64Deref // for back-compat

// Float64Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
var Float64Equal = ptr.Equal[float64]

// Duration returns a pointer to a time.Duration.
var Duration = ptr.To[time.Duration]

// DurationDeref dereferences the time.Duration ptr and returns it if not nil, or else
// returns def.
var DurationDeref = ptr.Deref[time.Duration]

// DurationEqual returns true if both arguments are nil or both arguments
// dereference to the same value.
var DurationEqual = ptr.Equal[time.Duration]
