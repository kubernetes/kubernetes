package ole

import "unsafe"

// NewVariant returns new variant based on type and value.
func NewVariant(vt VT, val int64) VARIANT {
	return VARIANT{VT: vt, Val: val}
}

// ToIUnknown converts Variant to Unknown object.
func (v *VARIANT) ToIUnknown() *IUnknown {
	if v.VT != VT_UNKNOWN {
		return nil
	}
	return (*IUnknown)(unsafe.Pointer(uintptr(v.Val)))
}

// ToIDispatch converts variant to dispatch object.
func (v *VARIANT) ToIDispatch() *IDispatch {
	if v.VT != VT_DISPATCH {
		return nil
	}
	return (*IDispatch)(unsafe.Pointer(uintptr(v.Val)))
}

// ToArray converts variant to SafeArray helper.
func (v *VARIANT) ToArray() *SafeArrayConversion {
	if v.VT != VT_SAFEARRAY {
		if v.VT&VT_ARRAY == 0 {
			return nil
		}
	}
	var safeArray *SafeArray = (*SafeArray)(unsafe.Pointer(uintptr(v.Val)))
	return &SafeArrayConversion{safeArray}
}

// ToString converts variant to Go string.
func (v *VARIANT) ToString() string {
	if v.VT != VT_BSTR {
		return ""
	}
	return BstrToString(*(**uint16)(unsafe.Pointer(&v.Val)))
}

// Clear the memory of variant object.
func (v *VARIANT) Clear() error {
	return VariantClear(v)
}

// Value returns variant value based on its type.
//
// Currently supported types: 2- and 4-byte integers, strings, bools.
// Note that 64-bit integers, datetimes, and other types are stored as strings
// and will be returned as strings.
//
// Needs to be further converted, because this returns an interface{}.
func (v *VARIANT) Value() interface{} {
	switch v.VT {
	case VT_I1:
		return int8(v.Val)
	case VT_UI1:
		return uint8(v.Val)
	case VT_I2:
		return int16(v.Val)
	case VT_UI2:
		return uint16(v.Val)
	case VT_I4:
		return int32(v.Val)
	case VT_UI4:
		return uint32(v.Val)
	case VT_I8:
		return int64(v.Val)
	case VT_UI8:
		return uint64(v.Val)
	case VT_INT:
		return int(v.Val)
	case VT_UINT:
		return uint(v.Val)
	case VT_INT_PTR:
		return uintptr(v.Val) // TODO
	case VT_UINT_PTR:
		return uintptr(v.Val)
	case VT_R4:
		return *(*float32)(unsafe.Pointer(&v.Val))
	case VT_R8:
		return *(*float64)(unsafe.Pointer(&v.Val))
	case VT_BSTR:
		return v.ToString()
	case VT_DATE:
		// VT_DATE type will either return float64 or time.Time.
		d := uint64(v.Val)
		date, err := GetVariantDate(d)
		if err != nil {
			return float64(v.Val)
		}
		return date
	case VT_UNKNOWN:
		return v.ToIUnknown()
	case VT_DISPATCH:
		return v.ToIDispatch()
	case VT_BOOL:
		return (v.Val & 0xffff) != 0
	}
	return nil
}
