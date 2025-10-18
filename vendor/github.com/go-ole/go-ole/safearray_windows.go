// +build windows

package ole

import (
	"unsafe"
)

var (
	procSafeArrayAccessData        = modoleaut32.NewProc("SafeArrayAccessData")
	procSafeArrayAllocData         = modoleaut32.NewProc("SafeArrayAllocData")
	procSafeArrayAllocDescriptor   = modoleaut32.NewProc("SafeArrayAllocDescriptor")
	procSafeArrayAllocDescriptorEx = modoleaut32.NewProc("SafeArrayAllocDescriptorEx")
	procSafeArrayCopy              = modoleaut32.NewProc("SafeArrayCopy")
	procSafeArrayCopyData          = modoleaut32.NewProc("SafeArrayCopyData")
	procSafeArrayCreate            = modoleaut32.NewProc("SafeArrayCreate")
	procSafeArrayCreateEx          = modoleaut32.NewProc("SafeArrayCreateEx")
	procSafeArrayCreateVector      = modoleaut32.NewProc("SafeArrayCreateVector")
	procSafeArrayCreateVectorEx    = modoleaut32.NewProc("SafeArrayCreateVectorEx")
	procSafeArrayDestroy           = modoleaut32.NewProc("SafeArrayDestroy")
	procSafeArrayDestroyData       = modoleaut32.NewProc("SafeArrayDestroyData")
	procSafeArrayDestroyDescriptor = modoleaut32.NewProc("SafeArrayDestroyDescriptor")
	procSafeArrayGetDim            = modoleaut32.NewProc("SafeArrayGetDim")
	procSafeArrayGetElement        = modoleaut32.NewProc("SafeArrayGetElement")
	procSafeArrayGetElemsize       = modoleaut32.NewProc("SafeArrayGetElemsize")
	procSafeArrayGetIID            = modoleaut32.NewProc("SafeArrayGetIID")
	procSafeArrayGetLBound         = modoleaut32.NewProc("SafeArrayGetLBound")
	procSafeArrayGetUBound         = modoleaut32.NewProc("SafeArrayGetUBound")
	procSafeArrayGetVartype        = modoleaut32.NewProc("SafeArrayGetVartype")
	procSafeArrayLock              = modoleaut32.NewProc("SafeArrayLock")
	procSafeArrayPtrOfIndex        = modoleaut32.NewProc("SafeArrayPtrOfIndex")
	procSafeArrayUnaccessData      = modoleaut32.NewProc("SafeArrayUnaccessData")
	procSafeArrayUnlock            = modoleaut32.NewProc("SafeArrayUnlock")
	procSafeArrayPutElement        = modoleaut32.NewProc("SafeArrayPutElement")
	//procSafeArrayRedim             = modoleaut32.NewProc("SafeArrayRedim") // TODO
	//procSafeArraySetIID            = modoleaut32.NewProc("SafeArraySetIID") // TODO
	procSafeArrayGetRecordInfo = modoleaut32.NewProc("SafeArrayGetRecordInfo")
	procSafeArraySetRecordInfo = modoleaut32.NewProc("SafeArraySetRecordInfo")
)

// safeArrayAccessData returns raw array pointer.
//
// AKA: SafeArrayAccessData in Windows API.
// Todo: Test
func safeArrayAccessData(safearray *SafeArray) (element uintptr, err error) {
	err = convertHresultToError(
		procSafeArrayAccessData.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&element))))
	return
}

// safeArrayUnaccessData releases raw array.
//
// AKA: SafeArrayUnaccessData in Windows API.
func safeArrayUnaccessData(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayUnaccessData.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayAllocData allocates SafeArray.
//
// AKA: SafeArrayAllocData in Windows API.
func safeArrayAllocData(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayAllocData.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayAllocDescriptor allocates SafeArray.
//
// AKA: SafeArrayAllocDescriptor in Windows API.
func safeArrayAllocDescriptor(dimensions uint32) (safearray *SafeArray, err error) {
	err = convertHresultToError(
		procSafeArrayAllocDescriptor.Call(uintptr(dimensions), uintptr(unsafe.Pointer(&safearray))))
	return
}

// safeArrayAllocDescriptorEx allocates SafeArray.
//
// AKA: SafeArrayAllocDescriptorEx in Windows API.
func safeArrayAllocDescriptorEx(variantType VT, dimensions uint32) (safearray *SafeArray, err error) {
	err = convertHresultToError(
		procSafeArrayAllocDescriptorEx.Call(
			uintptr(variantType),
			uintptr(dimensions),
			uintptr(unsafe.Pointer(&safearray))))
	return
}

// safeArrayCopy returns copy of SafeArray.
//
// AKA: SafeArrayCopy in Windows API.
func safeArrayCopy(original *SafeArray) (safearray *SafeArray, err error) {
	err = convertHresultToError(
		procSafeArrayCopy.Call(
			uintptr(unsafe.Pointer(original)),
			uintptr(unsafe.Pointer(&safearray))))
	return
}

// safeArrayCopyData duplicates SafeArray into another SafeArray object.
//
// AKA: SafeArrayCopyData in Windows API.
func safeArrayCopyData(original *SafeArray, duplicate *SafeArray) (err error) {
	err = convertHresultToError(
		procSafeArrayCopyData.Call(
			uintptr(unsafe.Pointer(original)),
			uintptr(unsafe.Pointer(duplicate))))
	return
}

// safeArrayCreate creates SafeArray.
//
// AKA: SafeArrayCreate in Windows API.
func safeArrayCreate(variantType VT, dimensions uint32, bounds *SafeArrayBound) (safearray *SafeArray, err error) {
	sa, _, err := procSafeArrayCreate.Call(
		uintptr(variantType),
		uintptr(dimensions),
		uintptr(unsafe.Pointer(bounds)))
	safearray = (*SafeArray)(unsafe.Pointer(&sa))
	return
}

// safeArrayCreateEx creates SafeArray.
//
// AKA: SafeArrayCreateEx in Windows API.
func safeArrayCreateEx(variantType VT, dimensions uint32, bounds *SafeArrayBound, extra uintptr) (safearray *SafeArray, err error) {
	sa, _, err := procSafeArrayCreateEx.Call(
		uintptr(variantType),
		uintptr(dimensions),
		uintptr(unsafe.Pointer(bounds)),
		extra)
	safearray = (*SafeArray)(unsafe.Pointer(sa))
	return
}

// safeArrayCreateVector creates SafeArray.
//
// AKA: SafeArrayCreateVector in Windows API.
func safeArrayCreateVector(variantType VT, lowerBound int32, length uint32) (safearray *SafeArray, err error) {
	sa, _, err := procSafeArrayCreateVector.Call(
		uintptr(variantType),
		uintptr(lowerBound),
		uintptr(length))
	safearray = (*SafeArray)(unsafe.Pointer(sa))
	return
}

// safeArrayCreateVectorEx creates SafeArray.
//
// AKA: SafeArrayCreateVectorEx in Windows API.
func safeArrayCreateVectorEx(variantType VT, lowerBound int32, length uint32, extra uintptr) (safearray *SafeArray, err error) {
	sa, _, err := procSafeArrayCreateVectorEx.Call(
		uintptr(variantType),
		uintptr(lowerBound),
		uintptr(length),
		extra)
	safearray = (*SafeArray)(unsafe.Pointer(sa))
	return
}

// safeArrayDestroy destroys SafeArray object.
//
// AKA: SafeArrayDestroy in Windows API.
func safeArrayDestroy(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayDestroy.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayDestroyData destroys SafeArray object.
//
// AKA: SafeArrayDestroyData in Windows API.
func safeArrayDestroyData(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayDestroyData.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayDestroyDescriptor destroys SafeArray object.
//
// AKA: SafeArrayDestroyDescriptor in Windows API.
func safeArrayDestroyDescriptor(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayDestroyDescriptor.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayGetDim is the amount of dimensions in the SafeArray.
//
// SafeArrays may have multiple dimensions. Meaning, it could be
// multidimensional array.
//
// AKA: SafeArrayGetDim in Windows API.
func safeArrayGetDim(safearray *SafeArray) (dimensions *uint32, err error) {
	l, _, err := procSafeArrayGetDim.Call(uintptr(unsafe.Pointer(safearray)))
	dimensions = (*uint32)(unsafe.Pointer(l))
	return
}

// safeArrayGetElementSize is the element size in bytes.
//
// AKA: SafeArrayGetElemsize in Windows API.
func safeArrayGetElementSize(safearray *SafeArray) (length *uint32, err error) {
	l, _, err := procSafeArrayGetElemsize.Call(uintptr(unsafe.Pointer(safearray)))
	length = (*uint32)(unsafe.Pointer(l))
	return
}

// safeArrayGetElement retrieves element at given index.
func safeArrayGetElement(safearray *SafeArray, index int32, pv unsafe.Pointer) error {
	return convertHresultToError(
		procSafeArrayGetElement.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&index)),
			uintptr(pv)))
}

// safeArrayGetElementString retrieves element at given index and converts to string.
func safeArrayGetElementString(safearray *SafeArray, index int32) (str string, err error) {
	var element *int16
	err = convertHresultToError(
		procSafeArrayGetElement.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&index)),
			uintptr(unsafe.Pointer(&element))))
	str = BstrToString(*(**uint16)(unsafe.Pointer(&element)))
	SysFreeString(element)
	return
}

// safeArrayGetIID is the InterfaceID of the elements in the SafeArray.
//
// AKA: SafeArrayGetIID in Windows API.
func safeArrayGetIID(safearray *SafeArray) (guid *GUID, err error) {
	err = convertHresultToError(
		procSafeArrayGetIID.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&guid))))
	return
}

// safeArrayGetLBound returns lower bounds of SafeArray.
//
// SafeArrays may have multiple dimensions. Meaning, it could be
// multidimensional array.
//
// AKA: SafeArrayGetLBound in Windows API.
func safeArrayGetLBound(safearray *SafeArray, dimension uint32) (lowerBound int32, err error) {
	err = convertHresultToError(
		procSafeArrayGetLBound.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(dimension),
			uintptr(unsafe.Pointer(&lowerBound))))
	return
}

// safeArrayGetUBound returns upper bounds of SafeArray.
//
// SafeArrays may have multiple dimensions. Meaning, it could be
// multidimensional array.
//
// AKA: SafeArrayGetUBound in Windows API.
func safeArrayGetUBound(safearray *SafeArray, dimension uint32) (upperBound int32, err error) {
	err = convertHresultToError(
		procSafeArrayGetUBound.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(dimension),
			uintptr(unsafe.Pointer(&upperBound))))
	return
}

// safeArrayGetVartype returns data type of SafeArray.
//
// AKA: SafeArrayGetVartype in Windows API.
func safeArrayGetVartype(safearray *SafeArray) (varType uint16, err error) {
	err = convertHresultToError(
		procSafeArrayGetVartype.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&varType))))
	return
}

// safeArrayLock locks SafeArray for reading to modify SafeArray.
//
// This must be called during some calls to ensure that another process does not
// read or write to the SafeArray during editing.
//
// AKA: SafeArrayLock in Windows API.
func safeArrayLock(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayLock.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayUnlock unlocks SafeArray for reading.
//
// AKA: SafeArrayUnlock in Windows API.
func safeArrayUnlock(safearray *SafeArray) (err error) {
	err = convertHresultToError(procSafeArrayUnlock.Call(uintptr(unsafe.Pointer(safearray))))
	return
}

// safeArrayPutElement stores the data element at the specified location in the
// array.
//
// AKA: SafeArrayPutElement in Windows API.
func safeArrayPutElement(safearray *SafeArray, index int64, element uintptr) (err error) {
	err = convertHresultToError(
		procSafeArrayPutElement.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&index)),
			uintptr(unsafe.Pointer(element))))
	return
}

// safeArrayGetRecordInfo accesses IRecordInfo info for custom types.
//
// AKA: SafeArrayGetRecordInfo in Windows API.
//
// XXX: Must implement IRecordInfo interface for this to return.
func safeArrayGetRecordInfo(safearray *SafeArray) (recordInfo interface{}, err error) {
	err = convertHresultToError(
		procSafeArrayGetRecordInfo.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&recordInfo))))
	return
}

// safeArraySetRecordInfo mutates IRecordInfo info for custom types.
//
// AKA: SafeArraySetRecordInfo in Windows API.
//
// XXX: Must implement IRecordInfo interface for this to return.
func safeArraySetRecordInfo(safearray *SafeArray, recordInfo interface{}) (err error) {
	err = convertHresultToError(
		procSafeArraySetRecordInfo.Call(
			uintptr(unsafe.Pointer(safearray)),
			uintptr(unsafe.Pointer(&recordInfo))))
	return
}
