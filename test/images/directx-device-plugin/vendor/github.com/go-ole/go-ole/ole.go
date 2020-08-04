package ole

import (
	"fmt"
	"strings"
)

// DISPPARAMS are the arguments that passed to methods or property.
type DISPPARAMS struct {
	rgvarg            uintptr
	rgdispidNamedArgs uintptr
	cArgs             uint32
	cNamedArgs        uint32
}

// EXCEPINFO defines exception info.
type EXCEPINFO struct {
	wCode             uint16
	wReserved         uint16
	bstrSource        *uint16
	bstrDescription   *uint16
	bstrHelpFile      *uint16
	dwHelpContext     uint32
	pvReserved        uintptr
	pfnDeferredFillIn uintptr
	scode             uint32
}

// WCode return wCode in EXCEPINFO.
func (e EXCEPINFO) WCode() uint16 {
	return e.wCode
}

// SCODE return scode in EXCEPINFO.
func (e EXCEPINFO) SCODE() uint32 {
	return e.scode
}

// String convert EXCEPINFO to string.
func (e EXCEPINFO) String() string {
	var src, desc, hlp string
	if e.bstrSource == nil {
		src = "<nil>"
	} else {
		src = BstrToString(e.bstrSource)
	}

	if e.bstrDescription == nil {
		desc = "<nil>"
	} else {
		desc = BstrToString(e.bstrDescription)
	}

	if e.bstrHelpFile == nil {
		hlp = "<nil>"
	} else {
		hlp = BstrToString(e.bstrHelpFile)
	}

	return fmt.Sprintf(
		"wCode: %#x, bstrSource: %v, bstrDescription: %v, bstrHelpFile: %v, dwHelpContext: %#x, scode: %#x",
		e.wCode, src, desc, hlp, e.dwHelpContext, e.scode,
	)
}

// Error implements error interface and returns error string.
func (e EXCEPINFO) Error() string {
	if e.bstrDescription != nil {
		return strings.TrimSpace(BstrToString(e.bstrDescription))
	}

	src := "Unknown"
	if e.bstrSource != nil {
		src = BstrToString(e.bstrSource)
	}

	code := e.scode
	if e.wCode != 0 {
		code = uint32(e.wCode)
	}

	return fmt.Sprintf("%v: %#x", src, code)
}

// PARAMDATA defines parameter data type.
type PARAMDATA struct {
	Name *int16
	Vt   uint16
}

// METHODDATA defines method info.
type METHODDATA struct {
	Name     *uint16
	Data     *PARAMDATA
	Dispid   int32
	Meth     uint32
	CC       int32
	CArgs    uint32
	Flags    uint16
	VtReturn uint32
}

// INTERFACEDATA defines interface info.
type INTERFACEDATA struct {
	MethodData *METHODDATA
	CMembers   uint32
}

// Point is 2D vector type.
type Point struct {
	X int32
	Y int32
}

// Msg is message between processes.
type Msg struct {
	Hwnd    uint32
	Message uint32
	Wparam  int32
	Lparam  int32
	Time    uint32
	Pt      Point
}

// TYPEDESC defines data type.
type TYPEDESC struct {
	Hreftype uint32
	VT       uint16
}

// IDLDESC defines IDL info.
type IDLDESC struct {
	DwReserved uint32
	WIDLFlags  uint16
}

// TYPEATTR defines type info.
type TYPEATTR struct {
	Guid             GUID
	Lcid             uint32
	dwReserved       uint32
	MemidConstructor int32
	MemidDestructor  int32
	LpstrSchema      *uint16
	CbSizeInstance   uint32
	Typekind         int32
	CFuncs           uint16
	CVars            uint16
	CImplTypes       uint16
	CbSizeVft        uint16
	CbAlignment      uint16
	WTypeFlags       uint16
	WMajorVerNum     uint16
	WMinorVerNum     uint16
	TdescAlias       TYPEDESC
	IdldescType      IDLDESC
}
