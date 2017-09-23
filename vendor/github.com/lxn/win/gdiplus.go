// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

type GpStatus int32

const (
	Ok                        GpStatus = 0
	GenericError              GpStatus = 1
	InvalidParameter          GpStatus = 2
	OutOfMemory               GpStatus = 3
	ObjectBusy                GpStatus = 4
	InsufficientBuffer        GpStatus = 5
	NotImplemented            GpStatus = 6
	Win32Error                GpStatus = 7
	WrongState                GpStatus = 8
	Aborted                   GpStatus = 9
	FileNotFound              GpStatus = 10
	ValueOverflow             GpStatus = 11
	AccessDenied              GpStatus = 12
	UnknownImageFormat        GpStatus = 13
	FontFamilyNotFound        GpStatus = 14
	FontStyleNotFound         GpStatus = 15
	NotTrueTypeFont           GpStatus = 16
	UnsupportedGdiplusVersion GpStatus = 17
	GdiplusNotInitialized     GpStatus = 18
	PropertyNotFound          GpStatus = 19
	PropertyNotSupported      GpStatus = 20
	ProfileNotFound           GpStatus = 21
)

func (s GpStatus) String() string {
	switch s {
	case Ok:
		return "Ok"

	case GenericError:
		return "GenericError"

	case InvalidParameter:
		return "InvalidParameter"

	case OutOfMemory:
		return "OutOfMemory"

	case ObjectBusy:
		return "ObjectBusy"

	case InsufficientBuffer:
		return "InsufficientBuffer"

	case NotImplemented:
		return "NotImplemented"

	case Win32Error:
		return "Win32Error"

	case WrongState:
		return "WrongState"

	case Aborted:
		return "Aborted"

	case FileNotFound:
		return "FileNotFound"

	case ValueOverflow:
		return "ValueOverflow"

	case AccessDenied:
		return "AccessDenied"

	case UnknownImageFormat:
		return "UnknownImageFormat"

	case FontFamilyNotFound:
		return "FontFamilyNotFound"

	case FontStyleNotFound:
		return "FontStyleNotFound"

	case NotTrueTypeFont:
		return "NotTrueTypeFont"

	case UnsupportedGdiplusVersion:
		return "UnsupportedGdiplusVersion"

	case GdiplusNotInitialized:
		return "GdiplusNotInitialized"

	case PropertyNotFound:
		return "PropertyNotFound"

	case PropertyNotSupported:
		return "PropertyNotSupported"

	case ProfileNotFound:
		return "ProfileNotFound"
	}

	return "Unknown Status Value"
}

type GdiplusStartupInput struct {
	GdiplusVersion           uint32
	DebugEventCallback       uintptr
	SuppressBackgroundThread BOOL
	SuppressExternalCodecs   BOOL
}

type GdiplusStartupOutput struct {
	NotificationHook   uintptr
	NotificationUnhook uintptr
}

type GpImage struct{}

type GpBitmap GpImage

type ARGB uint32

var (
	// Library
	libgdiplus uintptr

	// Functions
	gdipCreateBitmapFromFile    uintptr
	gdipCreateBitmapFromHBITMAP uintptr
	gdipCreateHBITMAPFromBitmap uintptr
	gdipDisposeImage            uintptr
	gdiplusShutdown             uintptr
	gdiplusStartup              uintptr
)

var (
	token uintptr
)

func init() {
	// Library
	libgdiplus = MustLoadLibrary("gdiplus.dll")

	// Functions
	gdipCreateBitmapFromFile = MustGetProcAddress(libgdiplus, "GdipCreateBitmapFromFile")
	gdipCreateBitmapFromHBITMAP = MustGetProcAddress(libgdiplus, "GdipCreateBitmapFromHBITMAP")
	gdipCreateHBITMAPFromBitmap = MustGetProcAddress(libgdiplus, "GdipCreateHBITMAPFromBitmap")
	gdipDisposeImage = MustGetProcAddress(libgdiplus, "GdipDisposeImage")
	gdiplusShutdown = MustGetProcAddress(libgdiplus, "GdiplusShutdown")
	gdiplusStartup = MustGetProcAddress(libgdiplus, "GdiplusStartup")
}

func GdipCreateBitmapFromFile(filename *uint16, bitmap **GpBitmap) GpStatus {
	ret, _, _ := syscall.Syscall(gdipCreateBitmapFromFile, 2,
		uintptr(unsafe.Pointer(filename)),
		uintptr(unsafe.Pointer(bitmap)),
		0)

	return GpStatus(ret)
}

func GdipCreateBitmapFromHBITMAP(hbm HBITMAP, hpal HPALETTE, bitmap **GpBitmap) GpStatus {
	ret, _, _ := syscall.Syscall(gdipCreateBitmapFromHBITMAP, 3,
		uintptr(hbm),
		uintptr(hpal),
		uintptr(unsafe.Pointer(bitmap)))

	return GpStatus(ret)
}

func GdipCreateHBITMAPFromBitmap(bitmap *GpBitmap, hbmReturn *HBITMAP, background ARGB) GpStatus {
	ret, _, _ := syscall.Syscall(gdipCreateHBITMAPFromBitmap, 3,
		uintptr(unsafe.Pointer(bitmap)),
		uintptr(unsafe.Pointer(hbmReturn)),
		uintptr(background))

	return GpStatus(ret)
}

func GdipDisposeImage(image *GpImage) GpStatus {
	ret, _, _ := syscall.Syscall(gdipDisposeImage, 1,
		uintptr(unsafe.Pointer(image)),
		0,
		0)

	return GpStatus(ret)
}

func GdiplusShutdown() {
	syscall.Syscall(gdiplusShutdown, 1,
		token,
		0,
		0)
}

func GdiplusStartup(input *GdiplusStartupInput, output *GdiplusStartupOutput) GpStatus {
	ret, _, _ := syscall.Syscall(gdiplusStartup, 3,
		uintptr(unsafe.Pointer(&token)),
		uintptr(unsafe.Pointer(input)),
		uintptr(unsafe.Pointer(output)))

	return GpStatus(ret)
}

/*GdipSaveImageToFile(image *GpImage, filename *uint16, clsidEncoder *CLSID, encoderParams *EncoderParameters) GpStatus {
	ret, _, _ := syscall.Syscall6(gdipSaveImageToFile, 4,
		uintptr(unsafe.Pointer(image)),
		uintptr(unsafe.Pointer(filename)),
		uintptr(unsafe.Pointer(clsidEncoder)),
		uintptr(unsafe.Pointer(encoderParams)),
		0,
		0)

	return GpStatus(ret)
}*/
