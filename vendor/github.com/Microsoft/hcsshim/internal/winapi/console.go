//go:build windows

package winapi

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

const PSEUDOCONSOLE_INHERIT_CURSOR = 0x1

// CreatePseudoConsole creates a windows pseudo console.
func CreatePseudoConsole(size windows.Coord, hInput windows.Handle, hOutput windows.Handle, dwFlags uint32, hpcon *windows.Handle) error {
	// We need this wrapper as the function takes a COORD struct and not a pointer to one, so we need to cast to something beforehand.
	return createPseudoConsole(*((*uint32)(unsafe.Pointer(&size))), hInput, hOutput, 0, hpcon)
}

// ResizePseudoConsole resizes the internal buffers of the pseudo console to the width and height specified in `size`.
func ResizePseudoConsole(hpcon windows.Handle, size windows.Coord) error {
	// We need this wrapper as the function takes a COORD struct and not a pointer to one, so we need to cast to something beforehand.
	return resizePseudoConsole(hpcon, *((*uint32)(unsafe.Pointer(&size))))
}

// HRESULT WINAPI CreatePseudoConsole(
//     _In_ COORD size,
//     _In_ HANDLE hInput,
//     _In_ HANDLE hOutput,
//     _In_ DWORD dwFlags,
//     _Out_ HPCON* phPC
// );
//
//sys createPseudoConsole(size uint32, hInput windows.Handle, hOutput windows.Handle, dwFlags uint32, hpcon *windows.Handle) (hr error) = kernel32.CreatePseudoConsole

// void WINAPI ClosePseudoConsole(
//     _In_ HPCON hPC
// );
//
//sys ClosePseudoConsole(hpc windows.Handle) = kernel32.ClosePseudoConsole

// HRESULT WINAPI ResizePseudoConsole(
//     _In_ HPCON hPC ,
//     _In_ COORD size
// );
//
//sys resizePseudoConsole(hPc windows.Handle, size uint32) (hr error) = kernel32.ResizePseudoConsole
