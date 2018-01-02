// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win32

import (
	"fmt"
	"syscall"
	"unicode/utf16"

	"golang.org/x/mobile/event/key"
)

// convVirtualKeyCode converts a Win32 virtual key code number
// into the standard keycodes used by the key package.
func convVirtualKeyCode(vKey uint32) key.Code {
	switch vKey {
	case 0x01: // VK_LBUTTON left mouse button
	case 0x02: // VK_RBUTTON right mouse button
	case 0x03: // VK_CANCEL control-break processing
	case 0x04: // VK_MBUTTON middle mouse button
	case 0x05: // VK_XBUTTON1 X1 mouse button
	case 0x06: // VK_XBUTTON2 X2 mouse button
	case 0x08: // VK_BACK
		return key.CodeDeleteBackspace
	case 0x09: // VK_TAB
		return key.CodeTab
	case 0x0C: // VK_CLEAR
	case 0x0D: // VK_RETURN
		return key.CodeReturnEnter
	case 0x10: // VK_SHIFT
		return key.CodeLeftShift
	case 0x11: // VK_CONTROL
		return key.CodeLeftControl
	case 0x12: // VK_MENU
		return key.CodeLeftAlt
	case 0x13: // VK_PAUSE
	case 0x14: // VK_CAPITAL
		return key.CodeCapsLock
	case 0x15: // VK_KANA, VK_HANGUEL, VK_HANGUL
	case 0x17: // VK_JUNJA
	case 0x18: // VK_FINA, L
	case 0x19: // VK_HANJA, VK_KANJI
	case 0x1B: // VK_ESCAPE
		return key.CodeEscape
	case 0x1C: // VK_CONVERT
	case 0x1D: // VK_NONCONVERT
	case 0x1E: // VK_ACCEPT
	case 0x1F: // VK_MODECHANGE
	case 0x20: // VK_SPACE
		return key.CodeSpacebar
	case 0x21: // VK_PRIOR
		return key.CodePageUp
	case 0x22: // VK_NEXT
		return key.CodePageDown
	case 0x23: // VK_END
		return key.CodeEnd
	case 0x24: // VK_HOME
		return key.CodeHome
	case 0x25: // VK_LEFT
		return key.CodeLeftArrow
	case 0x26: // VK_UP
		return key.CodeUpArrow
	case 0x27: // VK_RIGHT
		return key.CodeRightArrow
	case 0x28: // VK_DOWN
		return key.CodeDownArrow
	case 0x29: // VK_SELECT
	case 0x2A: // VK_PRINT
	case 0x2B: // VK_EXECUTE
	case 0x2C: // VK_SNAPSHOT
	case 0x2D: // VK_INSERT
	case 0x2E: // VK_DELETE
		return key.CodeDeleteForward
	case 0x2F: // VK_HELP
		return key.CodeHelp
	case 0x30:
		return key.Code0
	case 0x31:
		return key.Code1
	case 0x32:
		return key.Code2
	case 0x33:
		return key.Code3
	case 0x34:
		return key.Code4
	case 0x35:
		return key.Code5
	case 0x36:
		return key.Code6
	case 0x37:
		return key.Code7
	case 0x38:
		return key.Code8
	case 0x39:
		return key.Code9
	case 0x41:
		return key.CodeA
	case 0x42:
		return key.CodeB
	case 0x43:
		return key.CodeC
	case 0x44:
		return key.CodeD
	case 0x45:
		return key.CodeE
	case 0x46:
		return key.CodeF
	case 0x47:
		return key.CodeG
	case 0x48:
		return key.CodeH
	case 0x49:
		return key.CodeI
	case 0x4A:
		return key.CodeJ
	case 0x4B:
		return key.CodeK
	case 0x4C:
		return key.CodeL
	case 0x4D:
		return key.CodeM
	case 0x4E:
		return key.CodeN
	case 0x4F:
		return key.CodeO
	case 0x50:
		return key.CodeP
	case 0x51:
		return key.CodeQ
	case 0x52:
		return key.CodeR
	case 0x53:
		return key.CodeS
	case 0x54:
		return key.CodeT
	case 0x55:
		return key.CodeU
	case 0x56:
		return key.CodeV
	case 0x57:
		return key.CodeW
	case 0x58:
		return key.CodeX
	case 0x59:
		return key.CodeY
	case 0x5A:
		return key.CodeZ
	case 0x5B: // VK_LWIN
		return key.CodeLeftGUI
	case 0x5C: // VK_RWIN
		return key.CodeRightGUI
	case 0x5D: // VK_APPS
	case 0x5F: // VK_SLEEP
	case 0x60: // VK_NUMPAD0
		return key.CodeKeypad0
	case 0x61: // VK_NUMPAD1
		return key.CodeKeypad1
	case 0x62: // VK_NUMPAD2
		return key.CodeKeypad2
	case 0x63: // VK_NUMPAD3
		return key.CodeKeypad3
	case 0x64: // VK_NUMPAD4
		return key.CodeKeypad4
	case 0x65: // VK_NUMPAD5
		return key.CodeKeypad5
	case 0x66: // VK_NUMPAD6
		return key.CodeKeypad6
	case 0x67: // VK_NUMPAD7
		return key.CodeKeypad7
	case 0x68: // VK_NUMPAD8
		return key.CodeKeypad8
	case 0x69: // VK_NUMPAD9
		return key.CodeKeypad9
	case 0x6A: // VK_MULTIPLY
		return key.CodeKeypadAsterisk
	case 0x6B: // VK_ADD
		return key.CodeKeypadPlusSign
	case 0x6C: // VK_SEPARATOR
	case 0x6D: // VK_SUBTRACT
		return key.CodeKeypadHyphenMinus
	case 0x6E: // VK_DECIMAL
		return key.CodeFullStop
	case 0x6F: // VK_DIVIDE
		return key.CodeKeypadSlash
	case 0x70: // VK_F1
		return key.CodeF1
	case 0x71: // VK_F2
		return key.CodeF2
	case 0x72: // VK_F3
		return key.CodeF3
	case 0x73: // VK_F4
		return key.CodeF4
	case 0x74: // VK_F5
		return key.CodeF5
	case 0x75: // VK_F6
		return key.CodeF6
	case 0x76: // VK_F7
		return key.CodeF7
	case 0x77: // VK_F8
		return key.CodeF8
	case 0x78: // VK_F9
		return key.CodeF9
	case 0x79: // VK_F10
		return key.CodeF10
	case 0x7A: // VK_F11
		return key.CodeF11
	case 0x7B: // VK_F12
		return key.CodeF12
	case 0x7C: // VK_F13
		return key.CodeF13
	case 0x7D: // VK_F14
		return key.CodeF14
	case 0x7E: // VK_F15
		return key.CodeF15
	case 0x7F: // VK_F16
		return key.CodeF16
	case 0x80: // VK_F17
		return key.CodeF17
	case 0x81: // VK_F18
		return key.CodeF18
	case 0x82: // VK_F19
		return key.CodeF19
	case 0x83: // VK_F20
		return key.CodeF20
	case 0x84: // VK_F21
		return key.CodeF21
	case 0x85: // VK_F22
		return key.CodeF22
	case 0x86: // VK_F23
		return key.CodeF23
	case 0x87: // VK_F24
		return key.CodeF24
	case 0x90: // VK_NUMLOCK
		return key.CodeKeypadNumLock
	case 0x91: // VK_SCROLL
	case 0xA0: // VK_LSHIFT
		return key.CodeLeftShift
	case 0xA1: // VK_RSHIFT
		return key.CodeRightShift
	case 0xA2: // VK_LCONTROL
		return key.CodeLeftControl
	case 0xA3: // VK_RCONTROL
		return key.CodeRightControl
	case 0xA4: // VK_LMENU
	case 0xA5: // VK_RMENU
	case 0xA6: // VK_BROWSER_BACK
	case 0xA7: // VK_BROWSER_FORWARD
	case 0xA8: // VK_BROWSER_REFRESH
	case 0xA9: // VK_BROWSER_STOP
	case 0xAA: // VK_BROWSER_SEARCH
	case 0xAB: // VK_BROWSER_FAVORITES
	case 0xAC: // VK_BROWSER_HOME
	case 0xAD: // VK_VOLUME_MUTE
		return key.CodeMute
	case 0xAE: // VK_VOLUME_DOWN
		return key.CodeVolumeDown
	case 0xAF: // VK_VOLUME_UP
		return key.CodeVolumeUp
	case 0xB0: // VK_MEDIA_NEXT_TRACK
	case 0xB1: // VK_MEDIA_PREV_TRACK
	case 0xB2: // VK_MEDIA_STOP
	case 0xB3: // VK_MEDIA_PLAY_PAUSE
	case 0xB4: // VK_LAUNCH_MAIL
	case 0xB5: // VK_LAUNCH_MEDIA_SELECT
	case 0xB6: // VK_LAUNCH_APP1
	case 0xB7: // VK_LAUNCH_APP2
	case 0xBA: // VK_OEM_1 ';:'
		return key.CodeSemicolon
	case 0xBB: // VK_OEM_PLUS '+'
		return key.CodeEqualSign
	case 0xBC: // VK_OEM_COMMA ','
		return key.CodeComma
	case 0xBD: // VK_OEM_MINUS '-'
		return key.CodeHyphenMinus
	case 0xBE: // VK_OEM_PERIOD '.'
		return key.CodeFullStop
	case 0xBF: // VK_OEM_2 '/?'
		return key.CodeSlash
	case 0xC0: // VK_OEM_3 '`~'
		return key.CodeGraveAccent
	case 0xDB: // VK_OEM_4 '[{'
		return key.CodeLeftSquareBracket
	case 0xDC: // VK_OEM_5 '\|'
		return key.CodeBackslash
	case 0xDD: // VK_OEM_6 ']}'
		return key.CodeRightSquareBracket
	case 0xDE: // VK_OEM_7 'single-quote/double-quote'
		return key.CodeApostrophe
	case 0xDF: // VK_OEM_8
		return key.CodeUnknown
	case 0xE2: // VK_OEM_102
	case 0xE5: // VK_PROCESSKEY
	case 0xE7: // VK_PACKET
	case 0xF6: // VK_ATTN
	case 0xF7: // VK_CRSEL
	case 0xF8: // VK_EXSEL
	case 0xF9: // VK_EREOF
	case 0xFA: // VK_PLAY
	case 0xFB: // VK_ZOOM
	case 0xFC: // VK_NONAME
	case 0xFD: // VK_PA1
	case 0xFE: // VK_OEM_CLEAR
	}
	return key.CodeUnknown
}

func readRune(vKey uint32, scanCode uint8) rune {
	var (
		keystate [256]byte
		buf      [4]uint16
	)
	if err := _GetKeyboardState(&keystate[0]); err != nil {
		panic(fmt.Sprintf("win32: %v", err))
	}
	// TODO: cache GetKeyboardLayout result, update on WM_INPUTLANGCHANGE
	layout := _GetKeyboardLayout(0)
	ret := _ToUnicodeEx(vKey, uint32(scanCode), &keystate[0], &buf[0], int32(len(buf)), 0, layout)
	if ret < 1 {
		return -1
	}
	return utf16.Decode(buf[:ret])[0]
}

func sendKeyEvent(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	e := key.Event{
		Rune:      readRune(uint32(wParam), uint8(lParam>>16)),
		Code:      convVirtualKeyCode(uint32(wParam)),
		Modifiers: keyModifiers(),
	}
	switch uMsg {
	case _WM_KEYDOWN:
		const prevMask = 1 << 30
		if repeat := lParam&prevMask == prevMask; repeat {
			e.Direction = key.DirNone
		} else {
			e.Direction = key.DirPress
		}
	case _WM_KEYUP:
		e.Direction = key.DirRelease
	default:
		panic(fmt.Sprintf("win32: unexpected key message: %d", uMsg))
	}

	KeyEvent(hwnd, e)
	return 0
}
