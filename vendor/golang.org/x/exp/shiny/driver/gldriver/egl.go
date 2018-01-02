// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gldriver

// These constants match the values found in the EGL 1.4 headers,
// egl.h, eglext.h, and eglplatform.h.
const (
	_EGL_DONT_CARE = -1

	_EGL_NO_SURFACE = 0
	_EGL_NO_CONTEXT = 0
	_EGL_NO_DISPLAY = 0

	_EGL_OPENGL_ES2_BIT = 0x04 // EGL_RENDERABLE_TYPE mask
	_EGL_WINDOW_BIT     = 0x04 // EGL_SURFACE_TYPE mask

	_EGL_OPENGL_ES_API   = 0x30A0
	_EGL_RENDERABLE_TYPE = 0x3040
	_EGL_SURFACE_TYPE    = 0x3033
	_EGL_BUFFER_SIZE     = 0x3020
	_EGL_ALPHA_SIZE      = 0x3021
	_EGL_BLUE_SIZE       = 0x3022
	_EGL_GREEN_SIZE      = 0x3023
	_EGL_RED_SIZE        = 0x3024
	_EGL_DEPTH_SIZE      = 0x3025
	_EGL_STENCIL_SIZE    = 0x3026
	_EGL_SAMPLE_BUFFERS  = 0x3032
	_EGL_CONFIG_CAVEAT   = 0x3027
	_EGL_NONE            = 0x3038

	_EGL_PLATFORM_ANGLE_ANGLE                   = 0x3202
	_EGL_PLATFORM_ANGLE_TYPE_ANGLE              = 0x3203
	_EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE = 0x3204
	_EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE = 0x3205
	_EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE      = 0x3206

	_EGL_CONTEXT_CLIENT_VERSION = 0x3098
)

const (
	_EGL_SUCCESS             = 0x3000
	_EGL_NOT_INITIALIZED     = 0x3001
	_EGL_BAD_ACCESS          = 0x3002
	_EGL_BAD_ALLOC           = 0x3003
	_EGL_BAD_ATTRIBUTE       = 0x3004
	_EGL_BAD_CONFIG          = 0x3005
	_EGL_BAD_CONTEXT         = 0x3006
	_EGL_BAD_CURRENT_SURFACE = 0x3007
	_EGL_BAD_DISPLAY         = 0x3008
	_EGL_BAD_MATCH           = 0x3009
	_EGL_BAD_NATIVE_PIXMAP   = 0x300A
	_EGL_BAD_NATIVE_WINDOW   = 0x300B
	_EGL_BAD_PARAMETER       = 0x300C
	_EGL_BAD_SURFACE         = 0x300D
	_EGL_CONTEXT_LOST        = 0x300E
)

func eglErrString(errno uintptr) string {
	switch errno {
	case _EGL_SUCCESS:
		return "EGL_SUCCESS"
	case _EGL_NOT_INITIALIZED:
		return "EGL_NOT_INITIALIZED"
	case _EGL_BAD_ACCESS:
		return "EGL_BAD_ACCESS"
	case _EGL_BAD_ALLOC:
		return "EGL_BAD_ALLOC"
	case _EGL_BAD_ATTRIBUTE:
		return "EGL_BAD_ATTRIBUTE"
	case _EGL_BAD_CONFIG:
		return "EGL_BAD_CONFIG"
	case _EGL_BAD_CONTEXT:
		return "EGL_BAD_CONTEXT"
	case _EGL_BAD_CURRENT_SURFACE:
		return "EGL_BAD_CURRENT_SURFACE"
	case _EGL_BAD_DISPLAY:
		return "EGL_BAD_DISPLAY"
	case _EGL_BAD_MATCH:
		return "EGL_BAD_MATCH"
	case _EGL_BAD_NATIVE_PIXMAP:
		return "EGL_BAD_NATIVE_PIXMAP"
	case _EGL_BAD_NATIVE_WINDOW:
		return "EGL_BAD_NATIVE_WINDOW"
	case _EGL_BAD_PARAMETER:
		return "EGL_BAD_PARAMETER"
	case _EGL_BAD_SURFACE:
		return "EGL_BAD_SURFACE"
	case _EGL_CONTEXT_LOST:
		return "EGL_CONTEXT_LOST"
	}
	return "EGL: unknown error"
}
