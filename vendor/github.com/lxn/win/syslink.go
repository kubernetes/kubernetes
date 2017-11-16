// Copyright 2017 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

const (
	INVALID_LINK_INDEX = -1
	MAX_LINKID_TEXT    = 48
	L_MAX_URL_LENGTH   = 2048 + 32 + len("://")
	WC_LINK = "SysLink"
)

const (
	LWS_TRANSPARENT    = 0x0001
	LWS_IGNORERETURN   = 0x0002
	LWS_NOPREFIX       = 0x0004
	LWS_USEVISUALSTYLE = 0x0008
	LWS_USECUSTOMTEXT  = 0x0010
	LWS_RIGHT          = 0x0020
)

const (
	LIF_ITEMINDEX = 0x00000001
	LIF_STATE     = 0x00000002
	LIF_ITEMID    = 0x00000004
	LIF_URL       = 0x00000008
)

const (
	LIS_FOCUSED       = 0x00000001
	LIS_ENABLED       = 0x00000002
	LIS_VISITED       = 0x00000004
	LIS_HOTTRACK      = 0x00000008
	LIS_DEFAULTCOLORS = 0x00000010
)

const (
	LM_HITTEST        = WM_USER + 0x300
	LM_GETIDEALHEIGHT = WM_USER + 0x301
	LM_SETITEM        = WM_USER + 0x302
	LM_GETITEM        = WM_USER + 0x303
	LM_GETIDEALSIZE   = LM_GETIDEALHEIGHT
)

type LITEM struct {
	Mask      uint32
	ILink     int32
	State     uint32
	StateMask uint32
	SzID      [MAX_LINKID_TEXT]uint16
	SzUrl     [L_MAX_URL_LENGTH]uint16
}

type LHITTESTINFO struct {
	Pt   POINT
	Item LITEM
}

type NMLINK struct {
	Hdr  NMHDR
	Item LITEM
}
