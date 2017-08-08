// Copyright 2012 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

const (
	HDF_SORTDOWN = 0x200
	HDF_SORTUP   = 0x400
)

const (
	HDI_FORMAT = 4
)

const (
	HDM_FIRST   = 0x1200
	HDM_GETITEM = HDM_FIRST + 11
	HDM_SETITEM = HDM_FIRST + 12
)

const (
	HDS_NOSIZING = 0x0800
)

type HDITEM struct {
	Mask       uint32
	Cxy        int32
	PszText    *uint16
	Hbm        HBITMAP
	CchTextMax int32
	Fmt        int32
	LParam     uintptr
	IImage     int32
	IOrder     int32
	Type       uint32
	PvFilter   uintptr
}
