// Copyright 2011 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

const DTM_FIRST = 0x1000
const DTN_FIRST = ^uint32(739)  // -740
const DTN_FIRST2 = ^uint32(752) // -753

const (
	GDTR_MIN = 0x0001
	GDTR_MAX = 0x0002
)

const (
	GDT_ERROR = -1
	GDT_VALID = 0
	GDT_NONE  = 1
)

// Messages
const (
	DTM_GETSYSTEMTIME = DTM_FIRST + 1
	DTM_SETSYSTEMTIME = DTM_FIRST + 2
	DTM_GETRANGE      = DTM_FIRST + 3
	DTM_SETRANGE      = DTM_FIRST + 4
	DTM_SETFORMAT     = DTM_FIRST + 50
	DTM_SETMCCOLOR    = DTM_FIRST + 6
	DTM_GETMCCOLOR    = DTM_FIRST + 7
	DTM_GETMONTHCAL   = DTM_FIRST + 8
	DTM_SETMCFONT     = DTM_FIRST + 9
	DTM_GETMCFONT     = DTM_FIRST + 10
)

// Styles
const (
	DTS_UPDOWN                 = 0x0001
	DTS_SHOWNONE               = 0x0002
	DTS_SHORTDATEFORMAT        = 0x0000
	DTS_LONGDATEFORMAT         = 0x0004
	DTS_SHORTDATECENTURYFORMAT = 0x000C
	DTS_TIMEFORMAT             = 0x0009
	DTS_APPCANPARSE            = 0x0010
	DTS_RIGHTALIGN             = 0x0020
)

// Notifications
const (
	DTN_DATETIMECHANGE = DTN_FIRST2 - 6
	DTN_USERSTRING     = DTN_FIRST - 5
	DTN_WMKEYDOWN      = DTN_FIRST - 4
	DTN_FORMAT         = DTN_FIRST - 3
	DTN_FORMATQUERY    = DTN_FIRST - 2
	DTN_DROPDOWN       = DTN_FIRST2 - 1
	DTN_CLOSEUP        = DTN_FIRST2
)

// Structs
type (
	NMDATETIMECHANGE struct {
		Nmhdr   NMHDR
		DwFlags uint32
		St      SYSTEMTIME
	}

	NMDATETIMESTRING struct {
		Nmhdr         NMHDR
		PszUserString *uint16
		St            SYSTEMTIME
		DwFlags       uint32
	}

	NMDATETIMEWMKEYDOWN struct {
		Nmhdr     NMHDR
		NVirtKey  int
		PszFormat *uint16
		St        SYSTEMTIME
	}

	NMDATETIMEFORMAT struct {
		Nmhdr      NMHDR
		PszFormat  *uint16
		St         SYSTEMTIME
		PszDisplay *uint16
		SzDisplay  [64]uint16
	}

	NMDATETIMEFORMATQUERY struct {
		Nmhdr     NMHDR
		PszFormat *uint16
		SzMax     SIZE
	}
)
