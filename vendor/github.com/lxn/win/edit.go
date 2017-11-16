// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

// Edit styles
const (
	ES_LEFT        = 0x0000
	ES_CENTER      = 0x0001
	ES_RIGHT       = 0x0002
	ES_MULTILINE   = 0x0004
	ES_UPPERCASE   = 0x0008
	ES_LOWERCASE   = 0x0010
	ES_PASSWORD    = 0x0020
	ES_AUTOVSCROLL = 0x0040
	ES_AUTOHSCROLL = 0x0080
	ES_NOHIDESEL   = 0x0100
	ES_OEMCONVERT  = 0x0400
	ES_READONLY    = 0x0800
	ES_WANTRETURN  = 0x1000
	ES_NUMBER      = 0x2000
)

// Edit notifications
const (
	EN_SETFOCUS     = 0x0100
	EN_KILLFOCUS    = 0x0200
	EN_CHANGE       = 0x0300
	EN_UPDATE       = 0x0400
	EN_ERRSPACE     = 0x0500
	EN_MAXTEXT      = 0x0501
	EN_HSCROLL      = 0x0601
	EN_VSCROLL      = 0x0602
	EN_ALIGN_LTR_EC = 0x0700
	EN_ALIGN_RTL_EC = 0x0701
)

// Edit messages
const (
	EM_GETSEL              = 0x00B0
	EM_SETSEL              = 0x00B1
	EM_GETRECT             = 0x00B2
	EM_SETRECT             = 0x00B3
	EM_SETRECTNP           = 0x00B4
	EM_SCROLL              = 0x00B5
	EM_LINESCROLL          = 0x00B6
	EM_SCROLLCARET         = 0x00B7
	EM_GETMODIFY           = 0x00B8
	EM_SETMODIFY           = 0x00B9
	EM_GETLINECOUNT        = 0x00BA
	EM_LINEINDEX           = 0x00BB
	EM_SETHANDLE           = 0x00BC
	EM_GETHANDLE           = 0x00BD
	EM_GETTHUMB            = 0x00BE
	EM_LINELENGTH          = 0x00C1
	EM_REPLACESEL          = 0x00C2
	EM_GETLINE             = 0x00C4
	EM_LIMITTEXT           = 0x00C5
	EM_CANUNDO             = 0x00C6
	EM_UNDO                = 0x00C7
	EM_FMTLINES            = 0x00C8
	EM_LINEFROMCHAR        = 0x00C9
	EM_SETTABSTOPS         = 0x00CB
	EM_SETPASSWORDCHAR     = 0x00CC
	EM_EMPTYUNDOBUFFER     = 0x00CD
	EM_GETFIRSTVISIBLELINE = 0x00CE
	EM_SETREADONLY         = 0x00CF
	EM_SETWORDBREAKPROC    = 0x00D0
	EM_GETWORDBREAKPROC    = 0x00D1
	EM_GETPASSWORDCHAR     = 0x00D2
	EM_SETMARGINS          = 0x00D3
	EM_GETMARGINS          = 0x00D4
	EM_SETLIMITTEXT        = EM_LIMITTEXT
	EM_GETLIMITTEXT        = 0x00D5
	EM_POSFROMCHAR         = 0x00D6
	EM_CHARFROMPOS         = 0x00D7
	EM_SETIMESTATUS        = 0x00D8
	EM_GETIMESTATUS        = 0x00D9
	EM_SETCUEBANNER        = 0x1501
	EM_GETCUEBANNER        = 0x1502
)
