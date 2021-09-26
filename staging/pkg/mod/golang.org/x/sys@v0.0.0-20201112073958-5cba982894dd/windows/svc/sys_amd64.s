// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// func servicemain(argc uint32, argv **uint16)
TEXT ·servicemain(SB),7,$0
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	SUBQ	$48, SP		// room for SP and 4 args as per Windows requirement
				// plus one extra word to keep stack 16 bytes aligned
	MOVQ	AX, 32(SP)

	MOVL	CX, ·sArgc(SB)
	MOVQ	DX, ·sArgv(SB)

	MOVQ	·sName(SB), CX
	MOVQ	$·servicectlhandler(SB), DX
	// BUG(pastarmovj): Figure out a way to pass in context in R8.
	// Set context to 123456 to test issue #25660.
	MOVQ	$123456, R8
	MOVQ	·cRegisterServiceCtrlHandlerExW(SB), AX
	CALL	AX
	CMPQ	AX, $0
	JE	exit
	MOVQ	AX, ·ssHandle(SB)

	MOVQ	·goWaitsH(SB), CX
	MOVQ	·cSetEvent(SB), AX
	CALL	AX

	MOVQ	·cWaitsH(SB), CX
	MOVQ	$4294967295, DX
	MOVQ	·cWaitForSingleObject(SB), AX
	CALL	AX

exit:
	MOVQ	32(SP), SP
	RET

// I do not know why, but this seems to be the only way to call
// ctlHandlerProc on Windows 7.

// func ·servicectlhandler(ctl uint32, evtype uint32, evdata uintptr, context uintptr) uintptr {
TEXT ·servicectlhandler(SB),7,$0
	MOVQ	·ctlHandlerExProc(SB), AX
	JMP	AX
