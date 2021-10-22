// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func servicemain(argc uint32, argv **uint16)
TEXT ·servicemain(SB),NOSPLIT|NOFRAME,$0
	MOVD	R0, ·sArgc(SB)
	MOVD	R1, ·sArgv(SB)

	MOVD	·sName(SB), R0
	MOVD	·ctlHandlerExProc(SB), R1
	MOVD	$0, R2
	MOVD	·cRegisterServiceCtrlHandlerExW(SB), R3
	BL	(R3)
	CMP	$0, R0
	BEQ	exit
	MOVD	R0, ·ssHandle(SB)

	MOVD	·goWaitsH(SB), R0
	MOVD	·cSetEvent(SB), R1
	BL	(R1)

	MOVD	·cWaitsH(SB), R0
	MOVD	$-1, R1
	MOVD	·cWaitForSingleObject(SB), R2
	BL	(R2)

exit:
	RET
