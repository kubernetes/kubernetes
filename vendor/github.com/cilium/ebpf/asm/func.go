package asm

//go:generate stringer -output func_string.go -type=BuiltinFunc

// BuiltinFunc is a built-in eBPF function.
type BuiltinFunc int32

func (_ BuiltinFunc) Max() BuiltinFunc {
	return maxBuiltinFunc - 1
}

// eBPF built-in functions
//
// You can regenerate this list using the following gawk script:
//
//	/FN\(.+\),/ {
//	  match($1, /\(([a-z_0-9]+),/, r)
//	  split(r[1], p, "_")
//	  printf "Fn"
//	  for (i in p) {
//	    printf "%s%s", toupper(substr(p[i], 1, 1)), substr(p[i], 2)
//	  }
//	  print ""
//	}
//
// The script expects include/uapi/linux/bpf.h as it's input.
const (
	FnUnspec BuiltinFunc = iota
	FnMapLookupElem
	FnMapUpdateElem
	FnMapDeleteElem
	FnProbeRead
	FnKtimeGetNs
	FnTracePrintk
	FnGetPrandomU32
	FnGetSmpProcessorId
	FnSkbStoreBytes
	FnL3CsumReplace
	FnL4CsumReplace
	FnTailCall
	FnCloneRedirect
	FnGetCurrentPidTgid
	FnGetCurrentUidGid
	FnGetCurrentComm
	FnGetCgroupClassid
	FnSkbVlanPush
	FnSkbVlanPop
	FnSkbGetTunnelKey
	FnSkbSetTunnelKey
	FnPerfEventRead
	FnRedirect
	FnGetRouteRealm
	FnPerfEventOutput
	FnSkbLoadBytes
	FnGetStackid
	FnCsumDiff
	FnSkbGetTunnelOpt
	FnSkbSetTunnelOpt
	FnSkbChangeProto
	FnSkbChangeType
	FnSkbUnderCgroup
	FnGetHashRecalc
	FnGetCurrentTask
	FnProbeWriteUser
	FnCurrentTaskUnderCgroup
	FnSkbChangeTail
	FnSkbPullData
	FnCsumUpdate
	FnSetHashInvalid
	FnGetNumaNodeId
	FnSkbChangeHead
	FnXdpAdjustHead
	FnProbeReadStr
	FnGetSocketCookie
	FnGetSocketUid
	FnSetHash
	FnSetsockopt
	FnSkbAdjustRoom
	FnRedirectMap
	FnSkRedirectMap
	FnSockMapUpdate
	FnXdpAdjustMeta
	FnPerfEventReadValue
	FnPerfProgReadValue
	FnGetsockopt
	FnOverrideReturn
	FnSockOpsCbFlagsSet
	FnMsgRedirectMap
	FnMsgApplyBytes
	FnMsgCorkBytes
	FnMsgPullData
	FnBind
	FnXdpAdjustTail
	FnSkbGetXfrmState
	FnGetStack
	FnSkbLoadBytesRelative
	FnFibLookup
	FnSockHashUpdate
	FnMsgRedirectHash
	FnSkRedirectHash
	FnLwtPushEncap
	FnLwtSeg6StoreBytes
	FnLwtSeg6AdjustSrh
	FnLwtSeg6Action
	FnRcRepeat
	FnRcKeydown
	FnSkbCgroupId
	FnGetCurrentCgroupId
	FnGetLocalStorage
	FnSkSelectReuseport
	FnSkbAncestorCgroupId
	FnSkLookupTcp
	FnSkLookupUdp
	FnSkRelease
	FnMapPushElem
	FnMapPopElem
	FnMapPeekElem
	FnMsgPushData
	FnMsgPopData
	FnRcPointerRel
	FnSpinLock
	FnSpinUnlock
	FnSkFullsock
	FnTcpSock
	FnSkbEcnSetCe
	FnGetListenerSock
	FnSkcLookupTcp
	FnTcpCheckSyncookie
	FnSysctlGetName
	FnSysctlGetCurrentValue
	FnSysctlGetNewValue
	FnSysctlSetNewValue
	FnStrtol
	FnStrtoul
	FnSkStorageGet
	FnSkStorageDelete
	FnSendSignal
	FnTcpGenSyncookie
	FnSkbOutput
	FnProbeReadUser
	FnProbeReadKernel
	FnProbeReadUserStr
	FnProbeReadKernelStr
	FnTcpSendAck
	FnSendSignalThread
	FnJiffies64
	FnReadBranchRecords
	FnGetNsCurrentPidTgid
	FnXdpOutput
	FnGetNetnsCookie
	FnGetCurrentAncestorCgroupId
	FnSkAssign
	FnKtimeGetBootNs
	FnSeqPrintf
	FnSeqWrite
	FnSkCgroupId
	FnSkAncestorCgroupId
	FnRingbufOutput
	FnRingbufReserve
	FnRingbufSubmit
	FnRingbufDiscard
	FnRingbufQuery
	FnCsumLevel
	FnSkcToTcp6Sock
	FnSkcToTcpSock
	FnSkcToTcpTimewaitSock
	FnSkcToTcpRequestSock
	FnSkcToUdp6Sock
	FnGetTaskStack
	FnLoadHdrOpt
	FnStoreHdrOpt
	FnReserveHdrOpt
	FnInodeStorageGet
	FnInodeStorageDelete
	FnDPath
	FnCopyFromUser
	FnSnprintfBtf
	FnSeqPrintfBtf
	FnSkbCgroupClassid
	FnRedirectNeigh
	FnPerCpuPtr
	FnThisCpuPtr
	FnRedirectPeer
	FnTaskStorageGet
	FnTaskStorageDelete
	FnGetCurrentTaskBtf
	FnBprmOptsSet
	FnKtimeGetCoarseNs
	FnImaInodeHash
	FnSockFromFile
	FnCheckMtu
	FnForEachMapElem
	FnSnprintf
	FnSysBpf
	FnBtfFindByNameKind
	FnSysClose
	FnTimerInit
	FnTimerSetCallback
	FnTimerStart
	FnTimerCancel
	FnGetFuncIp
	FnGetAttachCookie
	FnTaskPtRegs
	FnGetBranchSnapshot
	FnTraceVprintk
	FnSkcToUnixSock
	FnKallsymsLookupName
	FnFindVma
	FnLoop
	FnStrncmp
	FnGetFuncArg
	FnGetFuncRet
	FnGetFuncArgCnt
	FnGetRetval
	FnSetRetval
	FnXdpGetBuffLen
	FnXdpLoadBytes
	FnXdpStoreBytes
	FnCopyFromUserTask
	FnSkbSetTstamp
	FnImaFileHash
	FnKptrXchg
	FnMapLookupPercpuElem
	FnSkcToMptcpSock
	FnDynptrFromMem
	FnRingbufReserveDynptr
	FnRingbufSubmitDynptr
	FnRingbufDiscardDynptr
	FnDynptrRead
	FnDynptrWrite
	FnDynptrData
	FnTcpRawGenSyncookieIpv4
	FnTcpRawGenSyncookieIpv6
	FnTcpRawCheckSyncookieIpv4
	FnTcpRawCheckSyncookieIpv6
	FnKtimeGetTaiNs
	FnUserRingbufDrain
	FnCgrpStorageGet
	FnCgrpStorageDelete

	maxBuiltinFunc
)

// Call emits a function call.
func (fn BuiltinFunc) Call() Instruction {
	return Instruction{
		OpCode:   OpCode(JumpClass).SetJumpOp(Call),
		Constant: int64(fn),
	}
}
