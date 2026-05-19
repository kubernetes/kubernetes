// Copyright 2022 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// ProcNetstat models the content of /proc/<pid>/net/netstat.
type ProcNetstat struct {
	// The process ID.
	PID int
	TcpExt
	IpExt
}

type TcpExt struct { // nolint:revive
	SyncookiesSent            *float64
	SyncookiesRecv            *float64
	SyncookiesFailed          *float64
	EmbryonicRsts             *float64
	PruneCalled               *float64
	RcvPruned                 *float64
	OfoPruned                 *float64
	OutOfWindowIcmps          *float64
	LockDroppedIcmps          *float64
	ArpFilter                 *float64
	TW                        *float64
	TWRecycled                *float64
	TWKilled                  *float64
	PAWSActive                *float64
	PAWSEstab                 *float64
	DelayedACKs               *float64
	DelayedACKLocked          *float64
	DelayedACKLost            *float64
	ListenOverflows           *float64
	ListenDrops               *float64
	TCPHPHits                 *float64
	TCPPureAcks               *float64
	TCPHPAcks                 *float64
	TCPRenoRecovery           *float64
	TCPSackRecovery           *float64
	TCPSACKReneging           *float64
	TCPSACKReorder            *float64
	TCPRenoReorder            *float64
	TCPTSReorder              *float64
	TCPFullUndo               *float64
	TCPPartialUndo            *float64
	TCPDSACKUndo              *float64
	TCPLossUndo               *float64
	TCPLostRetransmit         *float64
	TCPRenoFailures           *float64
	TCPSackFailures           *float64
	TCPLossFailures           *float64
	TCPFastRetrans            *float64
	TCPSlowStartRetrans       *float64
	TCPTimeouts               *float64
	TCPLossProbes             *float64
	TCPLossProbeRecovery      *float64
	TCPRenoRecoveryFail       *float64
	TCPSackRecoveryFail       *float64
	TCPRcvCollapsed           *float64
	TCPDSACKOldSent           *float64
	TCPDSACKOfoSent           *float64
	TCPDSACKRecv              *float64
	TCPDSACKOfoRecv           *float64
	TCPAbortOnData            *float64
	TCPAbortOnClose           *float64
	TCPAbortOnMemory          *float64
	TCPAbortOnTimeout         *float64
	TCPAbortOnLinger          *float64
	TCPAbortFailed            *float64
	TCPMemoryPressures        *float64
	TCPMemoryPressuresChrono  *float64
	TCPSACKDiscard            *float64
	TCPDSACKIgnoredOld        *float64
	TCPDSACKIgnoredNoUndo     *float64
	TCPSpuriousRTOs           *float64
	TCPMD5NotFound            *float64
	TCPMD5Unexpected          *float64
	TCPMD5Failure             *float64
	TCPSackShifted            *float64
	TCPSackMerged             *float64
	TCPSackShiftFallback      *float64
	TCPBacklogDrop            *float64
	PFMemallocDrop            *float64
	TCPMinTTLDrop             *float64
	TCPDeferAcceptDrop        *float64
	IPReversePathFilter       *float64
	TCPTimeWaitOverflow       *float64
	TCPReqQFullDoCookies      *float64
	TCPReqQFullDrop           *float64
	TCPRetransFail            *float64
	TCPRcvCoalesce            *float64
	TCPRcvQDrop               *float64
	TCPOFOQueue               *float64
	TCPOFODrop                *float64
	TCPOFOMerge               *float64
	TCPChallengeACK           *float64
	TCPSYNChallenge           *float64
	TCPFastOpenActive         *float64
	TCPFastOpenActiveFail     *float64
	TCPFastOpenPassive        *float64
	TCPFastOpenPassiveFail    *float64
	TCPFastOpenListenOverflow *float64
	TCPFastOpenCookieReqd     *float64
	TCPFastOpenBlackhole      *float64
	TCPSpuriousRtxHostQueues  *float64
	BusyPollRxPackets         *float64
	TCPAutoCorking            *float64
	TCPFromZeroWindowAdv      *float64
	TCPToZeroWindowAdv        *float64
	TCPWantZeroWindowAdv      *float64
	TCPSynRetrans             *float64
	TCPOrigDataSent           *float64
	TCPHystartTrainDetect     *float64
	TCPHystartTrainCwnd       *float64
	TCPHystartDelayDetect     *float64
	TCPHystartDelayCwnd       *float64
	TCPACKSkippedSynRecv      *float64
	TCPACKSkippedPAWS         *float64
	TCPACKSkippedSeq          *float64
	TCPACKSkippedFinWait2     *float64
	TCPACKSkippedTimeWait     *float64
	TCPACKSkippedChallenge    *float64
	TCPWinProbe               *float64
	TCPKeepAlive              *float64
	TCPMTUPFail               *float64
	TCPMTUPSuccess            *float64
	TCPWqueueTooBig           *float64
}

type IpExt struct { // nolint:revive
	InNoRoutes      *float64
	InTruncatedPkts *float64
	InMcastPkts     *float64
	OutMcastPkts    *float64
	InBcastPkts     *float64
	OutBcastPkts    *float64
	InOctets        *float64
	OutOctets       *float64
	InMcastOctets   *float64
	OutMcastOctets  *float64
	InBcastOctets   *float64
	OutBcastOctets  *float64
	InCsumErrors    *float64
	InNoECTPkts     *float64
	InECT1Pkts      *float64
	InECT0Pkts      *float64
	InCEPkts        *float64
	ReasmOverlaps   *float64
}

func (p Proc) Netstat() (ProcNetstat, error) {
	filename := p.path("net/netstat")
	data, err := util.ReadFileNoStat(filename)
	if err != nil {
		return ProcNetstat{PID: p.PID}, err
	}
	procNetstat, err := parseProcNetstat(bytes.NewReader(data), filename)
	procNetstat.PID = p.PID
	return procNetstat, err
}

// parseProcNetstat parses the metrics from proc/<pid>/net/netstat file
// and returns a ProcNetstat structure.
func parseProcNetstat(r io.Reader, fileName string) (ProcNetstat, error) {
	var (
		scanner     = bufio.NewScanner(r)
		procNetstat = ProcNetstat{}
	)

	for scanner.Scan() {
		nameParts := strings.Split(scanner.Text(), " ")
		scanner.Scan()
		valueParts := strings.Split(scanner.Text(), " ")
		// Remove trailing :.
		protocol := strings.TrimSuffix(nameParts[0], ":")
		if len(nameParts) != len(valueParts) {
			return procNetstat, fmt.Errorf("%w: mismatch field count mismatch in %s: %s",
				ErrFileParse, fileName, protocol)
		}
		for i := 1; i < len(nameParts); i++ {
			value, err := strconv.ParseFloat(valueParts[i], 64)
			if err != nil {
				return procNetstat, err
			}
			key := nameParts[i]

			switch protocol {
			case "TcpExt":
				switch key {
				case "SyncookiesSent":
					procNetstat.SyncookiesSent = &value
				case "SyncookiesRecv":
					procNetstat.SyncookiesRecv = &value
				case "SyncookiesFailed":
					procNetstat.SyncookiesFailed = &value
				case "EmbryonicRsts":
					procNetstat.EmbryonicRsts = &value
				case "PruneCalled":
					procNetstat.PruneCalled = &value
				case "RcvPruned":
					procNetstat.RcvPruned = &value
				case "OfoPruned":
					procNetstat.OfoPruned = &value
				case "OutOfWindowIcmps":
					procNetstat.OutOfWindowIcmps = &value
				case "LockDroppedIcmps":
					procNetstat.LockDroppedIcmps = &value
				case "ArpFilter":
					procNetstat.ArpFilter = &value
				case "TW":
					procNetstat.TW = &value
				case "TWRecycled":
					procNetstat.TWRecycled = &value
				case "TWKilled":
					procNetstat.TWKilled = &value
				case "PAWSActive":
					procNetstat.PAWSActive = &value
				case "PAWSEstab":
					procNetstat.PAWSEstab = &value
				case "DelayedACKs":
					procNetstat.DelayedACKs = &value
				case "DelayedACKLocked":
					procNetstat.DelayedACKLocked = &value
				case "DelayedACKLost":
					procNetstat.DelayedACKLost = &value
				case "ListenOverflows":
					procNetstat.ListenOverflows = &value
				case "ListenDrops":
					procNetstat.ListenDrops = &value
				case "TCPHPHits":
					procNetstat.TCPHPHits = &value
				case "TCPPureAcks":
					procNetstat.TCPPureAcks = &value
				case "TCPHPAcks":
					procNetstat.TCPHPAcks = &value
				case "TCPRenoRecovery":
					procNetstat.TCPRenoRecovery = &value
				case "TCPSackRecovery":
					procNetstat.TCPSackRecovery = &value
				case "TCPSACKReneging":
					procNetstat.TCPSACKReneging = &value
				case "TCPSACKReorder":
					procNetstat.TCPSACKReorder = &value
				case "TCPRenoReorder":
					procNetstat.TCPRenoReorder = &value
				case "TCPTSReorder":
					procNetstat.TCPTSReorder = &value
				case "TCPFullUndo":
					procNetstat.TCPFullUndo = &value
				case "TCPPartialUndo":
					procNetstat.TCPPartialUndo = &value
				case "TCPDSACKUndo":
					procNetstat.TCPDSACKUndo = &value
				case "TCPLossUndo":
					procNetstat.TCPLossUndo = &value
				case "TCPLostRetransmit":
					procNetstat.TCPLostRetransmit = &value
				case "TCPRenoFailures":
					procNetstat.TCPRenoFailures = &value
				case "TCPSackFailures":
					procNetstat.TCPSackFailures = &value
				case "TCPLossFailures":
					procNetstat.TCPLossFailures = &value
				case "TCPFastRetrans":
					procNetstat.TCPFastRetrans = &value
				case "TCPSlowStartRetrans":
					procNetstat.TCPSlowStartRetrans = &value
				case "TCPTimeouts":
					procNetstat.TCPTimeouts = &value
				case "TCPLossProbes":
					procNetstat.TCPLossProbes = &value
				case "TCPLossProbeRecovery":
					procNetstat.TCPLossProbeRecovery = &value
				case "TCPRenoRecoveryFail":
					procNetstat.TCPRenoRecoveryFail = &value
				case "TCPSackRecoveryFail":
					procNetstat.TCPSackRecoveryFail = &value
				case "TCPRcvCollapsed":
					procNetstat.TCPRcvCollapsed = &value
				case "TCPDSACKOldSent":
					procNetstat.TCPDSACKOldSent = &value
				case "TCPDSACKOfoSent":
					procNetstat.TCPDSACKOfoSent = &value
				case "TCPDSACKRecv":
					procNetstat.TCPDSACKRecv = &value
				case "TCPDSACKOfoRecv":
					procNetstat.TCPDSACKOfoRecv = &value
				case "TCPAbortOnData":
					procNetstat.TCPAbortOnData = &value
				case "TCPAbortOnClose":
					procNetstat.TCPAbortOnClose = &value
				case "TCPDeferAcceptDrop":
					procNetstat.TCPDeferAcceptDrop = &value
				case "IPReversePathFilter":
					procNetstat.IPReversePathFilter = &value
				case "TCPTimeWaitOverflow":
					procNetstat.TCPTimeWaitOverflow = &value
				case "TCPReqQFullDoCookies":
					procNetstat.TCPReqQFullDoCookies = &value
				case "TCPReqQFullDrop":
					procNetstat.TCPReqQFullDrop = &value
				case "TCPRetransFail":
					procNetstat.TCPRetransFail = &value
				case "TCPRcvCoalesce":
					procNetstat.TCPRcvCoalesce = &value
				case "TCPRcvQDrop":
					procNetstat.TCPRcvQDrop = &value
				case "TCPOFOQueue":
					procNetstat.TCPOFOQueue = &value
				case "TCPOFODrop":
					procNetstat.TCPOFODrop = &value
				case "TCPOFOMerge":
					procNetstat.TCPOFOMerge = &value
				case "TCPChallengeACK":
					procNetstat.TCPChallengeACK = &value
				case "TCPSYNChallenge":
					procNetstat.TCPSYNChallenge = &value
				case "TCPFastOpenActive":
					procNetstat.TCPFastOpenActive = &value
				case "TCPFastOpenActiveFail":
					procNetstat.TCPFastOpenActiveFail = &value
				case "TCPFastOpenPassive":
					procNetstat.TCPFastOpenPassive = &value
				case "TCPFastOpenPassiveFail":
					procNetstat.TCPFastOpenPassiveFail = &value
				case "TCPFastOpenListenOverflow":
					procNetstat.TCPFastOpenListenOverflow = &value
				case "TCPFastOpenCookieReqd":
					procNetstat.TCPFastOpenCookieReqd = &value
				case "TCPFastOpenBlackhole":
					procNetstat.TCPFastOpenBlackhole = &value
				case "TCPSpuriousRtxHostQueues":
					procNetstat.TCPSpuriousRtxHostQueues = &value
				case "BusyPollRxPackets":
					procNetstat.BusyPollRxPackets = &value
				case "TCPAutoCorking":
					procNetstat.TCPAutoCorking = &value
				case "TCPFromZeroWindowAdv":
					procNetstat.TCPFromZeroWindowAdv = &value
				case "TCPToZeroWindowAdv":
					procNetstat.TCPToZeroWindowAdv = &value
				case "TCPWantZeroWindowAdv":
					procNetstat.TCPWantZeroWindowAdv = &value
				case "TCPSynRetrans":
					procNetstat.TCPSynRetrans = &value
				case "TCPOrigDataSent":
					procNetstat.TCPOrigDataSent = &value
				case "TCPHystartTrainDetect":
					procNetstat.TCPHystartTrainDetect = &value
				case "TCPHystartTrainCwnd":
					procNetstat.TCPHystartTrainCwnd = &value
				case "TCPHystartDelayDetect":
					procNetstat.TCPHystartDelayDetect = &value
				case "TCPHystartDelayCwnd":
					procNetstat.TCPHystartDelayCwnd = &value
				case "TCPACKSkippedSynRecv":
					procNetstat.TCPACKSkippedSynRecv = &value
				case "TCPACKSkippedPAWS":
					procNetstat.TCPACKSkippedPAWS = &value
				case "TCPACKSkippedSeq":
					procNetstat.TCPACKSkippedSeq = &value
				case "TCPACKSkippedFinWait2":
					procNetstat.TCPACKSkippedFinWait2 = &value
				case "TCPACKSkippedTimeWait":
					procNetstat.TCPACKSkippedTimeWait = &value
				case "TCPACKSkippedChallenge":
					procNetstat.TCPACKSkippedChallenge = &value
				case "TCPWinProbe":
					procNetstat.TCPWinProbe = &value
				case "TCPKeepAlive":
					procNetstat.TCPKeepAlive = &value
				case "TCPMTUPFail":
					procNetstat.TCPMTUPFail = &value
				case "TCPMTUPSuccess":
					procNetstat.TCPMTUPSuccess = &value
				case "TCPWqueueTooBig":
					procNetstat.TCPWqueueTooBig = &value
				}
			case "IpExt":
				switch key {
				case "InNoRoutes":
					procNetstat.InNoRoutes = &value
				case "InTruncatedPkts":
					procNetstat.InTruncatedPkts = &value
				case "InMcastPkts":
					procNetstat.InMcastPkts = &value
				case "OutMcastPkts":
					procNetstat.OutMcastPkts = &value
				case "InBcastPkts":
					procNetstat.InBcastPkts = &value
				case "OutBcastPkts":
					procNetstat.OutBcastPkts = &value
				case "InOctets":
					procNetstat.InOctets = &value
				case "OutOctets":
					procNetstat.OutOctets = &value
				case "InMcastOctets":
					procNetstat.InMcastOctets = &value
				case "OutMcastOctets":
					procNetstat.OutMcastOctets = &value
				case "InBcastOctets":
					procNetstat.InBcastOctets = &value
				case "OutBcastOctets":
					procNetstat.OutBcastOctets = &value
				case "InCsumErrors":
					procNetstat.InCsumErrors = &value
				case "InNoECTPkts":
					procNetstat.InNoECTPkts = &value
				case "InECT1Pkts":
					procNetstat.InECT1Pkts = &value
				case "InECT0Pkts":
					procNetstat.InECT0Pkts = &value
				case "InCEPkts":
					procNetstat.InCEPkts = &value
				case "ReasmOverlaps":
					procNetstat.ReasmOverlaps = &value
				}
			}
		}
	}
	return procNetstat, scanner.Err()
}
