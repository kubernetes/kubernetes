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

// ProcSnmp models the content of /proc/<pid>/net/snmp.
type ProcSnmp struct {
	// The process ID.
	PID int
	Ip
	Icmp
	IcmpMsg
	Tcp
	Udp
	UdpLite
}

type Ip struct { // nolint:revive
	Forwarding      *float64
	DefaultTTL      *float64
	InReceives      *float64
	InHdrErrors     *float64
	InAddrErrors    *float64
	ForwDatagrams   *float64
	InUnknownProtos *float64
	InDiscards      *float64
	InDelivers      *float64
	OutRequests     *float64
	OutDiscards     *float64
	OutNoRoutes     *float64
	ReasmTimeout    *float64
	ReasmReqds      *float64
	ReasmOKs        *float64
	ReasmFails      *float64
	FragOKs         *float64
	FragFails       *float64
	FragCreates     *float64
}

type Icmp struct { // nolint:revive
	InMsgs           *float64
	InErrors         *float64
	InCsumErrors     *float64
	InDestUnreachs   *float64
	InTimeExcds      *float64
	InParmProbs      *float64
	InSrcQuenchs     *float64
	InRedirects      *float64
	InEchos          *float64
	InEchoReps       *float64
	InTimestamps     *float64
	InTimestampReps  *float64
	InAddrMasks      *float64
	InAddrMaskReps   *float64
	OutMsgs          *float64
	OutErrors        *float64
	OutDestUnreachs  *float64
	OutTimeExcds     *float64
	OutParmProbs     *float64
	OutSrcQuenchs    *float64
	OutRedirects     *float64
	OutEchos         *float64
	OutEchoReps      *float64
	OutTimestamps    *float64
	OutTimestampReps *float64
	OutAddrMasks     *float64
	OutAddrMaskReps  *float64
}

type IcmpMsg struct {
	InType3  *float64
	OutType3 *float64
}

type Tcp struct { // nolint:revive
	RtoAlgorithm *float64
	RtoMin       *float64
	RtoMax       *float64
	MaxConn      *float64
	ActiveOpens  *float64
	PassiveOpens *float64
	AttemptFails *float64
	EstabResets  *float64
	CurrEstab    *float64
	InSegs       *float64
	OutSegs      *float64
	RetransSegs  *float64
	InErrs       *float64
	OutRsts      *float64
	InCsumErrors *float64
}

type Udp struct { // nolint:revive
	InDatagrams  *float64
	NoPorts      *float64
	InErrors     *float64
	OutDatagrams *float64
	RcvbufErrors *float64
	SndbufErrors *float64
	InCsumErrors *float64
	IgnoredMulti *float64
}

type UdpLite struct { // nolint:revive
	InDatagrams  *float64
	NoPorts      *float64
	InErrors     *float64
	OutDatagrams *float64
	RcvbufErrors *float64
	SndbufErrors *float64
	InCsumErrors *float64
	IgnoredMulti *float64
}

func (p Proc) Snmp() (ProcSnmp, error) {
	filename := p.path("net/snmp")
	data, err := util.ReadFileNoStat(filename)
	if err != nil {
		return ProcSnmp{PID: p.PID}, err
	}
	procSnmp, err := parseSnmp(bytes.NewReader(data), filename)
	procSnmp.PID = p.PID
	return procSnmp, err
}

// parseSnmp parses the metrics from proc/<pid>/net/snmp file
// and returns a map contains those metrics (e.g. {"Ip": {"Forwarding": 2}}).
func parseSnmp(r io.Reader, fileName string) (ProcSnmp, error) {
	var (
		scanner  = bufio.NewScanner(r)
		procSnmp = ProcSnmp{}
	)

	for scanner.Scan() {
		nameParts := strings.Split(scanner.Text(), " ")
		scanner.Scan()
		valueParts := strings.Split(scanner.Text(), " ")
		// Remove trailing :.
		protocol := strings.TrimSuffix(nameParts[0], ":")
		if len(nameParts) != len(valueParts) {
			return procSnmp, fmt.Errorf("%w: mismatch field count mismatch in %s: %s",
				ErrFileParse, fileName, protocol)
		}
		for i := 1; i < len(nameParts); i++ {
			value, err := strconv.ParseFloat(valueParts[i], 64)
			if err != nil {
				return procSnmp, err
			}
			key := nameParts[i]

			switch protocol {
			case "Ip":
				switch key {
				case "Forwarding":
					procSnmp.Ip.Forwarding = &value
				case "DefaultTTL":
					procSnmp.Ip.DefaultTTL = &value
				case "InReceives":
					procSnmp.Ip.InReceives = &value
				case "InHdrErrors":
					procSnmp.Ip.InHdrErrors = &value
				case "InAddrErrors":
					procSnmp.Ip.InAddrErrors = &value
				case "ForwDatagrams":
					procSnmp.Ip.ForwDatagrams = &value
				case "InUnknownProtos":
					procSnmp.Ip.InUnknownProtos = &value
				case "InDiscards":
					procSnmp.Ip.InDiscards = &value
				case "InDelivers":
					procSnmp.Ip.InDelivers = &value
				case "OutRequests":
					procSnmp.Ip.OutRequests = &value
				case "OutDiscards":
					procSnmp.Ip.OutDiscards = &value
				case "OutNoRoutes":
					procSnmp.Ip.OutNoRoutes = &value
				case "ReasmTimeout":
					procSnmp.Ip.ReasmTimeout = &value
				case "ReasmReqds":
					procSnmp.Ip.ReasmReqds = &value
				case "ReasmOKs":
					procSnmp.Ip.ReasmOKs = &value
				case "ReasmFails":
					procSnmp.Ip.ReasmFails = &value
				case "FragOKs":
					procSnmp.Ip.FragOKs = &value
				case "FragFails":
					procSnmp.Ip.FragFails = &value
				case "FragCreates":
					procSnmp.Ip.FragCreates = &value
				}
			case "Icmp":
				switch key {
				case "InMsgs":
					procSnmp.Icmp.InMsgs = &value
				case "InErrors":
					procSnmp.Icmp.InErrors = &value
				case "InCsumErrors":
					procSnmp.Icmp.InCsumErrors = &value
				case "InDestUnreachs":
					procSnmp.Icmp.InDestUnreachs = &value
				case "InTimeExcds":
					procSnmp.Icmp.InTimeExcds = &value
				case "InParmProbs":
					procSnmp.Icmp.InParmProbs = &value
				case "InSrcQuenchs":
					procSnmp.Icmp.InSrcQuenchs = &value
				case "InRedirects":
					procSnmp.Icmp.InRedirects = &value
				case "InEchos":
					procSnmp.Icmp.InEchos = &value
				case "InEchoReps":
					procSnmp.Icmp.InEchoReps = &value
				case "InTimestamps":
					procSnmp.Icmp.InTimestamps = &value
				case "InTimestampReps":
					procSnmp.Icmp.InTimestampReps = &value
				case "InAddrMasks":
					procSnmp.Icmp.InAddrMasks = &value
				case "InAddrMaskReps":
					procSnmp.Icmp.InAddrMaskReps = &value
				case "OutMsgs":
					procSnmp.Icmp.OutMsgs = &value
				case "OutErrors":
					procSnmp.Icmp.OutErrors = &value
				case "OutDestUnreachs":
					procSnmp.Icmp.OutDestUnreachs = &value
				case "OutTimeExcds":
					procSnmp.Icmp.OutTimeExcds = &value
				case "OutParmProbs":
					procSnmp.Icmp.OutParmProbs = &value
				case "OutSrcQuenchs":
					procSnmp.Icmp.OutSrcQuenchs = &value
				case "OutRedirects":
					procSnmp.Icmp.OutRedirects = &value
				case "OutEchos":
					procSnmp.Icmp.OutEchos = &value
				case "OutEchoReps":
					procSnmp.Icmp.OutEchoReps = &value
				case "OutTimestamps":
					procSnmp.Icmp.OutTimestamps = &value
				case "OutTimestampReps":
					procSnmp.Icmp.OutTimestampReps = &value
				case "OutAddrMasks":
					procSnmp.Icmp.OutAddrMasks = &value
				case "OutAddrMaskReps":
					procSnmp.Icmp.OutAddrMaskReps = &value
				}
			case "IcmpMsg":
				switch key {
				case "InType3":
					procSnmp.IcmpMsg.InType3 = &value
				case "OutType3":
					procSnmp.IcmpMsg.OutType3 = &value
				}
			case "Tcp":
				switch key {
				case "RtoAlgorithm":
					procSnmp.Tcp.RtoAlgorithm = &value
				case "RtoMin":
					procSnmp.Tcp.RtoMin = &value
				case "RtoMax":
					procSnmp.Tcp.RtoMax = &value
				case "MaxConn":
					procSnmp.Tcp.MaxConn = &value
				case "ActiveOpens":
					procSnmp.Tcp.ActiveOpens = &value
				case "PassiveOpens":
					procSnmp.Tcp.PassiveOpens = &value
				case "AttemptFails":
					procSnmp.Tcp.AttemptFails = &value
				case "EstabResets":
					procSnmp.Tcp.EstabResets = &value
				case "CurrEstab":
					procSnmp.Tcp.CurrEstab = &value
				case "InSegs":
					procSnmp.Tcp.InSegs = &value
				case "OutSegs":
					procSnmp.Tcp.OutSegs = &value
				case "RetransSegs":
					procSnmp.Tcp.RetransSegs = &value
				case "InErrs":
					procSnmp.Tcp.InErrs = &value
				case "OutRsts":
					procSnmp.Tcp.OutRsts = &value
				case "InCsumErrors":
					procSnmp.Tcp.InCsumErrors = &value
				}
			case "Udp":
				switch key {
				case "InDatagrams":
					procSnmp.Udp.InDatagrams = &value
				case "NoPorts":
					procSnmp.Udp.NoPorts = &value
				case "InErrors":
					procSnmp.Udp.InErrors = &value
				case "OutDatagrams":
					procSnmp.Udp.OutDatagrams = &value
				case "RcvbufErrors":
					procSnmp.Udp.RcvbufErrors = &value
				case "SndbufErrors":
					procSnmp.Udp.SndbufErrors = &value
				case "InCsumErrors":
					procSnmp.Udp.InCsumErrors = &value
				case "IgnoredMulti":
					procSnmp.Udp.IgnoredMulti = &value
				}
			case "UdpLite":
				switch key {
				case "InDatagrams":
					procSnmp.UdpLite.InDatagrams = &value
				case "NoPorts":
					procSnmp.UdpLite.NoPorts = &value
				case "InErrors":
					procSnmp.UdpLite.InErrors = &value
				case "OutDatagrams":
					procSnmp.UdpLite.OutDatagrams = &value
				case "RcvbufErrors":
					procSnmp.UdpLite.RcvbufErrors = &value
				case "SndbufErrors":
					procSnmp.UdpLite.SndbufErrors = &value
				case "InCsumErrors":
					procSnmp.UdpLite.InCsumErrors = &value
				case "IgnoredMulti":
					procSnmp.UdpLite.IgnoredMulti = &value
				}
			}
		}
	}
	return procSnmp, scanner.Err()
}
