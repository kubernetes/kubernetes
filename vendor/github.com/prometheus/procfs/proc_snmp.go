// Copyright The Prometheus Authors
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
					procSnmp.Forwarding = &value
				case "DefaultTTL":
					procSnmp.DefaultTTL = &value
				case "InReceives":
					procSnmp.InReceives = &value
				case "InHdrErrors":
					procSnmp.InHdrErrors = &value
				case "InAddrErrors":
					procSnmp.InAddrErrors = &value
				case "ForwDatagrams":
					procSnmp.ForwDatagrams = &value
				case "InUnknownProtos":
					procSnmp.InUnknownProtos = &value
				case "InDiscards":
					procSnmp.InDiscards = &value
				case "InDelivers":
					procSnmp.InDelivers = &value
				case "OutRequests":
					procSnmp.OutRequests = &value
				case "OutDiscards":
					procSnmp.OutDiscards = &value
				case "OutNoRoutes":
					procSnmp.OutNoRoutes = &value
				case "ReasmTimeout":
					procSnmp.ReasmTimeout = &value
				case "ReasmReqds":
					procSnmp.ReasmReqds = &value
				case "ReasmOKs":
					procSnmp.ReasmOKs = &value
				case "ReasmFails":
					procSnmp.ReasmFails = &value
				case "FragOKs":
					procSnmp.FragOKs = &value
				case "FragFails":
					procSnmp.FragFails = &value
				case "FragCreates":
					procSnmp.FragCreates = &value
				}
			case "Icmp":
				switch key {
				case "InMsgs":
					procSnmp.InMsgs = &value
				case "InErrors":
					procSnmp.Icmp.InErrors = &value
				case "InCsumErrors":
					procSnmp.Icmp.InCsumErrors = &value
				case "InDestUnreachs":
					procSnmp.InDestUnreachs = &value
				case "InTimeExcds":
					procSnmp.InTimeExcds = &value
				case "InParmProbs":
					procSnmp.InParmProbs = &value
				case "InSrcQuenchs":
					procSnmp.InSrcQuenchs = &value
				case "InRedirects":
					procSnmp.InRedirects = &value
				case "InEchos":
					procSnmp.InEchos = &value
				case "InEchoReps":
					procSnmp.InEchoReps = &value
				case "InTimestamps":
					procSnmp.InTimestamps = &value
				case "InTimestampReps":
					procSnmp.InTimestampReps = &value
				case "InAddrMasks":
					procSnmp.InAddrMasks = &value
				case "InAddrMaskReps":
					procSnmp.InAddrMaskReps = &value
				case "OutMsgs":
					procSnmp.OutMsgs = &value
				case "OutErrors":
					procSnmp.OutErrors = &value
				case "OutDestUnreachs":
					procSnmp.OutDestUnreachs = &value
				case "OutTimeExcds":
					procSnmp.OutTimeExcds = &value
				case "OutParmProbs":
					procSnmp.OutParmProbs = &value
				case "OutSrcQuenchs":
					procSnmp.OutSrcQuenchs = &value
				case "OutRedirects":
					procSnmp.OutRedirects = &value
				case "OutEchos":
					procSnmp.OutEchos = &value
				case "OutEchoReps":
					procSnmp.OutEchoReps = &value
				case "OutTimestamps":
					procSnmp.OutTimestamps = &value
				case "OutTimestampReps":
					procSnmp.OutTimestampReps = &value
				case "OutAddrMasks":
					procSnmp.OutAddrMasks = &value
				case "OutAddrMaskReps":
					procSnmp.OutAddrMaskReps = &value
				}
			case "IcmpMsg":
				switch key {
				case "InType3":
					procSnmp.InType3 = &value
				case "OutType3":
					procSnmp.OutType3 = &value
				}
			case "Tcp":
				switch key {
				case "RtoAlgorithm":
					procSnmp.RtoAlgorithm = &value
				case "RtoMin":
					procSnmp.RtoMin = &value
				case "RtoMax":
					procSnmp.RtoMax = &value
				case "MaxConn":
					procSnmp.MaxConn = &value
				case "ActiveOpens":
					procSnmp.ActiveOpens = &value
				case "PassiveOpens":
					procSnmp.PassiveOpens = &value
				case "AttemptFails":
					procSnmp.AttemptFails = &value
				case "EstabResets":
					procSnmp.EstabResets = &value
				case "CurrEstab":
					procSnmp.CurrEstab = &value
				case "InSegs":
					procSnmp.InSegs = &value
				case "OutSegs":
					procSnmp.OutSegs = &value
				case "RetransSegs":
					procSnmp.RetransSegs = &value
				case "InErrs":
					procSnmp.InErrs = &value
				case "OutRsts":
					procSnmp.OutRsts = &value
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
