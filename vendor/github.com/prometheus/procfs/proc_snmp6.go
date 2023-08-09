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
	"errors"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// ProcSnmp6 models the content of /proc/<pid>/net/snmp6.
type ProcSnmp6 struct {
	// The process ID.
	PID int
	Ip6
	Icmp6
	Udp6
	UdpLite6
}

type Ip6 struct { // nolint:revive
	InReceives       float64
	InHdrErrors      float64
	InTooBigErrors   float64
	InNoRoutes       float64
	InAddrErrors     float64
	InUnknownProtos  float64
	InTruncatedPkts  float64
	InDiscards       float64
	InDelivers       float64
	OutForwDatagrams float64
	OutRequests      float64
	OutDiscards      float64
	OutNoRoutes      float64
	ReasmTimeout     float64
	ReasmReqds       float64
	ReasmOKs         float64
	ReasmFails       float64
	FragOKs          float64
	FragFails        float64
	FragCreates      float64
	InMcastPkts      float64
	OutMcastPkts     float64
	InOctets         float64
	OutOctets        float64
	InMcastOctets    float64
	OutMcastOctets   float64
	InBcastOctets    float64
	OutBcastOctets   float64
	InNoECTPkts      float64
	InECT1Pkts       float64
	InECT0Pkts       float64
	InCEPkts         float64
}

type Icmp6 struct {
	InMsgs                    float64
	InErrors                  float64
	OutMsgs                   float64
	OutErrors                 float64
	InCsumErrors              float64
	InDestUnreachs            float64
	InPktTooBigs              float64
	InTimeExcds               float64
	InParmProblems            float64
	InEchos                   float64
	InEchoReplies             float64
	InGroupMembQueries        float64
	InGroupMembResponses      float64
	InGroupMembReductions     float64
	InRouterSolicits          float64
	InRouterAdvertisements    float64
	InNeighborSolicits        float64
	InNeighborAdvertisements  float64
	InRedirects               float64
	InMLDv2Reports            float64
	OutDestUnreachs           float64
	OutPktTooBigs             float64
	OutTimeExcds              float64
	OutParmProblems           float64
	OutEchos                  float64
	OutEchoReplies            float64
	OutGroupMembQueries       float64
	OutGroupMembResponses     float64
	OutGroupMembReductions    float64
	OutRouterSolicits         float64
	OutRouterAdvertisements   float64
	OutNeighborSolicits       float64
	OutNeighborAdvertisements float64
	OutRedirects              float64
	OutMLDv2Reports           float64
	InType1                   float64
	InType134                 float64
	InType135                 float64
	InType136                 float64
	InType143                 float64
	OutType133                float64
	OutType135                float64
	OutType136                float64
	OutType143                float64
}

type Udp6 struct { // nolint:revive
	InDatagrams  float64
	NoPorts      float64
	InErrors     float64
	OutDatagrams float64
	RcvbufErrors float64
	SndbufErrors float64
	InCsumErrors float64
	IgnoredMulti float64
}

type UdpLite6 struct { // nolint:revive
	InDatagrams  float64
	NoPorts      float64
	InErrors     float64
	OutDatagrams float64
	RcvbufErrors float64
	SndbufErrors float64
	InCsumErrors float64
}

func (p Proc) Snmp6() (ProcSnmp6, error) {
	filename := p.path("net/snmp6")
	data, err := util.ReadFileNoStat(filename)
	if err != nil {
		// On systems with IPv6 disabled, this file won't exist.
		// Do nothing.
		if errors.Is(err, os.ErrNotExist) {
			return ProcSnmp6{PID: p.PID}, nil
		}

		return ProcSnmp6{PID: p.PID}, err
	}

	procSnmp6, err := parseSNMP6Stats(bytes.NewReader(data))
	procSnmp6.PID = p.PID
	return procSnmp6, err
}

// parseSnmp6 parses the metrics from proc/<pid>/net/snmp6 file
// and returns a map contains those metrics.
func parseSNMP6Stats(r io.Reader) (ProcSnmp6, error) {
	var (
		scanner   = bufio.NewScanner(r)
		procSnmp6 = ProcSnmp6{}
	)

	for scanner.Scan() {
		stat := strings.Fields(scanner.Text())
		if len(stat) < 2 {
			continue
		}
		// Expect to have "6" in metric name, skip line otherwise
		if sixIndex := strings.Index(stat[0], "6"); sixIndex != -1 {
			protocol := stat[0][:sixIndex+1]
			key := stat[0][sixIndex+1:]
			value, err := strconv.ParseFloat(stat[1], 64)
			if err != nil {
				return procSnmp6, err
			}

			switch protocol {
			case "Ip6":
				switch key {
				case "InReceives":
					procSnmp6.Ip6.InReceives = value
				case "InHdrErrors":
					procSnmp6.Ip6.InHdrErrors = value
				case "InTooBigErrors":
					procSnmp6.Ip6.InTooBigErrors = value
				case "InNoRoutes":
					procSnmp6.Ip6.InNoRoutes = value
				case "InAddrErrors":
					procSnmp6.Ip6.InAddrErrors = value
				case "InUnknownProtos":
					procSnmp6.Ip6.InUnknownProtos = value
				case "InTruncatedPkts":
					procSnmp6.Ip6.InTruncatedPkts = value
				case "InDiscards":
					procSnmp6.Ip6.InDiscards = value
				case "InDelivers":
					procSnmp6.Ip6.InDelivers = value
				case "OutForwDatagrams":
					procSnmp6.Ip6.OutForwDatagrams = value
				case "OutRequests":
					procSnmp6.Ip6.OutRequests = value
				case "OutDiscards":
					procSnmp6.Ip6.OutDiscards = value
				case "OutNoRoutes":
					procSnmp6.Ip6.OutNoRoutes = value
				case "ReasmTimeout":
					procSnmp6.Ip6.ReasmTimeout = value
				case "ReasmReqds":
					procSnmp6.Ip6.ReasmReqds = value
				case "ReasmOKs":
					procSnmp6.Ip6.ReasmOKs = value
				case "ReasmFails":
					procSnmp6.Ip6.ReasmFails = value
				case "FragOKs":
					procSnmp6.Ip6.FragOKs = value
				case "FragFails":
					procSnmp6.Ip6.FragFails = value
				case "FragCreates":
					procSnmp6.Ip6.FragCreates = value
				case "InMcastPkts":
					procSnmp6.Ip6.InMcastPkts = value
				case "OutMcastPkts":
					procSnmp6.Ip6.OutMcastPkts = value
				case "InOctets":
					procSnmp6.Ip6.InOctets = value
				case "OutOctets":
					procSnmp6.Ip6.OutOctets = value
				case "InMcastOctets":
					procSnmp6.Ip6.InMcastOctets = value
				case "OutMcastOctets":
					procSnmp6.Ip6.OutMcastOctets = value
				case "InBcastOctets":
					procSnmp6.Ip6.InBcastOctets = value
				case "OutBcastOctets":
					procSnmp6.Ip6.OutBcastOctets = value
				case "InNoECTPkts":
					procSnmp6.Ip6.InNoECTPkts = value
				case "InECT1Pkts":
					procSnmp6.Ip6.InECT1Pkts = value
				case "InECT0Pkts":
					procSnmp6.Ip6.InECT0Pkts = value
				case "InCEPkts":
					procSnmp6.Ip6.InCEPkts = value

				}
			case "Icmp6":
				switch key {
				case "InMsgs":
					procSnmp6.Icmp6.InMsgs = value
				case "InErrors":
					procSnmp6.Icmp6.InErrors = value
				case "OutMsgs":
					procSnmp6.Icmp6.OutMsgs = value
				case "OutErrors":
					procSnmp6.Icmp6.OutErrors = value
				case "InCsumErrors":
					procSnmp6.Icmp6.InCsumErrors = value
				case "InDestUnreachs":
					procSnmp6.Icmp6.InDestUnreachs = value
				case "InPktTooBigs":
					procSnmp6.Icmp6.InPktTooBigs = value
				case "InTimeExcds":
					procSnmp6.Icmp6.InTimeExcds = value
				case "InParmProblems":
					procSnmp6.Icmp6.InParmProblems = value
				case "InEchos":
					procSnmp6.Icmp6.InEchos = value
				case "InEchoReplies":
					procSnmp6.Icmp6.InEchoReplies = value
				case "InGroupMembQueries":
					procSnmp6.Icmp6.InGroupMembQueries = value
				case "InGroupMembResponses":
					procSnmp6.Icmp6.InGroupMembResponses = value
				case "InGroupMembReductions":
					procSnmp6.Icmp6.InGroupMembReductions = value
				case "InRouterSolicits":
					procSnmp6.Icmp6.InRouterSolicits = value
				case "InRouterAdvertisements":
					procSnmp6.Icmp6.InRouterAdvertisements = value
				case "InNeighborSolicits":
					procSnmp6.Icmp6.InNeighborSolicits = value
				case "InNeighborAdvertisements":
					procSnmp6.Icmp6.InNeighborAdvertisements = value
				case "InRedirects":
					procSnmp6.Icmp6.InRedirects = value
				case "InMLDv2Reports":
					procSnmp6.Icmp6.InMLDv2Reports = value
				case "OutDestUnreachs":
					procSnmp6.Icmp6.OutDestUnreachs = value
				case "OutPktTooBigs":
					procSnmp6.Icmp6.OutPktTooBigs = value
				case "OutTimeExcds":
					procSnmp6.Icmp6.OutTimeExcds = value
				case "OutParmProblems":
					procSnmp6.Icmp6.OutParmProblems = value
				case "OutEchos":
					procSnmp6.Icmp6.OutEchos = value
				case "OutEchoReplies":
					procSnmp6.Icmp6.OutEchoReplies = value
				case "OutGroupMembQueries":
					procSnmp6.Icmp6.OutGroupMembQueries = value
				case "OutGroupMembResponses":
					procSnmp6.Icmp6.OutGroupMembResponses = value
				case "OutGroupMembReductions":
					procSnmp6.Icmp6.OutGroupMembReductions = value
				case "OutRouterSolicits":
					procSnmp6.Icmp6.OutRouterSolicits = value
				case "OutRouterAdvertisements":
					procSnmp6.Icmp6.OutRouterAdvertisements = value
				case "OutNeighborSolicits":
					procSnmp6.Icmp6.OutNeighborSolicits = value
				case "OutNeighborAdvertisements":
					procSnmp6.Icmp6.OutNeighborAdvertisements = value
				case "OutRedirects":
					procSnmp6.Icmp6.OutRedirects = value
				case "OutMLDv2Reports":
					procSnmp6.Icmp6.OutMLDv2Reports = value
				case "InType1":
					procSnmp6.Icmp6.InType1 = value
				case "InType134":
					procSnmp6.Icmp6.InType134 = value
				case "InType135":
					procSnmp6.Icmp6.InType135 = value
				case "InType136":
					procSnmp6.Icmp6.InType136 = value
				case "InType143":
					procSnmp6.Icmp6.InType143 = value
				case "OutType133":
					procSnmp6.Icmp6.OutType133 = value
				case "OutType135":
					procSnmp6.Icmp6.OutType135 = value
				case "OutType136":
					procSnmp6.Icmp6.OutType136 = value
				case "OutType143":
					procSnmp6.Icmp6.OutType143 = value
				}
			case "Udp6":
				switch key {
				case "InDatagrams":
					procSnmp6.Udp6.InDatagrams = value
				case "NoPorts":
					procSnmp6.Udp6.NoPorts = value
				case "InErrors":
					procSnmp6.Udp6.InErrors = value
				case "OutDatagrams":
					procSnmp6.Udp6.OutDatagrams = value
				case "RcvbufErrors":
					procSnmp6.Udp6.RcvbufErrors = value
				case "SndbufErrors":
					procSnmp6.Udp6.SndbufErrors = value
				case "InCsumErrors":
					procSnmp6.Udp6.InCsumErrors = value
				case "IgnoredMulti":
					procSnmp6.Udp6.IgnoredMulti = value
				}
			case "UdpLite6":
				switch key {
				case "InDatagrams":
					procSnmp6.UdpLite6.InDatagrams = value
				case "NoPorts":
					procSnmp6.UdpLite6.NoPorts = value
				case "InErrors":
					procSnmp6.UdpLite6.InErrors = value
				case "OutDatagrams":
					procSnmp6.UdpLite6.OutDatagrams = value
				case "RcvbufErrors":
					procSnmp6.UdpLite6.RcvbufErrors = value
				case "SndbufErrors":
					procSnmp6.UdpLite6.SndbufErrors = value
				case "InCsumErrors":
					procSnmp6.UdpLite6.InCsumErrors = value
				}
			}
		}
	}
	return procSnmp6, scanner.Err()
}
