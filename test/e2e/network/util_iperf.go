/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package network

// Tests network performance using iperf or other containers.
import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	megabyte = 1024 * 1024
)

// IPerfResults is a struct that stores some IPerfCSVResult
type IPerfResults struct {
	BandwidthMap map[string]int64
}

// IPerfCSVResult struct modelling an iperf record....
// 20160314154239,172.17.0.3,34152,172.17.0.2,5001,3,0.0-10.0,33843707904,27074774092
type IPerfCSVResult struct {
	date          string // field 1 in the csv
	cli           string // field 2 in the csv
	cliPort       int64  // ...
	server        string
	servPort      int64
	id            string
	intervalStart string
	intervalEnd   string
	transferBits  int64
	bandwidthBits int64
}

func (i *IPerfCSVResult) bandwidthMB() int64 {
	return int64(math.Round(float64(i.bandwidthBits) / float64(megabyte) / 8))
}

// Add adds a new result to the Results struct.
func (i *IPerfResults) Add(ipr *IPerfCSVResult) {
	if i.BandwidthMap == nil {
		i.BandwidthMap = map[string]int64{}
	}
	i.BandwidthMap[ipr.cli] = ipr.bandwidthBits
}

// ToTSV exports an easily readable tab delimited format of all IPerfResults.
func (i *IPerfResults) ToTSV() string {
	if len(i.BandwidthMap) < 1 {
		framework.Logf("Warning: no data in bandwidth map")
	}

	var buffer bytes.Buffer
	for node, bandwidth := range i.BandwidthMap {
		asJSON, _ := json.Marshal(node)
		buffer.WriteString("\t " + string(asJSON) + "\t " + fmt.Sprintf("%E", float64(bandwidth)))
	}
	return buffer.String()
}

// NewIPerf parses an IPerf CSV output line into an IPerfCSVResult.
func NewIPerf(csvLine string) (*IPerfCSVResult, error) {
	if len(csvLine) == 0 {
		return nil, fmt.Errorf("no iperf output received in csv line")
	}
	csvLine = strings.Trim(csvLine, "\n")
	slice := StrSlice(strings.Split(csvLine, ","))
	// iperf 2.2.0+ reports 17 fields, before it was just 15
	if len(slice) != 17 {
		return nil, fmt.Errorf("incorrect fields in the output: %v (%v out of 17)", slice, len(slice))
	}
	i := IPerfCSVResult{}
	i.date = slice.get(0)
	i.cli = slice.get(1)
	i.cliPort = intOrFail("client port", slice.get(2))
	i.server = slice.get(3)
	i.servPort = intOrFail("server port", slice.get(4))
	i.id = slice.get(5)
	i.intervalStart = slice.get(6)
	i.intervalEnd = slice.get(7)
	i.transferBits = intOrFail("transfer port", slice.get(8))
	i.bandwidthBits = intOrFail("bandwidth port", slice.get(9))
	return &i, nil
}

// StrSlice represents a string slice
type StrSlice []string

func (s StrSlice) get(i int) string {
	if i >= 0 && i < len(s) {
		return s[i]
	}
	return ""
}

// intOrFail is a convenience function for parsing integers.
func intOrFail(debugName string, rawValue string) int64 {
	value, err := strconv.ParseInt(rawValue, 10, 64)
	if err != nil {
		framework.Failf("Failed parsing value %v from the string '%v' as an integer", debugName, rawValue)
	}
	return value
}

// IPerf2EnhancedCSVResults models the results produced by iperf2 when run with the -e (--enhancedreports) flag.
type IPerf2EnhancedCSVResults struct {
	Intervals []*IPerfCSVResult
	Total     *IPerfCSVResult
}

// ParseIPerf2EnhancedResultsFromCSV parses results from iperf2 when given the -e (--enhancedreports)
// and `--reportstyle C` options.
// Example output for version < 2.1.9 (agnhost < 2.53):
// 20201210141800.884,10.244.2.24,47880,10.96.114.79,6789,3,0.0-1.0,1677852672,13422821376
// 20201210141801.881,10.244.2.24,47880,10.96.114.79,6789,3,1.0-2.0,1980760064,15846080512
// 20201210141802.883,10.244.2.24,47880,10.96.114.79,6789,3,2.0-3.0,1886650368,15093202944
// Example output with version >= 2.1.9 (agnhost >= 2.53)
// +0000:20240908113035.128,192.168.9.3,58256,192.168.9.4,5001,1,0.0-1.0,5220466748,41763733984,-1,-1,-1,-1,0,0
// +0000:20240908113036.128,192.168.9.3,58256,192.168.9.4,5001,1,1.0-2.0,5127667712,41021341696,-1,-1,-1,-1,0,0
// Example output with version >= 2.2.0 (agnhost >= 2.55)
// time,srcaddress,srcport,dstaddr,dstport,transferid,istart,iend,bytes,speed,writecnt,writeerr,tcpretry,tcpcwnd,tcppcwnd,tcprtt,tcprttvar
// +0000:20250605191028.955,10.244.2.64,43380,10.96.223.154,6789,1,0.0,1.0,7817396288,62539170304,-1,-1,4294967295,-1,4294967295,0,0
// +0000:20250605191029.955,10.244.2.64,43380,10.96.223.154,6789,1,1.0,2.0,7550795776,60406366208,-1,-1,4294967295,-1,4294967295,0,0
// +0000:20250605191030.955,10.244.2.64,43380,10.96.223.154,6789,1,2.0,3.0,8703574016,69628592128,-1,-1,4294967295,-1,4294967295,0,0
func ParseIPerf2EnhancedResultsFromCSV(output string) (*IPerf2EnhancedCSVResults, error) {
	var parsedResults []*IPerfCSVResult
	for i, line := range strings.Split(output, "\n") {
		if i == 0 {
			// we skip the first line, iperf 2.2.0+ returns headers as first line in output
			continue
		}
		parsed, err := NewIPerf(line)
		if err != nil {
			return nil, err
		}
		parsedResults = append(parsedResults, parsed)
	}
	if len(parsedResults) == 0 {
		return nil, fmt.Errorf("no results parsed from iperf2 output")
	}
	// format:
	// all but last lines are intervals
	intervals := parsedResults[:len(parsedResults)-1]
	// last line is an aggregation
	total := parsedResults[len(parsedResults)-1]
	return &IPerf2EnhancedCSVResults{
		Intervals: intervals,
		Total:     total,
	}, nil
}

// IPerf2NodeToNodeCSVResults models the results of running iperf2 between a daemonset of clients and
// a single server.  The node name of the server is captured, along with a map of client node name
// to iperf2 results.
type IPerf2NodeToNodeCSVResults struct {
	ServerNode string
	Results    map[string]*IPerf2EnhancedCSVResults
}
