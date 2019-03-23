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
	"strconv"
	"strings"

	"k8s.io/kubernetes/test/e2e/framework"
)

type IPerfResults struct {
	BandwidthMap map[string]int64
}

// IPerfResult struct modelling an iperf record....
// 20160314154239,172.17.0.3,34152,172.17.0.2,5001,3,0.0-10.0,33843707904,27074774092
type IPerfResult struct {
	date          string // field 1 in the csv
	cli           string // field 2 in the csv
	cliPort       int64  // ...
	server        string
	servPort      int64
	id            string
	interval      string
	transferBits  int64
	bandwidthBits int64
}

// Add adds a new result to the Results struct.
func (i *IPerfResults) Add(ipr *IPerfResult) {
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
		asJson, _ := json.Marshal(node)
		buffer.WriteString("\t " + string(asJson) + "\t " + fmt.Sprintf("%E", float64(bandwidth)))
	}
	return buffer.String()
}

// NewIPerf parses an IPerf CSV output line into an IPerfResult.
func NewIPerf(csvLine string) *IPerfResult {
	csvLine = strings.Trim(csvLine, "\n")
	slice := StrSlice(strings.Split(csvLine, ","))
	if len(slice) != 9 {
		framework.Failf("Incorrect fields in the output: %v (%v out of 9)", slice, len(slice))
	}
	i := IPerfResult{}
	i.date = slice.get(0)
	i.cli = slice.get(1)
	i.cliPort = intOrFail("client port", slice.get(2))
	i.server = slice.get(3)
	i.servPort = intOrFail("server port", slice.get(4))
	i.id = slice.get(5)
	i.interval = slice.get(6)
	i.transferBits = intOrFail("transfer port", slice.get(7))
	i.bandwidthBits = intOrFail("bandwidth port", slice.get(8))
	return &i
}

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
