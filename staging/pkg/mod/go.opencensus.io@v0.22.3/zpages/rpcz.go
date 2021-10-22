// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package zpages

import (
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sort"
	"sync"
	"text/tabwriter"
	"time"

	"go.opencensus.io/plugin/ocgrpc"
	"go.opencensus.io/stats/view"
)

const bytesPerKb = 1024

var (
	programStartTime = time.Now()
	mu               sync.Mutex // protects snaps
	snaps            = make(map[methodKey]*statSnapshot)

	// viewType lists the views we are interested in for RPC stats.
	// A view's map value indicates whether that view contains data for received
	// RPCs.
	viewType = map[*view.View]bool{
		ocgrpc.ClientCompletedRPCsView:          false,
		ocgrpc.ClientSentBytesPerRPCView:        false,
		ocgrpc.ClientSentMessagesPerRPCView:     false,
		ocgrpc.ClientReceivedBytesPerRPCView:    false,
		ocgrpc.ClientReceivedMessagesPerRPCView: false,
		ocgrpc.ClientRoundtripLatencyView:       false,
		ocgrpc.ServerCompletedRPCsView:          true,
		ocgrpc.ServerReceivedBytesPerRPCView:    true,
		ocgrpc.ServerReceivedMessagesPerRPCView: true,
		ocgrpc.ServerSentBytesPerRPCView:        true,
		ocgrpc.ServerSentMessagesPerRPCView:     true,
		ocgrpc.ServerLatencyView:                true,
	}
)

func registerRPCViews() {
	views := make([]*view.View, 0, len(viewType))
	for v := range viewType {
		views = append(views, v)
	}
	if err := view.Register(views...); err != nil {
		log.Printf("error subscribing to views: %v", err)
	}
	view.RegisterExporter(snapExporter{})
}

func rpczHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	WriteHTMLRpczPage(w)
}

// WriteHTMLRpczPage writes an HTML document to w containing per-method RPC stats.
func WriteHTMLRpczPage(w io.Writer) {
	if err := headerTemplate.Execute(w, headerData{Title: "RPC Stats"}); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
	WriteHTMLRpczSummary(w)
	if err := footerTemplate.Execute(w, nil); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
}

// WriteHTMLRpczSummary writes HTML to w containing per-method RPC stats.
//
// It includes neither a header nor footer, so you can embed this data in other pages.
func WriteHTMLRpczSummary(w io.Writer) {
	mu.Lock()
	if err := statsTemplate.Execute(w, getStatsPage()); err != nil {
		log.Printf("zpages: executing template: %v", err)
	}
	mu.Unlock()
}

// WriteTextRpczPage writes formatted text to w containing per-method RPC stats.
func WriteTextRpczPage(w io.Writer) {
	mu.Lock()
	defer mu.Unlock()
	page := getStatsPage()

	for i, sg := range page.StatGroups {
		switch i {
		case 0:
			fmt.Fprint(w, "Sent:\n")
		case 1:
			fmt.Fprint(w, "\nReceived:\n")
		}
		tw := tabwriter.NewWriter(w, 6, 8, 1, ' ', 0)
		fmt.Fprint(tw, "Method\tCount\t\t\tAvgLat\t\t\tMaxLat\t\t\tRate\t\t\tIn (MiB/s)\t\t\tOut (MiB/s)\t\t\tErrors\t\t\n")
		fmt.Fprint(tw, "\tMin\tHr\tTot\tMin\tHr\tTot\tMin\tHr\tTot\tMin\tHr\tTot\tMin\tHr\tTot\tMin\tHr\tTot\tMin\tHr\tTot\n")
		for _, s := range sg.Snapshots {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%v\t%v\t%v\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\n",
				s.Method,
				s.CountMinute,
				s.CountHour,
				s.CountTotal,
				s.AvgLatencyMinute,
				s.AvgLatencyHour,
				s.AvgLatencyTotal,
				s.RPCRateMinute,
				s.RPCRateHour,
				s.RPCRateTotal,
				s.InputRateMinute/bytesPerKb,
				s.InputRateHour/bytesPerKb,
				s.InputRateTotal/bytesPerKb,
				s.OutputRateMinute/bytesPerKb,
				s.OutputRateHour/bytesPerKb,
				s.OutputRateTotal/bytesPerKb,
				s.ErrorsMinute,
				s.ErrorsHour,
				s.ErrorsTotal)
		}
		tw.Flush()
	}
}

// headerData contains data for the header template.
type headerData struct {
	Title string
}

// statsPage aggregates stats on the page for 'sent' and 'received' categories
type statsPage struct {
	StatGroups []*statGroup
}

// statGroup aggregates snapshots for a directional category
type statGroup struct {
	Direction string
	Snapshots []*statSnapshot
}

func (s *statGroup) Len() int {
	return len(s.Snapshots)
}

func (s *statGroup) Swap(i, j int) {
	s.Snapshots[i], s.Snapshots[j] = s.Snapshots[j], s.Snapshots[i]
}

func (s *statGroup) Less(i, j int) bool {
	return s.Snapshots[i].Method < s.Snapshots[j].Method
}

// statSnapshot holds the data items that are presented in a single row of RPC
// stat information.
type statSnapshot struct {
	// TODO: compute hour/minute values from cumulative
	Method           string
	Received         bool
	CountMinute      uint64
	CountHour        uint64
	CountTotal       uint64
	AvgLatencyMinute time.Duration
	AvgLatencyHour   time.Duration
	AvgLatencyTotal  time.Duration
	RPCRateMinute    float64
	RPCRateHour      float64
	RPCRateTotal     float64
	InputRateMinute  float64
	InputRateHour    float64
	InputRateTotal   float64
	OutputRateMinute float64
	OutputRateHour   float64
	OutputRateTotal  float64
	ErrorsMinute     uint64
	ErrorsHour       uint64
	ErrorsTotal      uint64
}

type methodKey struct {
	method   string
	received bool
}

type snapExporter struct{}

func (s snapExporter) ExportView(vd *view.Data) {
	received, ok := viewType[vd.View]
	if !ok {
		return
	}
	if len(vd.Rows) == 0 {
		return
	}
	ageSec := float64(time.Since(programStartTime)) / float64(time.Second)

	computeRate := func(maxSec, x float64) float64 {
		dur := ageSec
		if maxSec > 0 && dur > maxSec {
			dur = maxSec
		}
		return x / dur
	}

	convertTime := func(ms float64) time.Duration {
		if math.IsInf(ms, 0) || math.IsNaN(ms) {
			return 0
		}
		return time.Duration(float64(time.Millisecond) * ms)
	}

	haveResetErrors := make(map[string]struct{})

	mu.Lock()
	defer mu.Unlock()
	for _, row := range vd.Rows {
		var method string
		for _, tag := range row.Tags {
			if tag.Key == ocgrpc.KeyClientMethod || tag.Key == ocgrpc.KeyServerMethod {
				method = tag.Value
				break
			}
		}

		key := methodKey{method: method, received: received}
		s := snaps[key]
		if s == nil {
			s = &statSnapshot{Method: method, Received: received}
			snaps[key] = s
		}

		var (
			sum   float64
			count float64
		)
		switch v := row.Data.(type) {
		case *view.CountData:
			sum = float64(v.Value)
			count = float64(v.Value)
		case *view.DistributionData:
			sum = v.Sum()
			count = float64(v.Count)
		case *view.SumData:
			sum = v.Value
			count = v.Value
		}

		// Update field of s corresponding to the view.
		switch vd.View {
		case ocgrpc.ClientCompletedRPCsView:
			if _, ok := haveResetErrors[method]; !ok {
				haveResetErrors[method] = struct{}{}
				s.ErrorsTotal = 0
			}
			for _, tag := range row.Tags {
				if tag.Key == ocgrpc.KeyClientStatus && tag.Value != "OK" {
					s.ErrorsTotal += uint64(count)
				}
			}

		case ocgrpc.ClientRoundtripLatencyView:
			s.AvgLatencyTotal = convertTime(sum / count)

		case ocgrpc.ClientSentBytesPerRPCView:
			s.OutputRateTotal = computeRate(0, sum)

		case ocgrpc.ClientReceivedBytesPerRPCView:
			s.InputRateTotal = computeRate(0, sum)

		case ocgrpc.ClientSentMessagesPerRPCView:
			s.CountTotal = uint64(count)
			s.RPCRateTotal = computeRate(0, count)

		case ocgrpc.ClientReceivedMessagesPerRPCView:
			// currently unused

		case ocgrpc.ServerCompletedRPCsView:
			if _, ok := haveResetErrors[method]; !ok {
				haveResetErrors[method] = struct{}{}
				s.ErrorsTotal = 0
			}
			for _, tag := range row.Tags {
				if tag.Key == ocgrpc.KeyServerStatus && tag.Value != "OK" {
					s.ErrorsTotal += uint64(count)
				}
			}

		case ocgrpc.ServerLatencyView:
			s.AvgLatencyTotal = convertTime(sum / count)

		case ocgrpc.ServerSentBytesPerRPCView:
			s.OutputRateTotal = computeRate(0, sum)

		case ocgrpc.ServerReceivedMessagesPerRPCView:
			s.CountTotal = uint64(count)
			s.RPCRateTotal = computeRate(0, count)

		case ocgrpc.ServerSentMessagesPerRPCView:
			// currently unused
		}
	}
}

func getStatsPage() *statsPage {
	sentStats := statGroup{Direction: "Sent"}
	receivedStats := statGroup{Direction: "Received"}
	for key, sg := range snaps {
		if key.received {
			receivedStats.Snapshots = append(receivedStats.Snapshots, sg)
		} else {
			sentStats.Snapshots = append(sentStats.Snapshots, sg)
		}
	}
	sort.Sort(&sentStats)
	sort.Sort(&receivedStats)

	return &statsPage{
		StatGroups: []*statGroup{&sentStats, &receivedStats},
	}
}
