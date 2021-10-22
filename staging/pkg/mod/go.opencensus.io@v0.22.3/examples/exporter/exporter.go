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

package exporter // import "go.opencensus.io/examples/exporter"

import (
	"encoding/hex"
	"fmt"
	"regexp"
	"time"

	"go.opencensus.io/stats/view"
	"go.opencensus.io/trace"
)

// indent these many spaces
const indent = "  "

// reZero provides a simple way to detect an empty ID
var reZero = regexp.MustCompile(`^0+$`)

// PrintExporter is a stats and trace exporter that logs
// the exported data to the console.
//
// The intent is help new users familiarize themselves with the
// capabilities of opencensus.
//
// This should NOT be used for production workloads.
type PrintExporter struct{}

// ExportView logs the view data.
func (e *PrintExporter) ExportView(vd *view.Data) {
	for _, row := range vd.Rows {
		fmt.Printf("%v %-45s", vd.End.Format("15:04:05"), vd.View.Name)

		switch v := row.Data.(type) {
		case *view.DistributionData:
			fmt.Printf("distribution: min=%.1f max=%.1f mean=%.1f", v.Min, v.Max, v.Mean)
		case *view.CountData:
			fmt.Printf("count:        value=%v", v.Value)
		case *view.SumData:
			fmt.Printf("sum:          value=%v", v.Value)
		case *view.LastValueData:
			fmt.Printf("last:         value=%v", v.Value)
		}
		fmt.Println()

		for _, tag := range row.Tags {
			fmt.Printf("%v- %v=%v\n", indent, tag.Key.Name(), tag.Value)
		}
	}
}

// ExportSpan logs the trace span.
func (e *PrintExporter) ExportSpan(vd *trace.SpanData) {
	var (
		traceID      = hex.EncodeToString(vd.SpanContext.TraceID[:])
		spanID       = hex.EncodeToString(vd.SpanContext.SpanID[:])
		parentSpanID = hex.EncodeToString(vd.ParentSpanID[:])
	)
	fmt.Println()
	fmt.Println("#----------------------------------------------")
	fmt.Println()
	fmt.Println("TraceID:     ", traceID)
	fmt.Println("SpanID:      ", spanID)
	if !reZero.MatchString(parentSpanID) {
		fmt.Println("ParentSpanID:", parentSpanID)
	}

	fmt.Println()
	fmt.Printf("Span:    %v\n", vd.Name)
	fmt.Printf("Status:  %v [%v]\n", vd.Status.Message, vd.Status.Code)
	fmt.Printf("Elapsed: %v\n", vd.EndTime.Sub(vd.StartTime).Round(time.Millisecond))

	if len(vd.Annotations) > 0 {
		fmt.Println()
		fmt.Println("Annotations:")
		for _, item := range vd.Annotations {
			fmt.Print(indent, item.Message)
			for k, v := range item.Attributes {
				fmt.Printf(" %v=%v", k, v)
			}
			fmt.Println()
		}
	}

	if len(vd.Attributes) > 0 {
		fmt.Println()
		fmt.Println("Attributes:")
		for k, v := range vd.Attributes {
			fmt.Printf("%v- %v=%v\n", indent, k, v)
		}
	}
}
