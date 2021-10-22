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
	"html/template"
	"io/ioutil"
	"log"
	"strconv"
	"time"

	"go.opencensus.io/trace"
	"go.opencensus.io/zpages/internal"
)

var (
	fs                = internal.FS(false)
	templateFunctions = template.FuncMap{
		"count":    countFormatter,
		"ms":       msFormatter,
		"rate":     rateFormatter,
		"datarate": dataRateFormatter,
		"even":     even,
		"traceid":  traceIDFormatter,
	}
	headerTemplate       = parseTemplate("header")
	summaryTableTemplate = parseTemplate("summary")
	statsTemplate        = parseTemplate("rpcz")
	tracesTableTemplate  = parseTemplate("traces")
	footerTemplate       = parseTemplate("footer")
)

func parseTemplate(name string) *template.Template {
	f, err := fs.Open("/templates/" + name + ".html")
	if err != nil {
		log.Panicf("%v: %v", name, err)
	}
	defer f.Close()
	text, err := ioutil.ReadAll(f)
	if err != nil {
		log.Panicf("%v: %v", name, err)
	}
	return template.Must(template.New(name).Funcs(templateFunctions).Parse(string(text)))
}

func countFormatter(num uint64) string {
	if num <= 0 {
		return " "
	}
	var floatVal float64
	var suffix string

	if num >= 1e18 {
		floatVal = float64(num) / 1e18
		suffix = " E "
	} else if num >= 1e15 {
		floatVal = float64(num) / 1e15
		suffix = " P "
	} else if num >= 1e12 {
		floatVal = float64(num) / 1e12
		suffix = " T "
	} else if num >= 1e9 {
		floatVal = float64(num) / 1e9
		suffix = " G "
	} else if num >= 1e6 {
		floatVal = float64(num) / 1e6
		suffix = " M "
	}

	if floatVal != 0 {
		return fmt.Sprintf("%1.3f%s", floatVal, suffix)
	}
	return fmt.Sprint(num)
}

func msFormatter(d time.Duration) string {
	if d == 0 {
		return "0"
	}
	if d < 10*time.Millisecond {
		return fmt.Sprintf("%.3f", float64(d)*1e-6)
	}
	return strconv.Itoa(int(d / time.Millisecond))
}

func rateFormatter(r float64) string {
	return fmt.Sprintf("%.3f", r)
}

func dataRateFormatter(b float64) string {
	return fmt.Sprintf("%.3f", b/1e6)
}

func traceIDFormatter(r traceRow) template.HTML {
	sc := r.SpanContext
	if sc == (trace.SpanContext{}) {
		return ""
	}
	col := "black"
	if sc.TraceOptions.IsSampled() {
		col = "blue"
	}
	if r.ParentSpanID != (trace.SpanID{}) {
		return template.HTML(fmt.Sprintf(`trace_id: <b style="color:%s">%s</b> span_id: %s parent_span_id: %s`, col, sc.TraceID, sc.SpanID, r.ParentSpanID))
	}
	return template.HTML(fmt.Sprintf(`trace_id: <b style="color:%s">%s</b> span_id: %s`, col, sc.TraceID, sc.SpanID))
}

func even(x int) bool {
	return x%2 == 0
}
