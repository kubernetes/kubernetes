// Copyright 2018, OpenCensus Authors
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

package zpages

import (
	"bytes"
	"html/template"
	"testing"
)

const tmplBody = `
        <td><b>{{.Method}}</b></td>
        <td></td>
        <td align="right">{{.CountMinute|count}}</td>
        <td align="right">{{.CountHour|count}}</td>
        <td align="right">{{.CountTotal|count}}</td><td></td>
        <td align="right">{{.AvgLatencyMinute|ms}}</td>
        <td align="right">{{.AvgLatencyHour|ms}}</td>
        <td align="right">{{.AvgLatencyTotal|ms}}</td><td></td>
        <td align="right">{{.RPCRateMinute|rate}}</td>
        <td align="right">{{.RPCRateHour|rate}}</td>
        <td align="right">{{.RPCRateTotal|rate}}</td><td></td>
        <td align="right">{{.InputRateMinute|datarate}}</td>
        <td align="right">{{.InputRateHour|datarate}}</td>
        <td align="right">{{.InputRateTotal|datarate}}</td><td></td>
        <td align="right">{{.OutputRateMinute|datarate}}</td>
        <td align="right">{{.OutputRateHour|datarate}}</td>
        <td align="right">{{.OutputRateTotal|datarate}}</td><td></td>
        <td align="right">{{.ErrorsMinute|count}}</td>
        <td align="right">{{.ErrorsHour|count}}</td>
        <td align="right">{{.ErrorsTotal|count}}</td><td></td>
`

var tmpl = template.Must(template.New("countTest").Funcs(templateFunctions).Parse(tmplBody))

func TestTemplateFuncs(t *testing.T) {
	buf := new(bytes.Buffer)
	sshot := &statSnapshot{
		Method:           "Foo",
		CountMinute:      1e9,
		CountHour:        5000,
		CountTotal:       1e12,
		AvgLatencyMinute: 10000,
		AvgLatencyHour:   1000,
		AvgLatencyTotal:  20000,
		RPCRateMinute:    2000,
		RPCRateHour:      5000,
		RPCRateTotal:     75000,
		InputRateMinute:  75000,
		InputRateHour:    75000,
		InputRateTotal:   75000,
		OutputRateMinute: 75000,
		OutputRateHour:   75000,
		OutputRateTotal:  75000,
		ErrorsMinute:     120000000,
		ErrorsHour:       75000000,
		ErrorsTotal:      7500000,
	}
	if err := tmpl.Execute(buf, sshot); err != nil {
		t.Fatalf("Failed to execute template: %v", err)
	}
	want := `
        <td><b>Foo</b></td>
        <td></td>
        <td align="right">1.000 G </td>
        <td align="right">5000</td>
        <td align="right">1.000 T </td><td></td>
        <td align="right">0.010</td>
        <td align="right">0.001</td>
        <td align="right">0.020</td><td></td>
        <td align="right">2000.000</td>
        <td align="right">5000.000</td>
        <td align="right">75000.000</td><td></td>
        <td align="right">0.075</td>
        <td align="right">0.075</td>
        <td align="right">0.075</td><td></td>
        <td align="right">0.075</td>
        <td align="right">0.075</td>
        <td align="right">0.075</td><td></td>
        <td align="right">120.000 M </td>
        <td align="right">75.000 M </td>
        <td align="right">7.500 M </td><td></td>
`
	if g, w := buf.String(), want; g != w {
		t.Errorf("Output mismatch:\nGot:\n\t%s\nWant:\n\t%s", g, w)
	}
}
