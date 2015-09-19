// Copyright 2014 The Prometheus Authors
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

package extraction

import (
	"io"

	"github.com/prometheus/client_golang/text"
)

type processor004 struct{}

// Processor004 s responsible for decoding payloads from the text based variety
// of protocol version 0.0.4.
var Processor004 = &processor004{}

func (t *processor004) ProcessSingle(i io.Reader, out Ingester, o *ProcessOptions) error {
	var parser text.Parser
	metricFamilies, err := parser.TextToMetricFamilies(i)
	if err != nil {
		return err
	}
	for _, metricFamily := range metricFamilies {
		if err := extractMetricFamily(out, o, metricFamily); err != nil {
			return err
		}
	}
	return nil
}
