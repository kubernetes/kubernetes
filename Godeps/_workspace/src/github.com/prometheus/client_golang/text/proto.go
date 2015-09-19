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

package text

import (
	"fmt"
	"io"

	"github.com/golang/protobuf/proto"
	"github.com/matttproud/golang_protobuf_extensions/pbutil"

	dto "github.com/prometheus/client_model/go"
)

// WriteProtoDelimited writes the MetricFamily to the writer in delimited
// protobuf format and returns the number of bytes written and any error
// encountered.
func WriteProtoDelimited(w io.Writer, p *dto.MetricFamily) (int, error) {
	return pbutil.WriteDelimited(w, p)
}

// WriteProtoText writes the MetricFamily to the writer in text format and
// returns the number of bytes written and any error encountered.
func WriteProtoText(w io.Writer, p *dto.MetricFamily) (int, error) {
	return fmt.Fprintf(w, "%s\n", proto.MarshalTextString(p))
}

// WriteProtoCompactText writes the MetricFamily to the writer in compact text
// format and returns the number of bytes written and any error encountered.
func WriteProtoCompactText(w io.Writer, p *dto.MetricFamily) (int, error) {
	return fmt.Fprintf(w, "%s\n", p)
}
