// Copyright 2015 The Prometheus Authors
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

package expfmt

import (
	"fmt"
	"io"
	"net/http"

	"bitbucket.org/ww/goautoneg"
	"github.com/golang/protobuf/proto"
	"github.com/matttproud/golang_protobuf_extensions/pbutil"

	dto "github.com/prometheus/client_model/go"
)

// Encoder types encode metric families into an underlying wire protocol.
type Encoder interface {
	Encode(*dto.MetricFamily) error
}

type encoder func(*dto.MetricFamily) error

func (e encoder) Encode(v *dto.MetricFamily) error {
	return e(v)
}

// Negotiate returns the Content-Type based on the given Accept header.
// If no appropriate accepted type is found, FmtText is returned.
func Negotiate(h http.Header) Format {
	for _, ac := range goautoneg.ParseAccept(h.Get(hdrAccept)) {
		// Check for protocol buffer
		if ac.Type+"/"+ac.SubType == ProtoType && ac.Params["proto"] == ProtoProtocol {
			switch ac.Params["encoding"] {
			case "delimited":
				return FmtProtoDelim
			case "text":
				return FmtProtoText
			case "compact-text":
				return FmtProtoCompact
			}
		}
		// Check for text format.
		ver := ac.Params["version"]
		if ac.Type == "text" && ac.SubType == "plain" && (ver == TextVersion || ver == "") {
			return FmtText
		}
	}
	return FmtText
}

// NewEncoder returns a new encoder based on content type negotiation.
func NewEncoder(w io.Writer, format Format) Encoder {
	switch format {
	case FmtProtoDelim:
		return encoder(func(v *dto.MetricFamily) error {
			_, err := pbutil.WriteDelimited(w, v)
			return err
		})
	case FmtProtoCompact:
		return encoder(func(v *dto.MetricFamily) error {
			_, err := fmt.Fprintln(w, v.String())
			return err
		})
	case FmtProtoText:
		return encoder(func(v *dto.MetricFamily) error {
			_, err := fmt.Fprintln(w, proto.MarshalTextString(v))
			return err
		})
	case FmtText:
		return encoder(func(v *dto.MetricFamily) error {
			_, err := MetricFamilyToText(w, v)
			return err
		})
	}
	panic("expfmt.NewEncoder: unknown format")
}
