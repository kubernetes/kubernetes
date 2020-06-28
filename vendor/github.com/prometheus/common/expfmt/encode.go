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

	"github.com/golang/protobuf/proto"
	"github.com/matttproud/golang_protobuf_extensions/pbutil"
	"github.com/prometheus/common/internal/bitbucket.org/ww/goautoneg"

	dto "github.com/prometheus/client_model/go"
)

// Encoder types encode metric families into an underlying wire protocol.
type Encoder interface {
	Encode(*dto.MetricFamily) error
}

// Closer is implemented by Encoders that need to be closed to finalize
// encoding. (For example, OpenMetrics needs a final `# EOF` line.)
//
// Note that all Encoder implementations returned from this package implement
// Closer, too, even if the Close call is a no-op. This happens in preparation
// for adding a Close method to the Encoder interface directly in a (mildly
// breaking) release in the future.
type Closer interface {
	Close() error
}

type encoderCloser struct {
	encode func(*dto.MetricFamily) error
	close  func() error
}

func (ec encoderCloser) Encode(v *dto.MetricFamily) error {
	return ec.encode(v)
}

func (ec encoderCloser) Close() error {
	return ec.close()
}

// Negotiate returns the Content-Type based on the given Accept header. If no
// appropriate accepted type is found, FmtText is returned (which is the
// Prometheus text format). This function will never negotiate FmtOpenMetrics,
// as the support is still experimental. To include the option to negotiate
// FmtOpenMetrics, use NegotiateOpenMetrics.
func Negotiate(h http.Header) Format {
	for _, ac := range goautoneg.ParseAccept(h.Get(hdrAccept)) {
		ver := ac.Params["version"]
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
		if ac.Type == "text" && ac.SubType == "plain" && (ver == TextVersion || ver == "") {
			return FmtText
		}
	}
	return FmtText
}

// NegotiateIncludingOpenMetrics works like Negotiate but includes
// FmtOpenMetrics as an option for the result. Note that this function is
// temporary and will disappear once FmtOpenMetrics is fully supported and as
// such may be negotiated by the normal Negotiate function.
func NegotiateIncludingOpenMetrics(h http.Header) Format {
	for _, ac := range goautoneg.ParseAccept(h.Get(hdrAccept)) {
		ver := ac.Params["version"]
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
		if ac.Type == "text" && ac.SubType == "plain" && (ver == TextVersion || ver == "") {
			return FmtText
		}
		if ac.Type+"/"+ac.SubType == OpenMetricsType && (ver == OpenMetricsVersion || ver == "") {
			return FmtOpenMetrics
		}
	}
	return FmtText
}

// NewEncoder returns a new encoder based on content type negotiation. All
// Encoder implementations returned by NewEncoder also implement Closer, and
// callers should always call the Close method. It is currently only required
// for FmtOpenMetrics, but a future (breaking) release will add the Close method
// to the Encoder interface directly. The current version of the Encoder
// interface is kept for backwards compatibility.
func NewEncoder(w io.Writer, format Format) Encoder {
	switch format {
	case FmtProtoDelim:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := pbutil.WriteDelimited(w, v)
				return err
			},
			close: func() error { return nil },
		}
	case FmtProtoCompact:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := fmt.Fprintln(w, v.String())
				return err
			},
			close: func() error { return nil },
		}
	case FmtProtoText:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := fmt.Fprintln(w, proto.MarshalTextString(v))
				return err
			},
			close: func() error { return nil },
		}
	case FmtText:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := MetricFamilyToText(w, v)
				return err
			},
			close: func() error { return nil },
		}
	case FmtOpenMetrics:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := MetricFamilyToOpenMetrics(w, v)
				return err
			},
			close: func() error {
				_, err := FinalizeOpenMetrics(w)
				return err
			},
		}
	}
	panic(fmt.Errorf("expfmt.NewEncoder: unknown format %q", format))
}
