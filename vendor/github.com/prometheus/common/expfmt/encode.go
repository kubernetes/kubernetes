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
	"mime"
	"net/http"

	"github.com/munnerz/goautoneg"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/prototext"

	"github.com/prometheus/common/model"
)

var formatToAccept = map[Format]goautoneg.Accept{}

func init() {
	for _, f := range []Format{
		FmtText,
		FmtProtoDelim,
		FmtProtoText,
		FmtProtoCompact,
		FmtOpenMetrics_1_0_0,
		fmtOpenMetrics_2_0_0,
		FmtOpenMetrics_0_0_1,
	} {
		if parsed := goautoneg.ParseAccept(string(f)); len(parsed) > 0 {
			formatToAccept[f] = parsed[0]
		}
	}
}

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
// Prometheus text format).
//
// Deprecated: Use NegotiateAccept(h, FmtProtoDelim, FmtProtoText, FmtProtoCompact, FmtText)
// or specify only the formats supported by your server.
func Negotiate(h http.Header) Format {
	return NegotiateAccept(h, FmtProtoDelim, FmtProtoText, FmtProtoCompact, FmtText)
}

// NegotiateIncludingOpenMetrics works like Negotiate but includes
// FmtOpenMetrics as an option for the result.
//
// Deprecated: Use NegotiateAccept(h, FmtOpenMetrics_1_0_0, FmtOpenMetrics_0_0_1, FmtProtoDelim, FmtProtoText, FmtProtoCompact, FmtText)
// or specify only the formats supported by your server.
func NegotiateIncludingOpenMetrics(h http.Header) Format {
	return NegotiateAccept(h, FmtOpenMetrics_1_0_0, FmtOpenMetrics_0_0_1, FmtProtoDelim, FmtProtoText, FmtProtoCompact, FmtText)
}

// NegotiateAccept returns the Content-Type based on the given Accept header
// and the list of accepted Formats provided by the caller, in order of preference.
// If no accepted format matches the Accept header, it falls back to the text
// format if present in the accepted list, or the first accepted format (or FmtText
// if accepted is empty).
func NegotiateAccept(h http.Header, accepted ...Format) Format {
	escapingScheme := Format("; escaping=" + model.NameEscapingScheme.String())
	for _, ac := range goautoneg.ParseAccept(h.Get(hdrAccept)) {
		if escapeParam := ac.Params[model.EscapingKey]; escapeParam != "" {
			switch Format(escapeParam) {
			case model.AllowUTF8, model.EscapeUnderscores, model.EscapeDots, model.EscapeValues:
				escapingScheme = Format("; escaping=" + escapeParam)
			default:
				// If the escaping parameter is unknown, ignore it.
			}
		}

		for _, f := range accepted {
			if matchFormat(ac, f) {
				return f + escapingScheme
			}
		}
	}
	for _, f := range accepted {
		if f.FormatType() == TypeTextPlain {
			return f + escapingScheme
		}
	}
	if len(accepted) > 0 {
		return accepted[0] + escapingScheme
	}
	return FmtText + escapingScheme
}

// matchFormat checks if a parsed accept clause matches a given Format.
func matchFormat(ac goautoneg.Accept, f Format) bool {
	target, ok := formatToAccept[f]
	if !ok {
		parsed := goautoneg.ParseAccept(string(f))
		if len(parsed) == 0 {
			return false
		}
		target = parsed[0]
	}

	if ac.Type != "*" && ac.Type != target.Type {
		return false
	}
	if ac.SubType != "*" && ac.SubType != target.SubType {
		return false
	}

	// Default OpenMetrics version to OpenMetricsVersion_0_0_1.
	acVersion := ac.Params["version"]
	if acVersion == "" && ac.Type+"/"+ac.SubType == OpenMetricsType {
		acVersion = OpenMetricsVersion_0_0_1
	}
	if acVersion == "" && ac.Type == "text" && ac.SubType == "plain" {
		acVersion = TextVersion
	}

	// General param matching.
	for k, v := range target.Params {
		if k == "charset" {
			continue
		}
		acVal := ac.Params[k]
		if k == "version" {
			acVal = acVersion
		}
		if acVal != v {
			return false
		}
	}

	return true
}

// NewEncoder returns a new encoder based on content type negotiation. All
// Encoder implementations returned by NewEncoder also implement Closer, and
// callers should always call the Close method. It is currently only required
// for FmtOpenMetrics, but a future (breaking) release will add the Close method
// to the Encoder interface directly. The current version of the Encoder
// interface is kept for backwards compatibility.
// In cases where the Format does not allow for UTF-8 names, the global
// NameEscapingScheme will be applied.
//
// NewEncoder can be called with additional options to customize the OpenMetrics text output.
// For example:
// NewEncoder(w, FmtOpenMetrics_1_0_0, WithCreatedLines())
//
// Extra options are ignored for all other formats.
func NewEncoder(w io.Writer, format Format, options ...EncoderOption) Encoder {
	escapingScheme := format.ToEscapingScheme()

	switch format.FormatType() {
	case TypeProtoDelim:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := protodelim.MarshalTo(w, model.EscapeMetricFamily(v, escapingScheme))
				return err
			},
			close: func() error { return nil },
		}
	case TypeProtoCompact:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := fmt.Fprintln(w, model.EscapeMetricFamily(v, escapingScheme).String())
				return err
			},
			close: func() error { return nil },
		}
	case TypeProtoText:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := fmt.Fprintln(w, prototext.Format(model.EscapeMetricFamily(v, escapingScheme)))
				return err
			},
			close: func() error { return nil },
		}
	case TypeTextPlain:
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := MetricFamilyToText(w, model.EscapeMetricFamily(v, escapingScheme))
				return err
			},
			close: func() error { return nil },
		}
	case TypeOpenMetrics:
		_, params, err := mime.ParseMediaType(string(format))
		if err == nil && params["version"] == OpenMetricsVersion_2_0_0 {
			return encoderCloser{
				encode: func(v *dto.MetricFamily) error {
					_, err := MetricFamilyToOpenMetrics20(w, model.EscapeMetricFamily(v, escapingScheme), options...)
					return err
				},
				close: func() error {
					_, err := FinalizeOpenMetrics(w)
					return err
				},
			}
		}
		return encoderCloser{
			encode: func(v *dto.MetricFamily) error {
				_, err := MetricFamilyToOpenMetrics(w, model.EscapeMetricFamily(v, escapingScheme), options...)
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
