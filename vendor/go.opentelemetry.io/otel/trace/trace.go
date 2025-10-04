// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"encoding/json"
)

const (
	// FlagsSampled is a bitmask with the sampled bit set. A SpanContext
	// with the sampling bit set means the span is sampled.
	FlagsSampled = TraceFlags(0x01)

	errInvalidHexID errorConst = "trace-id and span-id can only contain [0-9a-f] characters, all lowercase"

	errInvalidTraceIDLength errorConst = "hex encoded trace-id must have length equals to 32"
	errNilTraceID           errorConst = "trace-id can't be all zero"

	errInvalidSpanIDLength errorConst = "hex encoded span-id must have length equals to 16"
	errNilSpanID           errorConst = "span-id can't be all zero"
)

type errorConst string

func (e errorConst) Error() string {
	return string(e)
}

// TraceID is a unique identity of a trace.
// nolint:revive // revive complains about stutter of `trace.TraceID`.
type TraceID [16]byte

var (
	nilTraceID TraceID
	_          json.Marshaler = nilTraceID
)

// IsValid reports whether the trace TraceID is valid. A valid trace ID does
// not consist of zeros only.
func (t TraceID) IsValid() bool {
	return t != nilTraceID
}

// MarshalJSON implements a custom marshal function to encode TraceID
// as a hex string.
func (t TraceID) MarshalJSON() ([]byte, error) {
	b := [32 + 2]byte{0: '"', 33: '"'}
	h := t.hexBytes()
	copy(b[1:], h[:])
	return b[:], nil
}

// String returns the hex string representation form of a TraceID.
func (t TraceID) String() string {
	h := t.hexBytes()
	return string(h[:])
}

// hexBytes returns the hex string representation form of a TraceID.
func (t TraceID) hexBytes() [32]byte {
	return [32]byte{
		hexLU[t[0x0]>>4], hexLU[t[0x0]&0xf],
		hexLU[t[0x1]>>4], hexLU[t[0x1]&0xf],
		hexLU[t[0x2]>>4], hexLU[t[0x2]&0xf],
		hexLU[t[0x3]>>4], hexLU[t[0x3]&0xf],
		hexLU[t[0x4]>>4], hexLU[t[0x4]&0xf],
		hexLU[t[0x5]>>4], hexLU[t[0x5]&0xf],
		hexLU[t[0x6]>>4], hexLU[t[0x6]&0xf],
		hexLU[t[0x7]>>4], hexLU[t[0x7]&0xf],
		hexLU[t[0x8]>>4], hexLU[t[0x8]&0xf],
		hexLU[t[0x9]>>4], hexLU[t[0x9]&0xf],
		hexLU[t[0xa]>>4], hexLU[t[0xa]&0xf],
		hexLU[t[0xb]>>4], hexLU[t[0xb]&0xf],
		hexLU[t[0xc]>>4], hexLU[t[0xc]&0xf],
		hexLU[t[0xd]>>4], hexLU[t[0xd]&0xf],
		hexLU[t[0xe]>>4], hexLU[t[0xe]&0xf],
		hexLU[t[0xf]>>4], hexLU[t[0xf]&0xf],
	}
}

// SpanID is a unique identity of a span in a trace.
type SpanID [8]byte

var (
	nilSpanID SpanID
	_         json.Marshaler = nilSpanID
)

// IsValid reports whether the SpanID is valid. A valid SpanID does not consist
// of zeros only.
func (s SpanID) IsValid() bool {
	return s != nilSpanID
}

// MarshalJSON implements a custom marshal function to encode SpanID
// as a hex string.
func (s SpanID) MarshalJSON() ([]byte, error) {
	b := [16 + 2]byte{0: '"', 17: '"'}
	h := s.hexBytes()
	copy(b[1:], h[:])
	return b[:], nil
}

// String returns the hex string representation form of a SpanID.
func (s SpanID) String() string {
	b := s.hexBytes()
	return string(b[:])
}

func (s SpanID) hexBytes() [16]byte {
	return [16]byte{
		hexLU[s[0]>>4], hexLU[s[0]&0xf],
		hexLU[s[1]>>4], hexLU[s[1]&0xf],
		hexLU[s[2]>>4], hexLU[s[2]&0xf],
		hexLU[s[3]>>4], hexLU[s[3]&0xf],
		hexLU[s[4]>>4], hexLU[s[4]&0xf],
		hexLU[s[5]>>4], hexLU[s[5]&0xf],
		hexLU[s[6]>>4], hexLU[s[6]&0xf],
		hexLU[s[7]>>4], hexLU[s[7]&0xf],
	}
}

// TraceIDFromHex returns a TraceID from a hex string if it is compliant with
// the W3C trace-context specification.  See more at
// https://www.w3.org/TR/trace-context/#trace-id
// nolint:revive // revive complains about stutter of `trace.TraceIDFromHex`.
func TraceIDFromHex(h string) (TraceID, error) {
	if len(h) != 32 {
		return [16]byte{}, errInvalidTraceIDLength
	}
	var b [16]byte
	invalidMark := byte(0)
	for i := 0; i < len(h); i += 4 {
		b[i/2] = (hexRev[h[i]] << 4) | hexRev[h[i+1]]
		b[i/2+1] = (hexRev[h[i+2]] << 4) | hexRev[h[i+3]]
		invalidMark |= hexRev[h[i]] | hexRev[h[i+1]] | hexRev[h[i+2]] | hexRev[h[i+3]]
	}
	// If the upper 4 bits of any byte are not zero, there was an invalid hex
	// character since invalid hex characters are 0xff in hexRev.
	if invalidMark&0xf0 != 0 {
		return [16]byte{}, errInvalidHexID
	}
	// If we didn't set any bits, then h was all zeros.
	if invalidMark == 0 {
		return [16]byte{}, errNilTraceID
	}
	return b, nil
}

// SpanIDFromHex returns a SpanID from a hex string if it is compliant
// with the w3c trace-context specification.
// See more at https://www.w3.org/TR/trace-context/#parent-id
func SpanIDFromHex(h string) (SpanID, error) {
	if len(h) != 16 {
		return [8]byte{}, errInvalidSpanIDLength
	}
	var b [8]byte
	invalidMark := byte(0)
	for i := 0; i < len(h); i += 4 {
		b[i/2] = (hexRev[h[i]] << 4) | hexRev[h[i+1]]
		b[i/2+1] = (hexRev[h[i+2]] << 4) | hexRev[h[i+3]]
		invalidMark |= hexRev[h[i]] | hexRev[h[i+1]] | hexRev[h[i+2]] | hexRev[h[i+3]]
	}
	// If the upper 4 bits of any byte are not zero, there was an invalid hex
	// character since invalid hex characters are 0xff in hexRev.
	if invalidMark&0xf0 != 0 {
		return [8]byte{}, errInvalidHexID
	}
	// If we didn't set any bits, then h was all zeros.
	if invalidMark == 0 {
		return [8]byte{}, errNilSpanID
	}
	return b, nil
}

// TraceFlags contains flags that can be set on a SpanContext.
type TraceFlags byte //nolint:revive // revive complains about stutter of `trace.TraceFlags`.

// IsSampled reports whether the sampling bit is set in the TraceFlags.
func (tf TraceFlags) IsSampled() bool {
	return tf&FlagsSampled == FlagsSampled
}

// WithSampled sets the sampling bit in a new copy of the TraceFlags.
func (tf TraceFlags) WithSampled(sampled bool) TraceFlags { // nolint:revive  // sampled is not a control flag.
	if sampled {
		return tf | FlagsSampled
	}

	return tf &^ FlagsSampled
}

// MarshalJSON implements a custom marshal function to encode TraceFlags
// as a hex string.
func (tf TraceFlags) MarshalJSON() ([]byte, error) {
	b := [2 + 2]byte{0: '"', 3: '"'}
	h := tf.hexBytes()
	copy(b[1:], h[:])
	return b[:], nil
}

// String returns the hex string representation form of TraceFlags.
func (tf TraceFlags) String() string {
	h := tf.hexBytes()
	return string(h[:])
}

func (tf TraceFlags) hexBytes() [2]byte {
	return [2]byte{hexLU[tf>>4], hexLU[tf&0xf]}
}

// SpanContextConfig contains mutable fields usable for constructing
// an immutable SpanContext.
type SpanContextConfig struct {
	TraceID    TraceID
	SpanID     SpanID
	TraceFlags TraceFlags
	TraceState TraceState
	Remote     bool
}

// NewSpanContext constructs a SpanContext using values from the provided
// SpanContextConfig.
func NewSpanContext(config SpanContextConfig) SpanContext {
	return SpanContext{
		traceID:    config.TraceID,
		spanID:     config.SpanID,
		traceFlags: config.TraceFlags,
		traceState: config.TraceState,
		remote:     config.Remote,
	}
}

// SpanContext contains identifying trace information about a Span.
type SpanContext struct {
	traceID    TraceID
	spanID     SpanID
	traceFlags TraceFlags
	traceState TraceState
	remote     bool
}

var _ json.Marshaler = SpanContext{}

// IsValid reports whether the SpanContext is valid. A valid span context has a
// valid TraceID and SpanID.
func (sc SpanContext) IsValid() bool {
	return sc.HasTraceID() && sc.HasSpanID()
}

// IsRemote reports whether the SpanContext represents a remotely-created Span.
func (sc SpanContext) IsRemote() bool {
	return sc.remote
}

// WithRemote returns a copy of sc with the Remote property set to remote.
func (sc SpanContext) WithRemote(remote bool) SpanContext {
	return SpanContext{
		traceID:    sc.traceID,
		spanID:     sc.spanID,
		traceFlags: sc.traceFlags,
		traceState: sc.traceState,
		remote:     remote,
	}
}

// TraceID returns the TraceID from the SpanContext.
func (sc SpanContext) TraceID() TraceID {
	return sc.traceID
}

// HasTraceID reports whether the SpanContext has a valid TraceID.
func (sc SpanContext) HasTraceID() bool {
	return sc.traceID.IsValid()
}

// WithTraceID returns a new SpanContext with the TraceID replaced.
func (sc SpanContext) WithTraceID(traceID TraceID) SpanContext {
	return SpanContext{
		traceID:    traceID,
		spanID:     sc.spanID,
		traceFlags: sc.traceFlags,
		traceState: sc.traceState,
		remote:     sc.remote,
	}
}

// SpanID returns the SpanID from the SpanContext.
func (sc SpanContext) SpanID() SpanID {
	return sc.spanID
}

// HasSpanID reports whether the SpanContext has a valid SpanID.
func (sc SpanContext) HasSpanID() bool {
	return sc.spanID.IsValid()
}

// WithSpanID returns a new SpanContext with the SpanID replaced.
func (sc SpanContext) WithSpanID(spanID SpanID) SpanContext {
	return SpanContext{
		traceID:    sc.traceID,
		spanID:     spanID,
		traceFlags: sc.traceFlags,
		traceState: sc.traceState,
		remote:     sc.remote,
	}
}

// TraceFlags returns the flags from the SpanContext.
func (sc SpanContext) TraceFlags() TraceFlags {
	return sc.traceFlags
}

// IsSampled reports whether the sampling bit is set in the SpanContext's TraceFlags.
func (sc SpanContext) IsSampled() bool {
	return sc.traceFlags.IsSampled()
}

// WithTraceFlags returns a new SpanContext with the TraceFlags replaced.
func (sc SpanContext) WithTraceFlags(flags TraceFlags) SpanContext {
	return SpanContext{
		traceID:    sc.traceID,
		spanID:     sc.spanID,
		traceFlags: flags,
		traceState: sc.traceState,
		remote:     sc.remote,
	}
}

// TraceState returns the TraceState from the SpanContext.
func (sc SpanContext) TraceState() TraceState {
	return sc.traceState
}

// WithTraceState returns a new SpanContext with the TraceState replaced.
func (sc SpanContext) WithTraceState(state TraceState) SpanContext {
	return SpanContext{
		traceID:    sc.traceID,
		spanID:     sc.spanID,
		traceFlags: sc.traceFlags,
		traceState: state,
		remote:     sc.remote,
	}
}

// Equal reports whether two SpanContext values are equal.
func (sc SpanContext) Equal(other SpanContext) bool {
	return sc.traceID == other.traceID &&
		sc.spanID == other.spanID &&
		sc.traceFlags == other.traceFlags &&
		sc.traceState.String() == other.traceState.String() &&
		sc.remote == other.remote
}

// MarshalJSON implements a custom marshal function to encode a SpanContext.
func (sc SpanContext) MarshalJSON() ([]byte, error) {
	return json.Marshal(SpanContextConfig{
		TraceID:    sc.traceID,
		SpanID:     sc.spanID,
		TraceFlags: sc.traceFlags,
		TraceState: sc.traceState,
		Remote:     sc.remote,
	})
}
