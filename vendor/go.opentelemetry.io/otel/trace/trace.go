// Copyright The OpenTelemetry Authors
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

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
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

	// based on the W3C Trace Context specification, see https://www.w3.org/TR/trace-context-1/#tracestate-header
	traceStateKeyFormat                      = `[a-z][_0-9a-z\-\*\/]{0,255}`
	traceStateKeyFormatWithMultiTenantVendor = `[a-z0-9][_0-9a-z\-\*\/]{0,240}@[a-z][_0-9a-z\-\*\/]{0,13}`
	traceStateValueFormat                    = `[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]`

	traceStateMaxListMembers = 32

	errInvalidTraceStateKeyValue errorConst = "provided key or value is not valid according to the" +
		" W3C Trace Context specification"
	errInvalidTraceStateMembersNumber errorConst = "trace state would exceed the maximum limit of members (32)"
	errInvalidTraceStateDuplicate     errorConst = "trace state key/value pairs with duplicate keys provided"
)

type errorConst string

func (e errorConst) Error() string {
	return string(e)
}

// TraceID is a unique identity of a trace.
// nolint:golint
type TraceID [16]byte

var nilTraceID TraceID
var _ json.Marshaler = nilTraceID

// IsValid checks whether the trace TraceID is valid. A valid trace ID does
// not consist of zeros only.
func (t TraceID) IsValid() bool {
	return !bytes.Equal(t[:], nilTraceID[:])
}

// MarshalJSON implements a custom marshal function to encode TraceID
// as a hex string.
func (t TraceID) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.String())
}

// String returns the hex string representation form of a TraceID
func (t TraceID) String() string {
	return hex.EncodeToString(t[:])
}

// SpanID is a unique identity of a span in a trace.
type SpanID [8]byte

var nilSpanID SpanID
var _ json.Marshaler = nilSpanID

// IsValid checks whether the SpanID is valid. A valid SpanID does not consist
// of zeros only.
func (s SpanID) IsValid() bool {
	return !bytes.Equal(s[:], nilSpanID[:])
}

// MarshalJSON implements a custom marshal function to encode SpanID
// as a hex string.
func (s SpanID) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// String returns the hex string representation form of a SpanID
func (s SpanID) String() string {
	return hex.EncodeToString(s[:])
}

// TraceIDFromHex returns a TraceID from a hex string if it is compliant with
// the W3C trace-context specification.  See more at
// https://www.w3.org/TR/trace-context/#trace-id
// nolint:golint
func TraceIDFromHex(h string) (TraceID, error) {
	t := TraceID{}
	if len(h) != 32 {
		return t, errInvalidTraceIDLength
	}

	if err := decodeHex(h, t[:]); err != nil {
		return t, err
	}

	if !t.IsValid() {
		return t, errNilTraceID
	}
	return t, nil
}

// SpanIDFromHex returns a SpanID from a hex string if it is compliant
// with the w3c trace-context specification.
// See more at https://www.w3.org/TR/trace-context/#parent-id
func SpanIDFromHex(h string) (SpanID, error) {
	s := SpanID{}
	if len(h) != 16 {
		return s, errInvalidSpanIDLength
	}

	if err := decodeHex(h, s[:]); err != nil {
		return s, err
	}

	if !s.IsValid() {
		return s, errNilSpanID
	}
	return s, nil
}

func decodeHex(h string, b []byte) error {
	for _, r := range h {
		switch {
		case 'a' <= r && r <= 'f':
			continue
		case '0' <= r && r <= '9':
			continue
		default:
			return errInvalidHexID
		}
	}

	decoded, err := hex.DecodeString(h)
	if err != nil {
		return err
	}

	copy(b, decoded)
	return nil
}

// TraceState provides additional vendor-specific trace identification information
// across different distributed tracing systems. It represents an immutable list consisting
// of key/value pairs. There can be a maximum of 32 entries in the list.
//
// Key and value of each list member must be valid according to the W3C Trace Context specification
// (see https://www.w3.org/TR/trace-context-1/#key and https://www.w3.org/TR/trace-context-1/#value
// respectively).
//
// Trace state must be valid according to the W3C Trace Context specification at all times. All
// mutating operations validate their input and, in case of valid parameters, return a new TraceState.
type TraceState struct { //nolint:golint
	// TODO @matej-g: Consider implementing this as attribute.Set, see
	// comment https://github.com/open-telemetry/opentelemetry-go/pull/1340#discussion_r540599226
	kvs []attribute.KeyValue
}

var _ json.Marshaler = TraceState{}
var _ json.Marshaler = SpanContext{}

var keyFormatRegExp = regexp.MustCompile(
	`^((` + traceStateKeyFormat + `)|(` + traceStateKeyFormatWithMultiTenantVendor + `))$`,
)
var valueFormatRegExp = regexp.MustCompile(`^(` + traceStateValueFormat + `)$`)

// MarshalJSON implements a custom marshal function to encode trace state.
func (ts TraceState) MarshalJSON() ([]byte, error) {
	return json.Marshal(ts.kvs)
}

// String returns trace state as a string valid according to the
// W3C Trace Context specification.
func (ts TraceState) String() string {
	var sb strings.Builder

	for i, kv := range ts.kvs {
		sb.WriteString((string)(kv.Key))
		sb.WriteByte('=')
		sb.WriteString(kv.Value.Emit())

		if i != len(ts.kvs)-1 {
			sb.WriteByte(',')
		}
	}

	return sb.String()
}

// Get returns a value for given key from the trace state.
// If no key is found or provided key is invalid, returns an empty value.
func (ts TraceState) Get(key attribute.Key) attribute.Value {
	if !isTraceStateKeyValid(key) {
		return attribute.Value{}
	}

	for _, kv := range ts.kvs {
		if kv.Key == key {
			return kv.Value
		}
	}

	return attribute.Value{}
}

// Insert adds a new key/value, if one doesn't exists; otherwise updates the existing entry.
// The new or updated entry is always inserted at the beginning of the TraceState, i.e.
// on the left side, as per the W3C Trace Context specification requirement.
func (ts TraceState) Insert(entry attribute.KeyValue) (TraceState, error) {
	if !isTraceStateKeyValueValid(entry) {
		return ts, errInvalidTraceStateKeyValue
	}

	ckvs := ts.copyKVsAndDeleteEntry(entry.Key)
	if len(ckvs)+1 > traceStateMaxListMembers {
		return ts, errInvalidTraceStateMembersNumber
	}

	ckvs = append(ckvs, attribute.KeyValue{})
	copy(ckvs[1:], ckvs)
	ckvs[0] = entry

	return TraceState{ckvs}, nil
}

// Delete removes specified entry from the trace state.
func (ts TraceState) Delete(key attribute.Key) (TraceState, error) {
	if !isTraceStateKeyValid(key) {
		return ts, errInvalidTraceStateKeyValue
	}

	return TraceState{ts.copyKVsAndDeleteEntry(key)}, nil
}

// IsEmpty returns true if the TraceState does not contain any entries
func (ts TraceState) IsEmpty() bool {
	return len(ts.kvs) == 0
}

func (ts TraceState) copyKVsAndDeleteEntry(key attribute.Key) []attribute.KeyValue {
	ckvs := make([]attribute.KeyValue, len(ts.kvs))
	copy(ckvs, ts.kvs)
	for i, kv := range ts.kvs {
		if kv.Key == key {
			ckvs = append(ckvs[:i], ckvs[i+1:]...)
			break
		}
	}

	return ckvs
}

// TraceStateFromKeyValues is a convenience method to create a new TraceState from
// provided key/value pairs.
func TraceStateFromKeyValues(kvs ...attribute.KeyValue) (TraceState, error) { //nolint:golint
	if len(kvs) == 0 {
		return TraceState{}, nil
	}

	if len(kvs) > traceStateMaxListMembers {
		return TraceState{}, errInvalidTraceStateMembersNumber
	}

	km := make(map[attribute.Key]bool)
	for _, kv := range kvs {
		if !isTraceStateKeyValueValid(kv) {
			return TraceState{}, errInvalidTraceStateKeyValue
		}
		_, ok := km[kv.Key]
		if ok {
			return TraceState{}, errInvalidTraceStateDuplicate
		}
		km[kv.Key] = true
	}

	ckvs := make([]attribute.KeyValue, len(kvs))
	copy(ckvs, kvs)
	return TraceState{ckvs}, nil
}

func isTraceStateKeyValid(key attribute.Key) bool {
	return keyFormatRegExp.MatchString(string(key))
}

func isTraceStateKeyValueValid(kv attribute.KeyValue) bool {
	return isTraceStateKeyValid(kv.Key) &&
		valueFormatRegExp.MatchString(kv.Value.Emit())
}

// TraceFlags contains flags that can be set on a SpanContext
type TraceFlags byte //nolint:golint

// IsSampled returns if the sampling bit is set in the TraceFlags.
func (tf TraceFlags) IsSampled() bool {
	return tf&FlagsSampled == FlagsSampled
}

// WithSampled sets the sampling bit in a new copy of the TraceFlags.
func (tf TraceFlags) WithSampled(sampled bool) TraceFlags {
	if sampled {
		return tf | FlagsSampled
	}

	return tf &^ FlagsSampled
}

// MarshalJSON implements a custom marshal function to encode TraceFlags
// as a hex string.
func (tf TraceFlags) MarshalJSON() ([]byte, error) {
	return json.Marshal(tf.String())
}

// String returns the hex string representation form of TraceFlags
func (tf TraceFlags) String() string {
	return hex.EncodeToString([]byte{byte(tf)}[:])
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

// IsValid returns if the SpanContext is valid. A valid span context has a
// valid TraceID and SpanID.
func (sc SpanContext) IsValid() bool {
	return sc.HasTraceID() && sc.HasSpanID()
}

// IsRemote indicates whether the SpanContext represents a remotely-created Span.
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

// HasTraceID checks if the SpanContext has a valid TraceID.
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

// HasSpanID checks if the SpanContext has a valid SpanID.
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

// IsSampled returns if the sampling bit is set in the SpanContext's TraceFlags.
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

// Equal is a predicate that determines whether two SpanContext values are equal.
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

// Span is the individual component of a trace. It represents a single named
// and timed operation of a workflow that is traced. A Tracer is used to
// create a Span and it is then up to the operation the Span represents to
// properly end the Span when the operation itself ends.
type Span interface {
	// Tracer returns the Tracer that created the Span. Tracer MUST NOT be
	// nil.
	Tracer() Tracer

	// End completes the Span. The Span is considered complete and ready to be
	// delivered through the rest of the telemetry pipeline after this method
	// is called. Therefore, updates to the Span are not allowed after this
	// method has been called.
	End(options ...SpanOption)

	// AddEvent adds an event with the provided name and options.
	AddEvent(name string, options ...EventOption)

	// IsRecording returns the recording state of the Span. It will return
	// true if the Span is active and events can be recorded.
	IsRecording() bool

	// RecordError will record err as an exception span event for this span. An
	// additional call toSetStatus is required if the Status of the Span should
	// be set to Error, this method does not change the Span status. If this
	// span is not being recorded or err is nil than this method does nothing.
	RecordError(err error, options ...EventOption)

	// SpanContext returns the SpanContext of the Span. The returned
	// SpanContext is usable even after the End has been called for the Span.
	SpanContext() SpanContext

	// SetStatus sets the status of the Span in the form of a code and a
	// message. SetStatus overrides the value of previous calls to SetStatus
	// on the Span.
	SetStatus(code codes.Code, msg string)

	// SetName sets the Span name.
	SetName(name string)

	// SetAttributes sets kv as attributes of the Span. If a key from kv
	// already exists for an attribute of the Span it will be overwritten with
	// the value contained in kv.
	SetAttributes(kv ...attribute.KeyValue)
}

// Event is a thing that happened during a Span's lifetime.
type Event struct {
	// Name is the name of this event
	Name string

	// Attributes describe the aspects of the event.
	Attributes []attribute.KeyValue

	// DroppedAttributeCount is the number of attributes that were not
	// recorded due to configured limits being reached.
	DroppedAttributeCount int

	// Time at which this event was recorded.
	Time time.Time
}

// Link is the relationship between two Spans. The relationship can be within
// the same Trace or across different Traces.
//
// For example, a Link is used in the following situations:
//
//   1. Batch Processing: A batch of operations may contain operations
//      associated with one or more traces/spans. Since there can only be one
//      parent SpanContext, a Link is used to keep reference to the
//      SpanContext of all operations in the batch.
//   2. Public Endpoint: A SpanContext for an in incoming client request on a
//      public endpoint should be considered untrusted. In such a case, a new
//      trace with its own identity and sampling decision needs to be created,
//      but this new trace needs to be related to the original trace in some
//      form. A Link is used to keep reference to the original SpanContext and
//      track the relationship.
type Link struct {
	// SpanContext of the linked Span.
	SpanContext

	// Attributes describe the aspects of the link.
	Attributes []attribute.KeyValue

	// DroppedAttributeCount is the number of attributes that were not
	// recorded due to configured limits being reached.
	DroppedAttributeCount int
}

// SpanKind is the role a Span plays in a Trace.
type SpanKind int

// As a convenience, these match the proto definition, see
// https://github.com/open-telemetry/opentelemetry-proto/blob/30d237e1ff3ab7aa50e0922b5bebdd93505090af/opentelemetry/proto/trace/v1/trace.proto#L101-L129
//
// The unspecified value is not a valid `SpanKind`. Use `ValidateSpanKind()`
// to coerce a span kind to a valid value.
const (
	// SpanKindUnspecified is an unspecified SpanKind and is not a valid
	// SpanKind. SpanKindUnspecified should be replaced with SpanKindInternal
	// if it is received.
	SpanKindUnspecified SpanKind = 0
	// SpanKindInternal is a SpanKind for a Span that represents an internal
	// operation within an application.
	SpanKindInternal SpanKind = 1
	// SpanKindServer is a SpanKind for a Span that represents the operation
	// of handling a request from a client.
	SpanKindServer SpanKind = 2
	// SpanKindClient is a SpanKind for a Span that represents the operation
	// of client making a request to a server.
	SpanKindClient SpanKind = 3
	// SpanKindProducer is a SpanKind for a Span that represents the operation
	// of a producer sending a message to a message broker. Unlike
	// SpanKindClient and SpanKindServer, there is often no direct
	// relationship between this kind of Span and a SpanKindConsumer kind. A
	// SpanKindProducer Span will end once the message is accepted by the
	// message broker which might not overlap with the processing of that
	// message.
	SpanKindProducer SpanKind = 4
	// SpanKindConsumer is a SpanKind for a Span that represents the operation
	// of a consumer receiving a message from a message broker. Like
	// SpanKindProducer Spans, there is often no direct relationship between
	// this Span and the Span that produced the message.
	SpanKindConsumer SpanKind = 5
)

// ValidateSpanKind returns a valid span kind value.  This will coerce
// invalid values into the default value, SpanKindInternal.
func ValidateSpanKind(spanKind SpanKind) SpanKind {
	switch spanKind {
	case SpanKindInternal,
		SpanKindServer,
		SpanKindClient,
		SpanKindProducer,
		SpanKindConsumer:
		// valid
		return spanKind
	default:
		return SpanKindInternal
	}
}

// String returns the specified name of the SpanKind in lower-case.
func (sk SpanKind) String() string {
	switch sk {
	case SpanKindInternal:
		return "internal"
	case SpanKindServer:
		return "server"
	case SpanKindClient:
		return "client"
	case SpanKindProducer:
		return "producer"
	case SpanKindConsumer:
		return "consumer"
	default:
		return "unspecified"
	}
}

// Tracer is the creator of Spans.
type Tracer interface {
	// Start creates a span.
	Start(ctx context.Context, spanName string, opts ...SpanOption) (context.Context, Span)
}

// TracerProvider provides access to instrumentation Tracers.
type TracerProvider interface {
	// Tracer creates an implementation of the Tracer interface.
	// The instrumentationName must be the name of the library providing
	// instrumentation. This name may be the same as the instrumented code
	// only if that code provides built-in instrumentation. If the
	// instrumentationName is empty, then a implementation defined default
	// name will be used instead.
	//
	// This method must be concurrency safe.
	Tracer(instrumentationName string, opts ...TracerOption) Tracer
}
