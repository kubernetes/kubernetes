package opentracing

import (
	"time"

	"github.com/opentracing/opentracing-go/log"
)

// SpanContext represents Span state that must propagate to descendant Spans and across process
// boundaries (e.g., a <trace_id, span_id, sampled> tuple).
type SpanContext interface {
	// ForeachBaggageItem grants access to all baggage items stored in the
	// SpanContext.
	// The handler function will be called for each baggage key/value pair.
	// The ordering of items is not guaranteed.
	//
	// The bool return value indicates if the handler wants to continue iterating
	// through the rest of the baggage items; for example if the handler is trying to
	// find some baggage item by pattern matching the name, it can return false
	// as soon as the item is found to stop further iterations.
	ForeachBaggageItem(handler func(k, v string) bool)
}

// Span represents an active, un-finished span in the OpenTracing system.
//
// Spans are created by the Tracer interface.
type Span interface {
	// Sets the end timestamp and finalizes Span state.
	//
	// With the exception of calls to Context() (which are always allowed),
	// Finish() must be the last call made to any span instance, and to do
	// otherwise leads to undefined behavior.
	Finish()
	// FinishWithOptions is like Finish() but with explicit control over
	// timestamps and log data.
	FinishWithOptions(opts FinishOptions)

	// Context() yields the SpanContext for this Span. Note that the return
	// value of Context() is still valid after a call to Span.Finish(), as is
	// a call to Span.Context() after a call to Span.Finish().
	Context() SpanContext

	// Sets or changes the operation name.
	SetOperationName(operationName string) Span

	// Adds a tag to the span.
	//
	// If there is a pre-existing tag set for `key`, it is overwritten.
	//
	// Tag values can be numeric types, strings, or bools. The behavior of
	// other tag value types is undefined at the OpenTracing level. If a
	// tracing system does not know how to handle a particular value type, it
	// may ignore the tag, but shall not panic.
	SetTag(key string, value interface{}) Span

	// LogFields is an efficient and type-checked way to record key:value
	// logging data about a Span, though the programming interface is a little
	// more verbose than LogKV(). Here's an example:
	//
	//    span.LogFields(
	//        log.String("event", "soft error"),
	//        log.String("type", "cache timeout"),
	//        log.Int("waited.millis", 1500))
	//
	// Also see Span.FinishWithOptions() and FinishOptions.BulkLogData.
	LogFields(fields ...log.Field)

	// LogKV is a concise, readable way to record key:value logging data about
	// a Span, though unfortunately this also makes it less efficient and less
	// type-safe than LogFields(). Here's an example:
	//
	//    span.LogKV(
	//        "event", "soft error",
	//        "type", "cache timeout",
	//        "waited.millis", 1500)
	//
	// For LogKV (as opposed to LogFields()), the parameters must appear as
	// key-value pairs, like
	//
	//    span.LogKV(key1, val1, key2, val2, key3, val3, ...)
	//
	// The keys must all be strings. The values may be strings, numeric types,
	// bools, Go error instances, or arbitrary structs.
	//
	// (Note to implementors: consider the log.InterleavedKVToFields() helper)
	LogKV(alternatingKeyValues ...interface{})

	// SetBaggageItem sets a key:value pair on this Span and its SpanContext
	// that also propagates to descendants of this Span.
	//
	// SetBaggageItem() enables powerful functionality given a full-stack
	// opentracing integration (e.g., arbitrary application data from a mobile
	// app can make it, transparently, all the way into the depths of a storage
	// system), and with it some powerful costs: use this feature with care.
	//
	// IMPORTANT NOTE #1: SetBaggageItem() will only propagate baggage items to
	// *future* causal descendants of the associated Span.
	//
	// IMPORTANT NOTE #2: Use this thoughtfully and with care. Every key and
	// value is copied into every local *and remote* child of the associated
	// Span, and that can add up to a lot of network and cpu overhead.
	//
	// Returns a reference to this Span for chaining.
	SetBaggageItem(restrictedKey, value string) Span

	// Gets the value for a baggage item given its key. Returns the empty string
	// if the value isn't found in this Span.
	BaggageItem(restrictedKey string) string

	// Provides access to the Tracer that created this Span.
	Tracer() Tracer

	// Deprecated: use LogFields or LogKV
	LogEvent(event string)
	// Deprecated: use LogFields or LogKV
	LogEventWithPayload(event string, payload interface{})
	// Deprecated: use LogFields or LogKV
	Log(data LogData)
}

// LogRecord is data associated with a single Span log. Every LogRecord
// instance must specify at least one Field.
type LogRecord struct {
	Timestamp time.Time
	Fields    []log.Field
}

// FinishOptions allows Span.FinishWithOptions callers to override the finish
// timestamp and provide log data via a bulk interface.
type FinishOptions struct {
	// FinishTime overrides the Span's finish time, or implicitly becomes
	// time.Now() if FinishTime.IsZero().
	//
	// FinishTime must resolve to a timestamp that's >= the Span's StartTime
	// (per StartSpanOptions).
	FinishTime time.Time

	// LogRecords allows the caller to specify the contents of many LogFields()
	// calls with a single slice. May be nil.
	//
	// None of the LogRecord.Timestamp values may be .IsZero() (i.e., they must
	// be set explicitly). Also, they must be >= the Span's start timestamp and
	// <= the FinishTime (or time.Now() if FinishTime.IsZero()). Otherwise the
	// behavior of FinishWithOptions() is undefined.
	//
	// If specified, the caller hands off ownership of LogRecords at
	// FinishWithOptions() invocation time.
	//
	// If specified, the (deprecated) BulkLogData must be nil or empty.
	LogRecords []LogRecord

	// BulkLogData is DEPRECATED.
	BulkLogData []LogData
}

// LogData is DEPRECATED
type LogData struct {
	Timestamp time.Time
	Event     string
	Payload   interface{}
}

// ToLogRecord converts a deprecated LogData to a non-deprecated LogRecord
func (ld *LogData) ToLogRecord() LogRecord {
	var literalTimestamp time.Time
	if ld.Timestamp.IsZero() {
		literalTimestamp = time.Now()
	} else {
		literalTimestamp = ld.Timestamp
	}
	rval := LogRecord{
		Timestamp: literalTimestamp,
	}
	if ld.Payload == nil {
		rval.Fields = []log.Field{
			log.String("event", ld.Event),
		}
	} else {
		rval.Fields = []log.Field{
			log.String("event", ld.Event),
			log.Object("payload", ld.Payload),
		}
	}
	return rval
}
