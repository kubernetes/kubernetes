package opentracing

import "time"

// Tracer is a simple, thin interface for Span creation and SpanContext
// propagation.
type Tracer interface {

	// Create, start, and return a new Span with the given `operationName` and
	// incorporate the given StartSpanOption `opts`. (Note that `opts` borrows
	// from the "functional options" pattern, per
	// http://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis)
	//
	// A Span with no SpanReference options (e.g., opentracing.ChildOf() or
	// opentracing.FollowsFrom()) becomes the root of its own trace.
	//
	// Examples:
	//
	//     var tracer opentracing.Tracer = ...
	//
	//     // The root-span case:
	//     sp := tracer.StartSpan("GetFeed")
	//
	//     // The vanilla child span case:
	//     sp := tracer.StartSpan(
	//         "GetFeed",
	//         opentracing.ChildOf(parentSpan.Context()))
	//
	//     // All the bells and whistles:
	//     sp := tracer.StartSpan(
	//         "GetFeed",
	//         opentracing.ChildOf(parentSpan.Context()),
	//         opentracing.Tag{"user_agent", loggedReq.UserAgent},
	//         opentracing.StartTime(loggedReq.Timestamp),
	//     )
	//
	StartSpan(operationName string, opts ...StartSpanOption) Span

	// Inject() takes the `sm` SpanContext instance and injects it for
	// propagation within `carrier`. The actual type of `carrier` depends on
	// the value of `format`.
	//
	// OpenTracing defines a common set of `format` values (see BuiltinFormat),
	// and each has an expected carrier type.
	//
	// Other packages may declare their own `format` values, much like the keys
	// used by `context.Context` (see
	// https://godoc.org/golang.org/x/net/context#WithValue).
	//
	// Example usage (sans error handling):
	//
	//     carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
	//     err := tracer.Inject(
	//         span.Context(),
	//         opentracing.HTTPHeaders,
	//         carrier)
	//
	// NOTE: All opentracing.Tracer implementations MUST support all
	// BuiltinFormats.
	//
	// Implementations may return opentracing.ErrUnsupportedFormat if `format`
	// is not supported by (or not known by) the implementation.
	//
	// Implementations may return opentracing.ErrInvalidCarrier or any other
	// implementation-specific error if the format is supported but injection
	// fails anyway.
	//
	// See Tracer.Extract().
	Inject(sm SpanContext, format interface{}, carrier interface{}) error

	// Extract() returns a SpanContext instance given `format` and `carrier`.
	//
	// OpenTracing defines a common set of `format` values (see BuiltinFormat),
	// and each has an expected carrier type.
	//
	// Other packages may declare their own `format` values, much like the keys
	// used by `context.Context` (see
	// https://godoc.org/golang.org/x/net/context#WithValue).
	//
	// Example usage (with StartSpan):
	//
	//
	//     carrier := opentracing.HTTPHeadersCarrier(httpReq.Header)
	//     clientContext, err := tracer.Extract(opentracing.HTTPHeaders, carrier)
	//
	//     // ... assuming the ultimate goal here is to resume the trace with a
	//     // server-side Span:
	//     var serverSpan opentracing.Span
	//     if err == nil {
	//         span = tracer.StartSpan(
	//             rpcMethodName, ext.RPCServerOption(clientContext))
	//     } else {
	//         span = tracer.StartSpan(rpcMethodName)
	//     }
	//
	//
	// NOTE: All opentracing.Tracer implementations MUST support all
	// BuiltinFormats.
	//
	// Return values:
	//  - A successful Extract returns a SpanContext instance and a nil error
	//  - If there was simply no SpanContext to extract in `carrier`, Extract()
	//    returns (nil, opentracing.ErrSpanContextNotFound)
	//  - If `format` is unsupported or unrecognized, Extract() returns (nil,
	//    opentracing.ErrUnsupportedFormat)
	//  - If there are more fundamental problems with the `carrier` object,
	//    Extract() may return opentracing.ErrInvalidCarrier,
	//    opentracing.ErrSpanContextCorrupted, or implementation-specific
	//    errors.
	//
	// See Tracer.Inject().
	Extract(format interface{}, carrier interface{}) (SpanContext, error)
}

// StartSpanOptions allows Tracer.StartSpan() callers and implementors a
// mechanism to override the start timestamp, specify Span References, and make
// a single Tag or multiple Tags available at Span start time.
//
// StartSpan() callers should look at the StartSpanOption interface and
// implementations available in this package.
//
// Tracer implementations can convert a slice of `StartSpanOption` instances
// into a `StartSpanOptions` struct like so:
//
//     func StartSpan(opName string, opts ...opentracing.StartSpanOption) {
//         sso := opentracing.StartSpanOptions{}
//         for _, o := range opts {
//             o.Apply(&sso)
//         }
//         ...
//     }
//
type StartSpanOptions struct {
	// Zero or more causal references to other Spans (via their SpanContext).
	// If empty, start a "root" Span (i.e., start a new trace).
	References []SpanReference

	// StartTime overrides the Span's start time, or implicitly becomes
	// time.Now() if StartTime.IsZero().
	StartTime time.Time

	// Tags may have zero or more entries; the restrictions on map values are
	// identical to those for Span.SetTag(). May be nil.
	//
	// If specified, the caller hands off ownership of Tags at
	// StartSpan() invocation time.
	Tags map[string]interface{}
}

// StartSpanOption instances (zero or more) may be passed to Tracer.StartSpan.
//
// StartSpanOption borrows from the "functional options" pattern, per
// http://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis
type StartSpanOption interface {
	Apply(*StartSpanOptions)
}

// SpanReferenceType is an enum type describing different categories of
// relationships between two Spans. If Span-2 refers to Span-1, the
// SpanReferenceType describes Span-1 from Span-2's perspective. For example,
// ChildOfRef means that Span-1 created Span-2.
//
// NOTE: Span-1 and Span-2 do *not* necessarily depend on each other for
// completion; e.g., Span-2 may be part of a background job enqueued by Span-1,
// or Span-2 may be sitting in a distributed queue behind Span-1.
type SpanReferenceType int

const (
	// ChildOfRef refers to a parent Span that caused *and* somehow depends
	// upon the new child Span. Often (but not always), the parent Span cannot
	// finish until the child Span does.
	//
	// An timing diagram for a ChildOfRef that's blocked on the new Span:
	//
	//     [-Parent Span---------]
	//          [-Child Span----]
	//
	// See http://opentracing.io/spec/
	//
	// See opentracing.ChildOf()
	ChildOfRef SpanReferenceType = iota

	// FollowsFromRef refers to a parent Span that does not depend in any way
	// on the result of the new child Span. For instance, one might use
	// FollowsFromRefs to describe pipeline stages separated by queues,
	// or a fire-and-forget cache insert at the tail end of a web request.
	//
	// A FollowsFromRef Span is part of the same logical trace as the new Span:
	// i.e., the new Span is somehow caused by the work of its FollowsFromRef.
	//
	// All of the following could be valid timing diagrams for children that
	// "FollowFrom" a parent.
	//
	//     [-Parent Span-]  [-Child Span-]
	//
	//
	//     [-Parent Span--]
	//      [-Child Span-]
	//
	//
	//     [-Parent Span-]
	//                 [-Child Span-]
	//
	// See http://opentracing.io/spec/
	//
	// See opentracing.FollowsFrom()
	FollowsFromRef
)

// SpanReference is a StartSpanOption that pairs a SpanReferenceType and a
// referenced SpanContext. See the SpanReferenceType documentation for
// supported relationships.  If SpanReference is created with
// ReferencedContext==nil, it has no effect. Thus it allows for a more concise
// syntax for starting spans:
//
//     sc, _ := tracer.Extract(someFormat, someCarrier)
//     span := tracer.StartSpan("operation", opentracing.ChildOf(sc))
//
// The `ChildOf(sc)` option above will not panic if sc == nil, it will just
// not add the parent span reference to the options.
type SpanReference struct {
	Type              SpanReferenceType
	ReferencedContext SpanContext
}

// Apply satisfies the StartSpanOption interface.
func (r SpanReference) Apply(o *StartSpanOptions) {
	if r.ReferencedContext != nil {
		o.References = append(o.References, r)
	}
}

// ChildOf returns a StartSpanOption pointing to a dependent parent span.
// If sc == nil, the option has no effect.
//
// See ChildOfRef, SpanReference
func ChildOf(sc SpanContext) SpanReference {
	return SpanReference{
		Type:              ChildOfRef,
		ReferencedContext: sc,
	}
}

// FollowsFrom returns a StartSpanOption pointing to a parent Span that caused
// the child Span but does not directly depend on its result in any way.
// If sc == nil, the option has no effect.
//
// See FollowsFromRef, SpanReference
func FollowsFrom(sc SpanContext) SpanReference {
	return SpanReference{
		Type:              FollowsFromRef,
		ReferencedContext: sc,
	}
}

// StartTime is a StartSpanOption that sets an explicit start timestamp for the
// new Span.
type StartTime time.Time

// Apply satisfies the StartSpanOption interface.
func (t StartTime) Apply(o *StartSpanOptions) {
	o.StartTime = time.Time(t)
}

// Tags are a generic map from an arbitrary string key to an opaque value type.
// The underlying tracing system is responsible for interpreting and
// serializing the values.
type Tags map[string]interface{}

// Apply satisfies the StartSpanOption interface.
func (t Tags) Apply(o *StartSpanOptions) {
	if o.Tags == nil {
		o.Tags = make(map[string]interface{})
	}
	for k, v := range t {
		o.Tags[k] = v
	}
}

// Tag may be passed as a StartSpanOption to add a tag to new spans,
// or its Set method may be used to apply the tag to an existing Span,
// for example:
//
// tracer.StartSpan("opName", Tag{"Key", value})
//
//   or
//
// Tag{"key", value}.Set(span)
type Tag struct {
	Key   string
	Value interface{}
}

// Apply satisfies the StartSpanOption interface.
func (t Tag) Apply(o *StartSpanOptions) {
	if o.Tags == nil {
		o.Tags = make(map[string]interface{})
	}
	o.Tags[t.Key] = t.Value
}

// Set applies the tag to an existing Span.
func (t Tag) Set(s Span) {
	s.SetTag(t.Key, t.Value)
}
