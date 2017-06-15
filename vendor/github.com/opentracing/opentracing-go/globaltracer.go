package opentracing

var (
	globalTracer Tracer = NoopTracer{}
)

// SetGlobalTracer sets the [singleton] opentracing.Tracer returned by
// GlobalTracer(). Those who use GlobalTracer (rather than directly manage an
// opentracing.Tracer instance) should call SetGlobalTracer as early as
// possible in main(), prior to calling the `StartSpan` global func below.
// Prior to calling `SetGlobalTracer`, any Spans started via the `StartSpan`
// (etc) globals are noops.
func SetGlobalTracer(tracer Tracer) {
	globalTracer = tracer
}

// GlobalTracer returns the global singleton `Tracer` implementation.
// Before `SetGlobalTracer()` is called, the `GlobalTracer()` is a noop
// implementation that drops all data handed to it.
func GlobalTracer() Tracer {
	return globalTracer
}

// StartSpan defers to `Tracer.StartSpan`. See `GlobalTracer()`.
func StartSpan(operationName string, opts ...StartSpanOption) Span {
	return globalTracer.StartSpan(operationName, opts...)
}

// InitGlobalTracer is deprecated. Please use SetGlobalTracer.
func InitGlobalTracer(tracer Tracer) {
	SetGlobalTracer(tracer)
}
