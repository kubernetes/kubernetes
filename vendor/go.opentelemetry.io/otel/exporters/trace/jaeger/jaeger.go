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

package jaeger

import (
	"context"
	"encoding/binary"
	"log"

	"google.golang.org/api/support/bundler"
	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel/api/core"
	"go.opentelemetry.io/otel/api/global"
	gen "go.opentelemetry.io/otel/exporters/trace/jaeger/internal/gen-go/jaeger"
	export "go.opentelemetry.io/otel/sdk/export/trace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

const defaultServiceName = "OpenTelemetry"

type Option func(*options)

// options are the options to be used when initializing a Jaeger export.
type options struct {
	// OnError is the hook to be called when there is
	// an error occurred when uploading the span data.
	// If no custom hook is set, errors are logged.
	OnError func(err error)

	// Process contains the information about the exporting process.
	Process Process

	//BufferMaxCount defines the total number of traces that can be buffered in memory
	BufferMaxCount int

	Config *sdktrace.Config

	// RegisterGlobal is set to true if the trace provider of the new pipeline should be
	// registered as Global Trace Provider
	RegisterGlobal bool
}

// WithOnError sets the hook to be called when there is
// an error occurred when uploading the span data.
// If no custom hook is set, errors are logged.
func WithOnError(onError func(err error)) Option {
	return func(o *options) {
		o.OnError = onError
	}
}

// WithProcess sets the process with the information about the exporting process.
func WithProcess(process Process) Option {
	return func(o *options) {
		o.Process = process
	}
}

//WithBufferMaxCount defines the total number of traces that can be buffered in memory
func WithBufferMaxCount(bufferMaxCount int) Option {
	return func(o *options) {
		o.BufferMaxCount = bufferMaxCount
	}
}

// WithSDK sets the SDK config for the exporter pipeline.
func WithSDK(config *sdktrace.Config) Option {
	return func(o *options) {
		o.Config = config
	}
}

// RegisterAsGlobal enables the registration of the trace provider of the new pipeline
// as Global Trace Provider.
func RegisterAsGlobal() Option {
	return func(o *options) {
		o.RegisterGlobal = true
	}
}

// NewRawExporter returns a trace.Exporter implementation that exports
// the collected spans to Jaeger.
func NewRawExporter(endpointOption EndpointOption, opts ...Option) (*Exporter, error) {
	uploader, err := endpointOption()
	if err != nil {
		return nil, err
	}

	o := options{}
	for _, opt := range opts {
		opt(&o)
	}

	onError := func(err error) {
		if o.OnError != nil {
			o.OnError(err)
			return
		}
		log.Printf("Error when uploading spans to Jaeger: %v", err)
	}
	service := o.Process.ServiceName
	if service == "" {
		service = defaultServiceName
	}
	tags := make([]*gen.Tag, 0, len(o.Process.Tags))
	for _, tag := range o.Process.Tags {
		t := keyValueToTag(tag)
		if t != nil {
			tags = append(tags, t)
		}
	}
	e := &Exporter{
		uploader: uploader,
		process: &gen.Process{
			ServiceName: service,
			Tags:        tags,
		},
		o: o,
	}
	bundler := bundler.NewBundler((*gen.Span)(nil), func(bundle interface{}) {
		if err := e.upload(bundle.([]*gen.Span)); err != nil {
			onError(err)
		}
	})

	// Set BufferedByteLimit with the total number of spans that are permissible to be held in memory.
	// This needs to be done since the size of messages is always set to 1. Failing to set this would allow
	// 1G messages to be held in memory since that is the default value of BufferedByteLimit.
	if o.BufferMaxCount != 0 {
		bundler.BufferedByteLimit = o.BufferMaxCount
	}

	e.bundler = bundler
	return e, nil
}

// NewExportPipeline sets up a complete export pipeline
// with the recommended setup for trace provider
func NewExportPipeline(endpointOption EndpointOption, opts ...Option) (*sdktrace.Provider, func(), error) {
	exporter, err := NewRawExporter(endpointOption, opts...)
	if err != nil {
		return nil, nil, err
	}
	syncer := sdktrace.WithSyncer(exporter)
	tp, err := sdktrace.NewProvider(syncer)
	if err != nil {
		return nil, nil, err
	}
	if exporter.o.Config != nil {
		tp.ApplyConfig(*exporter.o.Config)
	}
	if exporter.o.RegisterGlobal {
		global.SetTraceProvider(tp)
	}

	return tp, exporter.Flush, nil
}

// Process contains the information exported to jaeger about the source
// of the trace data.
type Process struct {
	// ServiceName is the Jaeger service name.
	ServiceName string

	// Tags are added to Jaeger Process exports
	Tags []core.KeyValue
}

// Exporter is an implementation of trace.SpanSyncer that uploads spans to Jaeger.
type Exporter struct {
	process  *gen.Process
	bundler  *bundler.Bundler
	uploader batchUploader
	o        options
}

var _ export.SpanSyncer = (*Exporter)(nil)

// ExportSpan exports a SpanData to Jaeger.
func (e *Exporter) ExportSpan(ctx context.Context, d *export.SpanData) {
	_ = e.bundler.Add(spanDataToThrift(d), 1)
	// TODO(jbd): Handle oversized bundlers.
}

func spanDataToThrift(data *export.SpanData) *gen.Span {
	tags := make([]*gen.Tag, 0, len(data.Attributes))
	for _, kv := range data.Attributes {
		tag := keyValueToTag(kv)
		if tag != nil {
			tags = append(tags, tag)
		}
	}

	// TODO (jmacd): OTel has a broad "last value wins"
	// semantic. Should resources be appended before span
	// attributes, above, to allow span attributes to
	// overwrite resource attributes?
	if data.Resource != nil {
		for iter := data.Resource.Iter(); iter.Next(); {
			if tag := keyValueToTag(iter.Attribute()); tag != nil {
				tags = append(tags, tag)
			}
		}
	}

	tags = append(tags,
		getInt64Tag("status.code", int64(data.StatusCode)),
		getStringTag("status.message", data.StatusMessage),
		getStringTag("span.kind", data.SpanKind.String()),
	)

	// Ensure that if Status.Code is not OK, that we set the "error" tag on the Jaeger span.
	// See Issue https://github.com/census-instrumentation/opencensus-go/issues/1041
	if data.StatusCode != codes.OK {
		tags = append(tags, getBoolTag("error", true))
	}

	var logs []*gen.Log
	for _, a := range data.MessageEvents {
		fields := make([]*gen.Tag, 0, len(a.Attributes))
		for _, kv := range a.Attributes {
			tag := keyValueToTag(kv)
			if tag != nil {
				fields = append(fields, tag)
			}
		}
		fields = append(fields, getStringTag("name", a.Name))
		logs = append(logs, &gen.Log{
			Timestamp: a.Time.UnixNano() / 1000,
			Fields:    fields,
		})
	}

	var refs []*gen.SpanRef
	for _, link := range data.Links {
		refs = append(refs, &gen.SpanRef{
			TraceIdHigh: int64(binary.BigEndian.Uint64(link.TraceID[0:8])),
			TraceIdLow:  int64(binary.BigEndian.Uint64(link.TraceID[8:16])),
			SpanId:      int64(binary.BigEndian.Uint64(link.SpanID[:])),
			// TODO(paivagustavo): properly set the reference type when specs are defined
			//  see https://github.com/open-telemetry/opentelemetry-specification/issues/65
			RefType: gen.SpanRefType_CHILD_OF,
		})
	}

	return &gen.Span{
		TraceIdHigh:   int64(binary.BigEndian.Uint64(data.SpanContext.TraceID[0:8])),
		TraceIdLow:    int64(binary.BigEndian.Uint64(data.SpanContext.TraceID[8:16])),
		SpanId:        int64(binary.BigEndian.Uint64(data.SpanContext.SpanID[:])),
		ParentSpanId:  int64(binary.BigEndian.Uint64(data.ParentSpanID[:])),
		OperationName: data.Name, // TODO: if span kind is added then add prefix "Sent"/"Recv"
		Flags:         int32(data.SpanContext.TraceFlags),
		StartTime:     data.StartTime.UnixNano() / 1000,
		Duration:      data.EndTime.Sub(data.StartTime).Nanoseconds() / 1000,
		Tags:          tags,
		Logs:          logs,
		References:    refs,
	}
}

func keyValueToTag(kv core.KeyValue) *gen.Tag {
	var tag *gen.Tag
	switch kv.Value.Type() {
	case core.STRING:
		s := kv.Value.AsString()
		tag = &gen.Tag{
			Key:   string(kv.Key),
			VStr:  &s,
			VType: gen.TagType_STRING,
		}
	case core.BOOL:
		b := kv.Value.AsBool()
		tag = &gen.Tag{
			Key:   string(kv.Key),
			VBool: &b,
			VType: gen.TagType_BOOL,
		}
	case core.INT32:
		i := int64(kv.Value.AsInt32())
		tag = &gen.Tag{
			Key:   string(kv.Key),
			VLong: &i,
			VType: gen.TagType_LONG,
		}
	case core.INT64:
		i := kv.Value.AsInt64()
		tag = &gen.Tag{
			Key:   string(kv.Key),
			VLong: &i,
			VType: gen.TagType_LONG,
		}
	case core.FLOAT32:
		f := float64(kv.Value.AsFloat32())
		tag = &gen.Tag{
			Key:     string(kv.Key),
			VDouble: &f,
			VType:   gen.TagType_DOUBLE,
		}
	case core.FLOAT64:
		f := kv.Value.AsFloat64()
		tag = &gen.Tag{
			Key:     string(kv.Key),
			VDouble: &f,
			VType:   gen.TagType_DOUBLE,
		}
	}
	return tag
}

func getInt64Tag(k string, i int64) *gen.Tag {
	return &gen.Tag{
		Key:   k,
		VLong: &i,
		VType: gen.TagType_LONG,
	}
}

func getStringTag(k, s string) *gen.Tag {
	return &gen.Tag{
		Key:   k,
		VStr:  &s,
		VType: gen.TagType_STRING,
	}
}

func getBoolTag(k string, b bool) *gen.Tag {
	return &gen.Tag{
		Key:   k,
		VBool: &b,
		VType: gen.TagType_BOOL,
	}
}

// Flush waits for exported trace spans to be uploaded.
//
// This is useful if your program is ending and you do not want to lose recent spans.
func (e *Exporter) Flush() {
	e.bundler.Flush()
}

func (e *Exporter) upload(spans []*gen.Span) error {
	batch := &gen.Batch{
		Spans:   spans,
		Process: e.process,
	}

	return e.uploader.upload(batch)
}
