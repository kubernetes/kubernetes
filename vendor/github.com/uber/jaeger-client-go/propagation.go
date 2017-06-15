// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package jaeger

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net/url"
	"strings"
	"sync"

	opentracing "github.com/opentracing/opentracing-go"
)

// Injector is responsible for injecting SpanContext instances in a manner suitable
// for propagation via a format-specific "carrier" object. Typically the
// injection will take place across an RPC boundary, but message queues and
// other IPC mechanisms are also reasonable places to use an Injector.
type Injector interface {
	// Inject takes `SpanContext` and injects it into `carrier`. The actual type
	// of `carrier` depends on the `format` passed to `Tracer.Inject()`.
	//
	// Implementations may return opentracing.ErrInvalidCarrier or any other
	// implementation-specific error if injection fails.
	Inject(ctx SpanContext, carrier interface{}) error
}

// Extractor is responsible for extracting SpanContext instances from a
// format-specific "carrier" object. Typically the extraction will take place
// on the server side of an RPC boundary, but message queues and other IPC
// mechanisms are also reasonable places to use an Extractor.
type Extractor interface {
	// Extract decodes a SpanContext instance from the given `carrier`,
	// or (nil, opentracing.ErrSpanContextNotFound) if no context could
	// be found in the `carrier`.
	Extract(carrier interface{}) (SpanContext, error)
}

type textMapPropagator struct {
	tracer      *tracer
	encodeValue func(string) string
	decodeValue func(string) string
}

func newTextMapPropagator(tracer *tracer) *textMapPropagator {
	return &textMapPropagator{
		tracer: tracer,
		encodeValue: func(val string) string {
			return val
		},
		decodeValue: func(val string) string {
			return val
		},
	}
}

func newHTTPHeaderPropagator(tracer *tracer) *textMapPropagator {
	return &textMapPropagator{
		tracer: tracer,
		encodeValue: func(val string) string {
			return url.QueryEscape(val)
		},
		decodeValue: func(val string) string {
			// ignore decoding errors, cannot do anything about them
			if v, err := url.QueryUnescape(val); err == nil {
				return v
			}
			return val
		},
	}
}

type binaryPropagator struct {
	tracer  *tracer
	buffers sync.Pool
}

func newBinaryPropagator(tracer *tracer) *binaryPropagator {
	return &binaryPropagator{
		tracer:  tracer,
		buffers: sync.Pool{New: func() interface{} { return &bytes.Buffer{} }},
	}
}

func (p *textMapPropagator) Inject(
	sc SpanContext,
	abstractCarrier interface{},
) error {
	textMapWriter, ok := abstractCarrier.(opentracing.TextMapWriter)
	if !ok {
		return opentracing.ErrInvalidCarrier
	}

	// Do not encode the string with trace context to avoid accidental double-encoding
	// if people are using opentracing < 0.10.0. Our colon-separated representation
	// of the trace context is already safe for HTTP headers.
	textMapWriter.Set(TracerStateHeaderName, sc.String())
	for k, v := range sc.baggage {
		safeKey := addBaggageKeyPrefix(k)
		safeVal := p.encodeValue(v)
		textMapWriter.Set(safeKey, safeVal)
	}
	return nil
}

func (p *textMapPropagator) Extract(abstractCarrier interface{}) (SpanContext, error) {
	textMapReader, ok := abstractCarrier.(opentracing.TextMapReader)
	if !ok {
		return emptyContext, opentracing.ErrInvalidCarrier
	}
	var ctx SpanContext
	var baggage map[string]string
	err := textMapReader.ForeachKey(func(rawKey, value string) error {
		key := strings.ToLower(rawKey) // TODO not necessary for plain TextMap
		if key == TracerStateHeaderName {
			var err error
			safeVal := p.decodeValue(value)
			if ctx, err = ContextFromString(safeVal); err != nil {
				return err
			}
		} else if key == JaegerDebugHeader {
			ctx.debugID = p.decodeValue(value)
		} else if key == JaegerBaggageHeader {
			if baggage == nil {
				baggage = make(map[string]string)
			}
			for k, v := range parseCommaSeparatedMap(value) {
				baggage[k] = v
			}
		} else if strings.HasPrefix(key, TraceBaggageHeaderPrefix) {
			if baggage == nil {
				baggage = make(map[string]string)
			}
			safeKey := removeBaggageKeyPrefix(key)
			safeVal := p.decodeValue(value)
			baggage[safeKey] = safeVal
		}
		return nil
	})
	if err != nil {
		p.tracer.metrics.DecodingErrors.Inc(1)
		return emptyContext, err
	}
	if !ctx.traceID.IsValid() && ctx.debugID == "" && len(baggage) == 0 {
		return emptyContext, opentracing.ErrSpanContextNotFound
	}
	ctx.baggage = baggage
	return ctx, nil
}

func (p *binaryPropagator) Inject(
	sc SpanContext,
	abstractCarrier interface{},
) error {
	carrier, ok := abstractCarrier.(io.Writer)
	if !ok {
		return opentracing.ErrInvalidCarrier
	}

	// Handle the tracer context
	if err := binary.Write(carrier, binary.BigEndian, sc.traceID); err != nil {
		return err
	}
	if err := binary.Write(carrier, binary.BigEndian, sc.spanID); err != nil {
		return err
	}
	if err := binary.Write(carrier, binary.BigEndian, sc.parentID); err != nil {
		return err
	}
	if err := binary.Write(carrier, binary.BigEndian, sc.flags); err != nil {
		return err
	}

	// Handle the baggage items
	if err := binary.Write(carrier, binary.BigEndian, int32(len(sc.baggage))); err != nil {
		return err
	}
	for k, v := range sc.baggage {
		if err := binary.Write(carrier, binary.BigEndian, int32(len(k))); err != nil {
			return err
		}
		io.WriteString(carrier, k)
		if err := binary.Write(carrier, binary.BigEndian, int32(len(v))); err != nil {
			return err
		}
		io.WriteString(carrier, v)
	}

	return nil
}

func (p *binaryPropagator) Extract(abstractCarrier interface{}) (SpanContext, error) {
	carrier, ok := abstractCarrier.(io.Reader)
	if !ok {
		return emptyContext, opentracing.ErrInvalidCarrier
	}
	var ctx SpanContext

	if err := binary.Read(carrier, binary.BigEndian, &ctx.traceID); err != nil {
		return emptyContext, opentracing.ErrSpanContextCorrupted
	}
	if err := binary.Read(carrier, binary.BigEndian, &ctx.spanID); err != nil {
		return emptyContext, opentracing.ErrSpanContextCorrupted
	}
	if err := binary.Read(carrier, binary.BigEndian, &ctx.parentID); err != nil {
		return emptyContext, opentracing.ErrSpanContextCorrupted
	}
	if err := binary.Read(carrier, binary.BigEndian, &ctx.flags); err != nil {
		return emptyContext, opentracing.ErrSpanContextCorrupted
	}

	// Handle the baggage items
	var numBaggage int32
	if err := binary.Read(carrier, binary.BigEndian, &numBaggage); err != nil {
		return emptyContext, opentracing.ErrSpanContextCorrupted
	}
	if iNumBaggage := int(numBaggage); iNumBaggage > 0 {
		ctx.baggage = make(map[string]string, iNumBaggage)
		buf := p.buffers.Get().(*bytes.Buffer)
		defer p.buffers.Put(buf)

		var keyLen, valLen int32
		for i := 0; i < iNumBaggage; i++ {
			if err := binary.Read(carrier, binary.BigEndian, &keyLen); err != nil {
				return emptyContext, opentracing.ErrSpanContextCorrupted
			}
			buf.Reset()
			buf.Grow(int(keyLen))
			if n, err := io.CopyN(buf, carrier, int64(keyLen)); err != nil || int32(n) != keyLen {
				return emptyContext, opentracing.ErrSpanContextCorrupted
			}
			key := buf.String()

			if err := binary.Read(carrier, binary.BigEndian, &valLen); err != nil {
				return emptyContext, opentracing.ErrSpanContextCorrupted
			}
			buf.Reset()
			buf.Grow(int(valLen))
			if n, err := io.CopyN(buf, carrier, int64(valLen)); err != nil || int32(n) != valLen {
				return emptyContext, opentracing.ErrSpanContextCorrupted
			}
			ctx.baggage[key] = buf.String()
		}
	}

	return ctx, nil
}

// Converts a comma separated key value pair list into a map
// e.g. key1=value1, key2=value2, key3 = value3
// is converted to map[string]string { "key1" : "value1",
//                                     "key2" : "value2",
//                                     "key3" : "value3" }
func parseCommaSeparatedMap(value string) map[string]string {
	baggage := make(map[string]string)
	value, err := url.QueryUnescape(value)
	if err != nil {
		log.Printf("Unable to unescape %s, %v", value, err)
		return baggage
	}
	for _, kvpair := range strings.Split(value, ",") {
		kv := strings.Split(strings.TrimSpace(kvpair), "=")
		if len(kv) == 2 {
			baggage[kv[0]] = kv[1]
		} else {
			log.Printf("Malformed value passed in for %s", JaegerBaggageHeader)
		}
	}
	return baggage
}

// Converts a baggage item key into an http header format,
// by prepending TraceBaggageHeaderPrefix and encoding the key string
func addBaggageKeyPrefix(key string) string {
	// TODO encodeBaggageKeyAsHeader add caching and escaping
	return fmt.Sprintf("%v%v", TraceBaggageHeaderPrefix, key)
}

func removeBaggageKeyPrefix(key string) string {
	// TODO decodeBaggageHeaderKey add caching and escaping
	return key[len(TraceBaggageHeaderPrefix):]
}
