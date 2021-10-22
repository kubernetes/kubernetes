// +build codegen

package api

import (
	"fmt"
	"io"
	"strings"
	"text/template"
)

func renderEventStreamAPI(w io.Writer, op *Operation) error {
	// Imports needed by the EventStream APIs.
	op.API.AddImport("fmt")
	op.API.AddImport("bytes")
	op.API.AddImport("io")
	op.API.AddImport("time")
	op.API.AddSDKImport("aws")
	op.API.AddSDKImport("aws/awserr")
	op.API.AddSDKImport("aws/request")
	op.API.AddSDKImport("private/protocol/eventstream")
	op.API.AddSDKImport("private/protocol/eventstream/eventstreamapi")

	w.Write([]byte(`
var _ awserr.Error
`))

	return eventStreamAPITmpl.Execute(w, op)
}

// Template for an EventStream API Shape that will provide read/writing events
// across the EventStream. This is a special shape that's only public members
// are the Events channel and a Close and Err method.
//
// Executed in the context of a Shape.
var eventStreamAPITmpl = template.Must(
	template.New("eventStreamAPITmplDef").
		Funcs(template.FuncMap{
			"unexported": func(v string) string {
				return strings.ToLower(string(v[0])) + v[1:]
			},
		}).
		Parse(eventStreamAPITmplDef),
)

const eventStreamAPITmplDef = `
{{- $esapi := $.EventStreamAPI }}
{{- $outputStream := $esapi.OutputStream }}
{{- $inputStream := $esapi.InputStream }}

// {{ $esapi.Name }} provides the event stream handling for the {{ $.ExportedName }}.
//
// For testing and mocking the event stream this type should be initialized via
// the New{{ $esapi.Name }} constructor function. Using the functional options
// to pass in nested mock behavior.
type {{ $esapi.Name }} struct {
	{{- if $inputStream }}

		// Writer is the EventStream writer for the {{ $inputStream.Name }}
		// events. This value is automatically set by the SDK when the API call is made
		// Use this member when unit testing your code with the SDK to mock out the
		// EventStream Writer.
		//
		// Must not be nil.
		Writer {{ $inputStream.StreamWriterAPIName }}

		inputWriter io.WriteCloser
		{{- if eq .API.Metadata.Protocol "json" }}
			input {{ $.InputRef.GoType }}
		{{- end }}
	{{- end }}

	{{- if $outputStream }}

		// Reader is the EventStream reader for the {{ $outputStream.Name }}
		// events. This value is automatically set by the SDK when the API call is made
		// Use this member when unit testing your code with the SDK to mock out the
		// EventStream Reader.
		//
		// Must not be nil.
		Reader {{ $outputStream.StreamReaderAPIName }}

		outputReader io.ReadCloser
		{{- if eq .API.Metadata.Protocol "json" }}
			output {{ $.OutputRef.GoType }}
		{{- end }}
	{{- end }}

	{{- if $esapi.Legacy }}

		// StreamCloser is the io.Closer for the EventStream connection. For HTTP
		// EventStream this is the response Body. The stream will be closed when
		// the Close method of the EventStream is called.
		StreamCloser io.Closer
	{{- end }}

	done      chan struct{}
	closeOnce sync.Once
	err       *eventstreamapi.OnceError
}

// New{{ $esapi.Name }} initializes an {{ $esapi.Name }}.
// This function should only be used for testing and mocking the {{ $esapi.Name }}
// stream within your application.
{{- if $inputStream }}
//
// The Writer member must be set before writing events to the stream.
{{- end }}
{{- if $outputStream }}
//
// The Reader member must be set before reading events from the stream.
{{- end }}
{{- if $esapi.Legacy }}
//
// The StreamCloser member should be set to the underlying io.Closer,
// (e.g. http.Response.Body), that will be closed when the stream Close method
// is called.
{{- end }}
//
//   es := New{{ $esapi.Name }}(func(o *{{ $esapi.Name}}{
{{- if $inputStream }}
//       es.Writer = myMockStreamWriter
{{- end }}
{{- if $outputStream }}
//       es.Reader = myMockStreamReader
{{- end }}
{{- if $esapi.Legacy }}
//       es.StreamCloser = myMockStreamCloser
{{- end }}
//   })
func New{{ $esapi.Name }}(opts ...func(*{{ $esapi.Name}})) *{{ $esapi.Name }} {
	es := &{{ $esapi.Name }} {
		done: make(chan struct{}),
		err: eventstreamapi.NewOnceError(),
	}

	for _, fn := range opts {
		fn(es)
	}

	return es
}

{{- if $esapi.Legacy }}

	func (es *{{ $esapi.Name }}) setStreamCloser(r *request.Request) {
		es.StreamCloser = r.HTTPResponse.Body
	}
{{- end }}

func (es *{{ $esapi.Name }}) runOnStreamPartClose(r *request.Request) {
	if es.done == nil {
		return
	}
	go es.waitStreamPartClose()

}

func (es *{{ $esapi.Name }}) waitStreamPartClose() {
	{{- if $inputStream }}
		var inputErrCh <-chan struct{}
		if v, ok := es.Writer.(interface{ErrorSet() <-chan struct{}}); ok {
			inputErrCh = v.ErrorSet()
		}
	{{- end }}
	{{- if $outputStream }}
		var outputErrCh <-chan struct{}
		if v, ok := es.Reader.(interface{ErrorSet() <-chan struct{}}); ok {
			outputErrCh = v.ErrorSet()
		}
		var outputClosedCh <- chan struct{}
		if v, ok := es.Reader.(interface{Closed() <-chan struct{}}); ok {
			outputClosedCh = v.Closed()
		}
	{{- end }}

	select {
		case <-es.done:

		{{- if $inputStream }}
		case <-inputErrCh:
			es.err.SetError(es.Writer.Err())
			es.Close()
		{{- end }}

		{{- if $outputStream }}
		case <-outputErrCh:
			es.err.SetError(es.Reader.Err())
			es.Close()
		case <-outputClosedCh:
			if err := es.Reader.Err(); err != nil {
				es.err.SetError(es.Reader.Err())
			}
			es.Close()
		{{- end }}
	}
}

{{- if $inputStream }}

	{{- if eq .API.Metadata.Protocol "json" }}

		func {{ $esapi.StreamInputEventTypeGetterName }}(event {{ $inputStream.EventGroupName }}) (string, error) {
			if _, ok := event.({{ $.InputRef.GoType }}); ok {
				return "initial-request", nil
			}
			return {{ $inputStream.StreamEventTypeGetterName }}(event)
		}
	{{- end }}

	func (es *{{ $esapi.Name }}) setupInputPipe(r *request.Request) {
			inputReader, inputWriter := io.Pipe()
			r.SetStreamingBody(inputReader)
			es.inputWriter = inputWriter
	}

	// Send writes the event to the stream blocking until the event is written.
	// Returns an error if the event was not written.
	//
	// These events are:
	// {{ range $_, $event := $inputStream.Events }}
	//     * {{ $event.Shape.ShapeName }}
	{{- end }}
	func (es *{{ $esapi.Name }}) Send(ctx aws.Context, event {{ $inputStream.EventGroupName }}) error {
		return es.Writer.Send(ctx, event)
	}

	func (es *{{ $esapi.Name }}) runInputStream(r *request.Request) {
		var opts []func(*eventstream.Encoder)
		if r.Config.Logger != nil && r.Config.LogLevel.Matches(aws.LogDebugWithEventStreamBody) {
			opts = append(opts, eventstream.EncodeWithLogger(r.Config.Logger))
		}
		var encoder eventstreamapi.Encoder = eventstream.NewEncoder(es.inputWriter, opts...)

		var closer aws.MultiCloser
		{{- if $.ShouldSignRequestBody }}
			{{- $_ := $.API.AddSDKImport "aws/signer/v4" }}
			sigSeed, err := v4.GetSignedRequestSignature(r.HTTPRequest)
			if err != nil {
				r.Error = awserr.New(request.ErrCodeSerialization,
					"unable to get initial request's signature", err)
				return
			}
			signer := eventstreamapi.NewSignEncoder(
				v4.NewStreamSigner(r.ClientInfo.SigningRegion, r.ClientInfo.SigningName,
					sigSeed, r.Config.Credentials),
				encoder,
			)
			encoder = signer
			closer = append(closer, signer)
		{{- end }}
		closer = append(closer, es.inputWriter)

		eventWriter := eventstreamapi.NewEventWriter(encoder,
			protocol.HandlerPayloadMarshal{
				Marshalers: r.Handlers.BuildStream,
			},
			{{- if eq .API.Metadata.Protocol "json" }}
				{{ $esapi.StreamInputEventTypeGetterName }},
			{{- else }}
				{{ $inputStream.StreamEventTypeGetterName }},
			{{- end }}
		)

		es.Writer = &{{ $inputStream.StreamWriterImplName }}{
			StreamWriter: eventstreamapi.NewStreamWriter(eventWriter, closer),
		}
	}

	{{- if eq .API.Metadata.Protocol "json" }}
		func (es *{{ $esapi.Name }}) sendInitialEvent(r *request.Request) {
			if err := es.Send(es.input); err != nil {
				r.Error = err
			}
		}
	{{- end }}
{{- end }}

{{- if $outputStream }}
	{{- if eq .API.Metadata.Protocol "json" }}

		type {{ $esapi.StreamOutputUnmarshalerForEventName }} struct {
			unmarshalerForEvent func(string) (eventstreamapi.Unmarshaler, error)
			output {{ $.OutputRef.GoType }}
		}
		func (e {{ $esapi.StreamOutputUnmarshalerForEventName }}) UnmarshalerForEventName(eventType string) (eventstreamapi.Unmarshaler, error) {
			if eventType == "initial-response" {
				return e.output, nil
			}
			return e.unmarshalerForEvent(eventType)
		}
	{{- end }}

	// Events returns a channel to read events from.
	//
	// These events are:
	// {{ range $_, $event := $outputStream.Events }}
	//     * {{ $event.Shape.ShapeName }}
	{{- end }}
    //     * {{ $outputStream.StreamUnknownEventName }}
	func (es *{{ $esapi.Name }}) Events() <-chan {{ $outputStream.EventGroupName }} {
		return es.Reader.Events()
	}

	func (es *{{ $esapi.Name }}) runOutputStream(r *request.Request) {
		var opts []func(*eventstream.Decoder)
		if r.Config.Logger != nil && r.Config.LogLevel.Matches(aws.LogDebugWithEventStreamBody) {
			opts = append(opts, eventstream.DecodeWithLogger(r.Config.Logger))
		}

		unmarshalerForEvent := {{ $outputStream.StreamUnmarshalerForEventName }}{
			metadata: protocol.ResponseMetadata{
				StatusCode: r.HTTPResponse.StatusCode,
				RequestID: r.RequestID,
			},
		}.UnmarshalerForEventName
		{{- if eq .API.Metadata.Protocol "json" }}
			unmarshalerForEvent = {{ $esapi.StreamOutputUnmarshalerForEventName }}{
				unmarshalerForEvent: unmarshalerForEvent,
				output: es.output,
			}.UnmarshalerForEventName
		{{- end }}

		decoder := eventstream.NewDecoder(r.HTTPResponse.Body, opts...)
		eventReader := eventstreamapi.NewEventReader(decoder,
			protocol.HandlerPayloadUnmarshal{
				Unmarshalers: r.Handlers.UnmarshalStream,
			},
			unmarshalerForEvent,
		)

		es.outputReader = r.HTTPResponse.Body
		es.Reader = {{ $outputStream.StreamReaderImplConstructorName }}(eventReader)
	}

	{{- if eq .API.Metadata.Protocol "json" }}
		func (es *{{ $esapi.Name }}) recvInitialEvent(r *request.Request) {
			// Wait for the initial response event, which must be the first
			// event to be received from the API.
			select {
			case event, ok := <- es.Events():
				if !ok {
					return
				}

				v, ok := event.({{ $.OutputRef.GoType }})
				if !ok || v == nil {
					r.Error = awserr.New(
						request.ErrCodeSerialization,
						fmt.Sprintf("invalid event, %T, expect %T, %v",
							event, ({{ $.OutputRef.GoType }})(nil), v),
						nil,
					)
					return
				}

				*es.output = *v
				es.output.{{ $.EventStreamAPI.OutputMemberName  }} = es
			}
		}
	{{- end }}
{{- end }}

// Close closes the stream. This will also cause the stream to be closed.
// Close must be called when done using the stream API. Not calling Close
// may result in resource leaks.
{{- if $inputStream }}
//
// Will close the underlying EventStream writer, and no more events can be
// sent.
{{- end }}
{{- if $outputStream }}
//
// You can use the closing of the Reader's Events channel to terminate your
// application's read from the API's stream.
{{- end }}
//
func (es *{{ $esapi.Name }}) Close() (err error) {
	es.closeOnce.Do(es.safeClose)
	return es.Err()
}

func (es *{{ $esapi.Name }}) safeClose() {
	if es.done != nil {
		close(es.done)
	}

	{{- if $inputStream }}

		t := time.NewTicker(time.Second)
		defer t.Stop()
		writeCloseDone := make(chan error)
		go func() {
			if err := es.Writer.Close(); err != nil {
				es.err.SetError(err)
			}
			close(writeCloseDone)
		}()
		select {
		case <-t.C:
		case <-writeCloseDone:
		}
		if es.inputWriter != nil {
			es.inputWriter.Close()
		}
	{{- end }}

	{{- if $outputStream }}

		es.Reader.Close()
		if es.outputReader != nil {
			es.outputReader.Close()
		}
	{{- end }}

	{{- if $esapi.Legacy }}

		es.StreamCloser.Close()
	{{- end }}
}

// Err returns any error that occurred while reading or writing EventStream
// Events from the service API's response. Returns nil if there were no errors.
func (es *{{ $esapi.Name }}) Err() error {
	if err := es.err.Err(); err != nil {
		return err
	}

	{{- if $inputStream }}
		if err := es.Writer.Err(); err != nil {
			return err
		}
	{{- end }}

	{{- if $outputStream }}
		if err := es.Reader.Err(); err != nil {
			return err
		}
	{{- end }}

	return nil
}
`

func renderEventStreamShape(w io.Writer, s *Shape) error {
	// Imports needed by the EventStream APIs.
	s.API.AddImport("fmt")
	s.API.AddImport("bytes")
	s.API.AddImport("io")
	s.API.AddImport("sync")
	s.API.AddSDKImport("aws")
	s.API.AddSDKImport("aws/awserr")
	s.API.AddSDKImport("private/protocol/eventstream")
	s.API.AddSDKImport("private/protocol/eventstream/eventstreamapi")

	return eventStreamShapeTmpl.Execute(w, s)
}

var eventStreamShapeTmpl = func() *template.Template {
	t := template.Must(
		template.New("eventStreamShapeTmplDef").
			Parse(eventStreamShapeTmplDef),
	)
	template.Must(
		t.AddParseTree(
			"eventStreamShapeWriterTmpl", eventStreamShapeWriterTmpl.Tree),
	)
	template.Must(
		t.AddParseTree(
			"eventStreamShapeReaderTmpl", eventStreamShapeReaderTmpl.Tree),
	)

	return t
}()

const eventStreamShapeTmplDef = `
{{- $eventStream := $.EventStream }}
{{- $eventStreamEventGroup := printf "%sEvent" $eventStream.Name }}

// {{ $eventStreamEventGroup }} groups together all EventStream
// events writes for {{ $eventStream.Name }}.
//
// These events are:
// {{ range $_, $event := $eventStream.Events }}
//     * {{ $event.Shape.ShapeName }}
{{- end }}
type {{ $eventStreamEventGroup }} interface {
	event{{ $eventStream.Name }}()
	eventstreamapi.Marshaler
	eventstreamapi.Unmarshaler
}

{{- if $.IsInputEventStream }}
	{{- template "eventStreamShapeWriterTmpl" $ }}
{{- end }}

{{- if $.IsOutputEventStream }}
	{{- template "eventStreamShapeReaderTmpl" $ }}
{{- end }}
`

// EventStreamHeaderTypeMap provides the mapping of a EventStream Header's
// Value type to the shape reference's member type.
type EventStreamHeaderTypeMap struct {
	Header string
	Member string
}

// Returns if the event has any members which are not the event's blob payload,
// nor a header.
func eventHasNonBlobPayloadMembers(s *Shape) bool {
	num := len(s.MemberRefs)
	for _, ref := range s.MemberRefs {
		if ref.IsEventHeader || (ref.IsEventPayload && (ref.Shape.Type == "blob" || ref.Shape.Type == "string")) {
			num--
		}
	}
	return num > 0
}

func setEventHeaderValueForType(s *Shape, memVar string) string {
	switch s.Type {
	case "blob":
		return fmt.Sprintf("eventstream.BytesValue(%s)", memVar)
	case "string":
		return fmt.Sprintf("eventstream.StringValue(*%s)", memVar)
	case "boolean":
		return fmt.Sprintf("eventstream.BoolValue(*%s)", memVar)
	case "byte":
		return fmt.Sprintf("eventstream.Int8Value(int8(*%s))", memVar)
	case "short":
		return fmt.Sprintf("eventstream.Int16Value(int16(*%s))", memVar)
	case "integer":
		return fmt.Sprintf("eventstream.Int32Value(int32(*%s))", memVar)
	case "long":
		return fmt.Sprintf("eventstream.Int64Value(*%s)", memVar)
	case "float":
		return fmt.Sprintf("eventstream.Float32Value(float32(*%s))", memVar)
	case "double":
		return fmt.Sprintf("eventstream.Float64Value(*%s)", memVar)
	case "timestamp":
		return fmt.Sprintf("eventstream.TimestampValue(*%s)", memVar)
	default:
		panic(fmt.Sprintf("value type %s not supported for event headers, %s", s.Type, s.ShapeName))
	}
}

func shapeMessageType(s *Shape) string {
	if s.Exception {
		return "eventstreamapi.ExceptionMessageType"
	}
	return "eventstreamapi.EventMessageType"
}

var eventStreamEventShapeTmplFuncs = template.FuncMap{
	"EventStreamHeaderTypeMap": func(ref *ShapeRef) EventStreamHeaderTypeMap {
		switch ref.Shape.Type {
		case "boolean":
			return EventStreamHeaderTypeMap{Header: "bool", Member: "bool"}
		case "byte":
			return EventStreamHeaderTypeMap{Header: "int8", Member: "int64"}
		case "short":
			return EventStreamHeaderTypeMap{Header: "int16", Member: "int64"}
		case "integer":
			return EventStreamHeaderTypeMap{Header: "int32", Member: "int64"}
		case "long":
			return EventStreamHeaderTypeMap{Header: "int64", Member: "int64"}
		case "timestamp":
			return EventStreamHeaderTypeMap{Header: "time.Time", Member: "time.Time"}
		case "blob":
			return EventStreamHeaderTypeMap{Header: "[]byte", Member: "[]byte"}
		case "string":
			return EventStreamHeaderTypeMap{Header: "string", Member: "string"}
		case "uuid":
			return EventStreamHeaderTypeMap{Header: "[]byte", Member: "[]byte"}
		default:
			panic("unsupported EventStream header type, " + ref.Shape.Type)
		}
	},
	"EventHeaderValueForType":  setEventHeaderValueForType,
	"ShapeMessageType":         shapeMessageType,
	"HasNonBlobPayloadMembers": eventHasNonBlobPayloadMembers,
}

// Template for an EventStream Event shape. This is a normal API shape that is
// decorated as an EventStream Event.
//
// Executed in the context of a Shape.
var eventStreamEventShapeTmpl = template.Must(template.New("eventStreamEventShapeTmpl").
	Funcs(eventStreamEventShapeTmplFuncs).Parse(`
{{ range $_, $eventStream := $.EventFor }}
	// The {{ $.ShapeName }} is and event in the {{ $eventStream.Name }} group of events.
	func (s *{{ $.ShapeName }}) event{{ $eventStream.Name }}() {}
{{ end }}

// UnmarshalEvent unmarshals the EventStream Message into the {{ $.ShapeName }} value.
// This method is only used internally within the SDK's EventStream handling.
func (s *{{ $.ShapeName }}) UnmarshalEvent(
	payloadUnmarshaler protocol.PayloadUnmarshaler,
	msg eventstream.Message,
) error {
	{{- range $memName, $memRef := $.MemberRefs }}
		{{- if $memRef.IsEventHeader }}
			if hv := msg.Headers.Get("{{ $memName }}"); hv != nil {
				{{ $types := EventStreamHeaderTypeMap $memRef -}}
				v := hv.Get().({{ $types.Header }})
				{{- if ne $types.Header $types.Member }}
					m := {{ $types.Member }}(v)
					s.{{ $memName }} = {{ if $memRef.UseIndirection }}&{{ end }}m
				{{- else }}
					s.{{ $memName }} = {{ if $memRef.UseIndirection }}&{{ end }}v
				{{- end }}
			}
		{{- else if (and ($memRef.IsEventPayload) (eq $memRef.Shape.Type "blob")) }}
			s.{{ $memName }} = make([]byte, len(msg.Payload))
			copy(s.{{ $memName }}, msg.Payload)
		{{- else if (and ($memRef.IsEventPayload) (eq $memRef.Shape.Type "string")) }}
			s.{{ $memName }} = aws.String(string(msg.Payload))
		{{- end }}
	{{- end }}
	{{- if HasNonBlobPayloadMembers $ }}
		if err := payloadUnmarshaler.UnmarshalPayload(
			bytes.NewReader(msg.Payload), s,
		); err != nil {
			return err
		}
	{{- end }}
	return nil
}

// MarshalEvent marshals the type into an stream event value. This method
// should only used internally within the SDK's EventStream handling.
func (s *{{ $.ShapeName}}) MarshalEvent(pm protocol.PayloadMarshaler) (msg eventstream.Message, err error) {
	msg.Headers.Set(eventstreamapi.MessageTypeHeader, eventstream.StringValue({{ ShapeMessageType $ }}))

	{{- range $memName, $memRef := $.MemberRefs }}
		{{- if $memRef.IsEventHeader }}
			{{ $memVar := printf "s.%s" $memName -}}
			{{ $typedMem := EventHeaderValueForType $memRef.Shape $memVar -}}
			msg.Headers.Set("{{ $memName }}", {{ $typedMem }})
		{{- else if (and ($memRef.IsEventPayload) (eq $memRef.Shape.Type "blob")) }}
			msg.Headers.Set(":content-type", eventstream.StringValue("application/octet-stream"))
			msg.Payload = s.{{ $memName }}
		{{- else if (and ($memRef.IsEventPayload) (eq $memRef.Shape.Type "string")) }}
			msg.Payload = []byte(aws.StringValue(s.{{ $memName }}))
		{{- end }}
	{{- end }}
	{{- if HasNonBlobPayloadMembers $ }}
		var buf bytes.Buffer
		if err = pm.MarshalPayload(&buf, s); err != nil {
			return eventstream.Message{}, err
		}
		msg.Payload = buf.Bytes()
	{{- end }}
	return msg, err
}
`))
