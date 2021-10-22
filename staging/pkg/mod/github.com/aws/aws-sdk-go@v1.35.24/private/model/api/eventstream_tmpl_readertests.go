// +build codegen

package api

import (
	"text/template"
)

var eventStreamReaderTestTmpl = template.Must(
	template.New("eventStreamReaderTestTmpl").Funcs(template.FuncMap{
		"ValueForType":             valueForType,
		"HasNonBlobPayloadMembers": eventHasNonBlobPayloadMembers,
		"EventHeaderValueForType":  setEventHeaderValueForType,
		"Map":                      templateMap,
		"OptionalAddInt": func(do bool, a, b int) int {
			if !do {
				return a
			}
			return a + b
		},
		"HasNonEventStreamMember": func(s *Shape) bool {
			for _, ref := range s.MemberRefs {
				if !ref.Shape.IsEventStream {
					return true
				}
			}
			return false
		},
	}).Parse(`
{{ range $opName, $op := $.Operations }}
	{{ if $op.EventStreamAPI }}
		{{ if  $op.EventStreamAPI.OutputStream }}
			{{ template "event stream outputStream tests" $op.EventStreamAPI }}
		{{ end }}
	{{ end }}
{{ end }}

type loopReader struct {
	source *bytes.Reader
}

func (c *loopReader) Read(p []byte) (int, error) {
	if c.source.Len() == 0 {
		c.source.Seek(0, 0)
	}

	return c.source.Read(p)
}

{{ define "event stream outputStream tests" }}
	func Test{{ $.Operation.ExportedName }}_Read(t *testing.T) {
		expectEvents, eventMsgs := mock{{ $.Operation.ExportedName }}ReadEvents()
		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			eventstreamtest.ServeEventStream{
				T:      t,
				Events: eventMsgs,
			},
			true,
		)
		if err != nil {
			t.Fatalf("expect no error, %v", err)
		}
		defer cleanupFn()

		svc := New(sess)
		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			t.Fatalf("expect no error got, %v", err)
		}
		defer resp.GetStream().Close()

		{{- if eq $.Operation.API.Metadata.Protocol "json" }}
			{{- if HasNonEventStreamMember $.Operation.OutputRef.Shape }}
				expectResp := expectEvents[0].(*{{ $.Operation.OutputRef.Shape.ShapeName }})
				{{- range $name, $ref := $.Operation.OutputRef.Shape.MemberRefs }}
					{{- if not $ref.Shape.IsEventStream }}
						if e, a := expectResp.{{ $name }}, resp.{{ $name }}; !reflect.DeepEqual(e,a) {
							t.Errorf("expect %v, got %v", e, a)
						}
					{{- end }}
				{{- end }}
			{{- end }}
			// Trim off response output type pseudo event so only event messages remain.
			expectEvents = expectEvents[1:]
		{{ end }}

		var i int
		for event := range resp.GetStream().Events() {
			if event == nil {
				t.Errorf("%d, expect event, got nil", i)
			}
			if e, a := expectEvents[i], event; !reflect.DeepEqual(e, a) {
				t.Errorf("%d, expect %T %v, got %T %v", i, e, e, a, a)
			}
			i++
		}

		if err := resp.GetStream().Err(); err != nil {
			t.Errorf("expect no error, %v", err)
		}
	}

	func Test{{ $.Operation.ExportedName }}_ReadClose(t *testing.T) {
		_, eventMsgs := mock{{ $.Operation.ExportedName }}ReadEvents()
		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			eventstreamtest.ServeEventStream{
				T:      t,
				Events: eventMsgs,
			},
			true,
		)
		if err != nil {
			t.Fatalf("expect no error, %v", err)
		}
		defer cleanupFn()

		svc := New(sess)
		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			t.Fatalf("expect no error got, %v", err)
		}

		{{ if gt (len $.OutputStream.Events) 0 -}}
			// Assert calling Err before close does not close the stream.
			resp.GetStream().Err()
			select {
			case _, ok := <-resp.GetStream().Events():
				if !ok {
					t.Fatalf("expect stream not to be closed, but was")
				}
			default:
			}
		{{- end }}

		resp.GetStream().Close()
		<-resp.GetStream().Events()

		if err := resp.GetStream().Err(); err != nil {
			t.Errorf("expect no error, %v", err)
		}
	}

	func Test{{ $.Operation.ExportedName }}_ReadUnknownEvent(t *testing.T) {
		expectEvents, eventMsgs := mock{{ $.Operation.ExportedName }}ReadEvents()

		{{- if eq $.Operation.API.Metadata.Protocol "json" }}
			eventOffset := 1
		{{- else }}
			var eventOffset int
		{{- end }}

		unknownEvent := eventstream.Message{
			Headers: eventstream.Headers{
				eventstreamtest.EventMessageTypeHeader,
				{
					Name:  eventstreamapi.EventTypeHeader,
					Value: eventstream.StringValue("UnknownEventName"),
				},
			},
			Payload: []byte("some unknown event"),
		}

		eventMsgs = append(eventMsgs[:eventOffset],
			append([]eventstream.Message{unknownEvent}, eventMsgs[eventOffset:]...)...)

		expectEvents = append(expectEvents[:eventOffset],
			append([]{{ $.OutputStream.Name }}Event{
					&{{ $.OutputStream.StreamUnknownEventName }}{
						Type: "UnknownEventName",
						Message: unknownEvent,
					},
				},
				expectEvents[eventOffset:]...)...)

		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			eventstreamtest.ServeEventStream{
				T:      t,
				Events: eventMsgs,
			},
			true,
		)
		if err != nil {
			t.Fatalf("expect no error, %v", err)
		}
		defer cleanupFn()

		svc := New(sess)
		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			t.Fatalf("expect no error got, %v", err)
		}
		defer resp.GetStream().Close()

		{{- if eq $.Operation.API.Metadata.Protocol "json" }}
			// Trim off response output type pseudo event so only event messages remain.
			expectEvents = expectEvents[1:]
		{{ end }}

		var i int
		for event := range resp.GetStream().Events() {
			if event == nil {
				t.Errorf("%d, expect event, got nil", i)
			}
			if e, a := expectEvents[i], event; !reflect.DeepEqual(e, a) {
				t.Errorf("%d, expect %T %v, got %T %v", i, e, e, a, a)
			}
			i++
		}

		if err := resp.GetStream().Err(); err != nil {
			t.Errorf("expect no error, %v", err)
		}
	}

	func Benchmark{{ $.Operation.ExportedName }}_Read(b *testing.B) {
		_, eventMsgs := mock{{ $.Operation.ExportedName }}ReadEvents()
		var buf bytes.Buffer
		encoder := eventstream.NewEncoder(&buf)
		for _, msg := range eventMsgs {
			if err := encoder.Encode(msg); err != nil {
				b.Fatalf("failed to encode message, %v", err)
			}
		}
		stream := &loopReader{source: bytes.NewReader(buf.Bytes())}

		sess := unit.Session
		svc := New(sess, &aws.Config{
			Endpoint:               aws.String("https://example.com"),
			DisableParamValidation: aws.Bool(true),
		})
		svc.Handlers.Send.Swap(corehandlers.SendHandler.Name,
			request.NamedHandler{Name: "mockSend",
				Fn: func(r *request.Request) {
					r.HTTPResponse = &http.Response{
						Status:     "200 OK",
						StatusCode: 200,
						Header:     http.Header{},
						Body:       ioutil.NopCloser(stream),
					}
				},
			},
		)

		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			b.Fatalf("failed to create request, %v", err)
		}
		defer resp.GetStream().Close()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			if err = resp.GetStream().Err(); err != nil {
				b.Fatalf("expect no error, got %v", err)
			}
			event := <-resp.GetStream().Events()
			if event == nil {
				b.Fatalf("expect event, got nil, %v, %d", resp.GetStream().Err(), i)
			}
		}
	}

	func mock{{ $.Operation.ExportedName }}ReadEvents() (
		[]{{ $.OutputStream.Name }}Event,
		[]eventstream.Message,
	) {
		expectEvents := []{{ $.OutputStream.Name }}Event {
			{{- if eq $.Operation.API.Metadata.Protocol "json" }}
				{{- template "set event type" $.Operation.OutputRef.Shape }}
			{{- end }}
			{{- range $_, $event := $.OutputStream.Events }}
				{{- template "set event type" $event.Shape }}
			{{- end }}
		}

		var marshalers request.HandlerList
		marshalers.PushBackNamed({{ $.API.ProtocolPackage }}.BuildHandler)
		payloadMarshaler := protocol.HandlerPayloadMarshal{
			Marshalers: marshalers,
		}
		_ = payloadMarshaler

		eventMsgs := []eventstream.Message{
			{{- if eq $.Operation.API.Metadata.Protocol "json" }}
				{{- template "set event message" Map "idx" 0 "parentShape" $.Operation.OutputRef.Shape "eventName" "initial-response" }}
			{{- end }}
			{{- range $idx, $event := $.OutputStream.Events }}
				{{- $offsetIdx := OptionalAddInt (eq $.Operation.API.Metadata.Protocol "json") $idx 1 }}
				{{- template "set event message" Map "idx" $offsetIdx "parentShape" $event.Shape "eventName" $event.Name }}
			{{- end }}
		}

		return expectEvents, eventMsgs
	}

	{{- if $.OutputStream.Exceptions }}
		func Test{{ $.Operation.ExportedName }}_ReadException(t *testing.T) {
			expectEvents := []{{ $.OutputStream.Name }}Event {
				{{- if eq $.Operation.API.Metadata.Protocol "json" }}
					{{- template "set event type" $.Operation.OutputRef.Shape }}
				{{- end }}

				{{- $exception := index $.OutputStream.Exceptions 0 }}
				{{- template "set event type" $exception.Shape }}
			}

			var marshalers request.HandlerList
			marshalers.PushBackNamed({{ $.API.ProtocolPackage }}.BuildHandler)
			payloadMarshaler := protocol.HandlerPayloadMarshal{
				Marshalers: marshalers,
			}

			eventMsgs := []eventstream.Message{
				{{- if eq $.Operation.API.Metadata.Protocol "json" }}
					{{- template "set event message" Map "idx" 0 "parentShape" $.Operation.OutputRef.Shape "eventName" "initial-response" }}
				{{- end }}

				{{- $offsetIdx := OptionalAddInt (eq $.Operation.API.Metadata.Protocol "json") 0 1 }}
				{{- $exception := index $.OutputStream.Exceptions 0 }}
				{{- template "set event message" Map "idx" $offsetIdx "parentShape" $exception.Shape "eventName" $exception.Name }}
			}

			sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
				eventstreamtest.ServeEventStream{
					T:      t,
					Events: eventMsgs,
				},
				true,
			)
			if err != nil {
				t.Fatalf("expect no error, %v", err)
			}
			defer cleanupFn()

			svc := New(sess)
			resp, err := svc.{{ $.Operation.ExportedName }}(nil)
			if err != nil {
				t.Fatalf("expect no error got, %v", err)
			}

			defer resp.GetStream().Close()

			<-resp.GetStream().Events()

			err = resp.GetStream().Err()
			if err == nil {
				t.Fatalf("expect err, got none")
			}

			expectErr := {{ ValueForType $exception.Shape nil }}
			aerr, ok := err.(awserr.Error)
			if !ok {
				t.Errorf("expect exception, got %T, %#v", err, err)
			}
			if e, a := expectErr.Code(), aerr.Code(); e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
			if e, a := expectErr.Message(), aerr.Message(); e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			if e, a := expectErr, aerr; !reflect.DeepEqual(e, a) {
				t.Errorf("expect error %+#v, got %+#v", e, a)
			}
		}

		{{- range $_, $exception := $.OutputStream.Exceptions }}
			var _ awserr.Error = (*{{ $exception.Shape.ShapeName }})(nil)
		{{- end }}

	{{ end }}
{{ end }}

{{/* Params: *Shape */}}
{{ define "set event type" }}
	&{{ $.ShapeName }}{
		{{- if $.Exception }}
			RespMetadata: protocol.ResponseMetadata{
				StatusCode: 200,
			},
		{{- end }}
		{{- range $memName, $memRef := $.MemberRefs }}
			{{- if not $memRef.Shape.IsEventStream }}
				{{ $memName }}: {{ ValueForType $memRef.Shape nil }},
			{{- end }}
		{{- end }}
	},
{{- end }}

{{/* Params: idx:int, parentShape:*Shape, eventName:string */}}
{{ define "set event message" }}
	{
		Headers: eventstream.Headers{
			{{- if $.parentShape.Exception }}
				eventstreamtest.EventExceptionTypeHeader,
				{
					Name:  eventstreamapi.ExceptionTypeHeader,
					Value: eventstream.StringValue("{{ $.eventName }}"),
				},
			{{- else }}
				eventstreamtest.EventMessageTypeHeader,
				{
					Name:  eventstreamapi.EventTypeHeader,
					Value: eventstream.StringValue("{{ $.eventName }}"),
				},
			{{- end }}
			{{- range $memName, $memRef := $.parentShape.MemberRefs }}
				{{- template "set event message header" Map "idx" $.idx "parentShape" $.parentShape "memName" $memName "memRef" $memRef }}
			{{- end }}
		},
		{{- template "set event message payload" Map "idx" $.idx "parentShape" $.parentShape }}
	},
{{- end }}

{{/* Params: idx:int, parentShape:*Shape, memName:string, memRef:*ShapeRef */}}
{{ define "set event message header" }}
	{{- if $.memRef.IsEventHeader }}
		{
			Name: "{{ $.memName }}",
			{{- $shapeValueVar := printf "expectEvents[%d].(%s).%s" $.idx $.parentShape.GoType $.memName }}
			Value: {{ EventHeaderValueForType $.memRef.Shape $shapeValueVar }},
		},
	{{- end }}
{{- end }}

{{/* Params: idx:int, parentShape:*Shape, memName:string, memRef:*ShapeRef */}}
{{ define "set event message payload" }}
	{{- $payloadMemName := $.parentShape.PayloadRefName }}
	{{- if HasNonBlobPayloadMembers $.parentShape }}
		Payload: eventstreamtest.MarshalEventPayload(payloadMarshaler, expectEvents[{{ $.idx }}]),
	{{- else if $payloadMemName }}
		{{- $shapeType := (index $.parentShape.MemberRefs $payloadMemName).Shape.Type }}
		{{- if eq $shapeType "blob" }}
			Payload: expectEvents[{{ $.idx }}].({{ $.parentShape.GoType }}).{{ $payloadMemName }},
		{{- else if eq $shapeType "string" }}
			Payload: []byte(*expectEvents[{{ $.idx }}].({{ $.parentShape.GoType }}).{{ $payloadMemName }}),
		{{- end }}
	{{- end }}
{{- end }}
`))
