// +build codegen

package api

import (
	"text/template"
)

var eventStreamWriterTestTmpl = template.Must(
	template.New("eventStreamWriterTestTmpl").Funcs(template.FuncMap{
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
		{{ if  $op.EventStreamAPI.InputStream }}
			{{ template "event stream inputStream tests" $op.EventStreamAPI }}
		{{ end }}
	{{ end }}
{{ end }}

{{ define "event stream inputStream tests" }}
	func Test{{ $.Operation.ExportedName }}_Write(t *testing.T) {
		clientEvents, expectedClientEvents := mock{{ $.Operation.ExportedName }}WriteEvents()

		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			&eventstreamtest.ServeEventStream{
				T:             t,
				ClientEvents:  expectedClientEvents,
				BiDirectional: true,
			},
			true)
		defer cleanupFn()

		svc := New(sess)
		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			t.Fatalf("expect no error, got %v", err)
		}

		stream := resp.GetStream()

		for _, event := range clientEvents {
			err = stream.Send(context.Background(), event)
			if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
		}

		if err := stream.Close(); err != nil {
			t.Errorf("expect no error, got %v", err)
		}
	}

	func Test{{ $.Operation.ExportedName }}_WriteClose(t *testing.T) {
		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			eventstreamtest.ServeEventStream{T: t, BiDirectional: true},
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

		// Assert calling Err before close does not close the stream.
		resp.GetStream().Err()
		{{ $eventShape := index $.InputStream.Events 0 }}
		err = resp.GetStream().Send(context.Background(), &{{ $eventShape.Shape.ShapeName }}{})
		if err != nil {
			t.Fatalf("expect no error, got %v", err)
		}

		resp.GetStream().Close()

		if err := resp.GetStream().Err(); err != nil {
			t.Errorf("expect no error, %v", err)
		}
	}

	func Test{{ $.Operation.ExportedName }}_WriteError(t *testing.T) {
		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			eventstreamtest.ServeEventStream{
				T:               t,
				BiDirectional:   true,
				ForceCloseAfter: time.Millisecond * 500,
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
		{{ $eventShape := index $.InputStream.Events 0 }}
		for {
			err = resp.GetStream().Send(context.Background(), &{{ $eventShape.Shape.ShapeName }}{})
			if err != nil {
				if strings.Contains("unable to send event", err.Error()) {
					t.Errorf("expected stream closed error, got %v", err)
				}
				break
			}
		}
	}

	func Test{{ $.Operation.ExportedName }}_ReadWrite(t *testing.T) {
		expectedServiceEvents, serviceEvents := mock{{ $.Operation.ExportedName }}ReadEvents()
		clientEvents, expectedClientEvents := mock{{ $.Operation.ExportedName }}WriteEvents()

		sess, cleanupFn, err := eventstreamtest.SetupEventStreamSession(t,
			&eventstreamtest.ServeEventStream{
				T:             t,
				ClientEvents:  expectedClientEvents,
				Events:        serviceEvents,
				BiDirectional: true,
			},
			true)
		defer cleanupFn()

		svc := New(sess)
		resp, err := svc.{{ $.Operation.ExportedName }}(nil)
		if err != nil {
			t.Fatalf("expect no error, got %v", err)
		}

		stream := resp.GetStream()
		defer stream.Close()

		var wg sync.WaitGroup

		wg.Add(1)
		go func() {
			defer wg.Done()
			var i int
			for event := range resp.GetStream().Events() {
				if event == nil {
					t.Errorf("%d, expect event, got nil", i)
				}
				if e, a := expectedServiceEvents[i], event; !reflect.DeepEqual(e, a) {
					t.Errorf("%d, expect %T %v, got %T %v", i, e, e, a, a)
				}
				i++
			}
		}()

		for _, event := range clientEvents {
			err = stream.Send(context.Background(), event)
			if err != nil {
				t.Errorf("expect no error, got %v", err)
			}
		}

		resp.GetStream().Close()

		wg.Wait()

		if err := resp.GetStream().Err(); err != nil {
			t.Errorf("expect no error, %v", err)
		}
	}

	func mock{{ $.Operation.ExportedName }}WriteEvents() (
		[]{{ $.InputStream.Name }}Event,
		[]eventstream.Message,
	) {
		inputEvents := []{{ $.InputStream.Name }}Event {
			{{- if eq $.Operation.API.Metadata.Protocol "json" }}
				{{- template "set event type" $.Operation.InputRef.Shape }}
			{{- end }}
			{{- range $_, $event := $.InputStream.Events }}
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
			{{- range $idx, $event := $.InputStream.Events }}
				{{- template "set event message" Map "idx" $idx "parentShape" $event.Shape "eventName" $event.Name }}
			{{- end }}
		}

		return inputEvents, eventMsgs
	}
{{ end }}

{{/* Params: *Shape */}}
{{ define "set event type" }}
	&{{ $.ShapeName }}{
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
			eventstreamtest.EventMessageTypeHeader,
			{{- range $memName, $memRef := $.parentShape.MemberRefs }}
				{{- template "set event message header" Map "idx" $.idx "parentShape" $.parentShape "memName" $memName "memRef" $memRef }}
			{{- end }}
			{
				Name:  eventstreamapi.EventTypeHeader,
				Value: eventstream.StringValue("{{ $.eventName }}"),
			},
		},
		{{- template "set event message payload" Map "idx" $.idx "parentShape" $.parentShape }}
	},
{{- end }}

{{/* Params: idx:int, parentShape:*Shape, memName:string, memRef:*ShapeRef */}}
{{ define "set event message header" }}
	{{- if (and ($.memRef.IsEventPayload) (eq $.memRef.Shape.Type "blob")) }}
		{
			Name: ":content-type",
			Value: eventstream.StringValue("application/octet-stream"),
		},
	{{- else if $.memRef.IsEventHeader }}
		{
			Name: "{{ $.memName }}",
			{{- $shapeValueVar := printf "inputEvents[%d].(%s).%s" $.idx $.parentShape.GoType $.memName }}
			Value: {{ EventHeaderValueForType $.memRef.Shape $shapeValueVar }},
		},
	{{- end }}
{{- end }}

{{/* Params: idx:int, parentShape:*Shape, memName:string, memRef:*ShapeRef */}}
{{ define "set event message payload" }}
	{{- $payloadMemName := $.parentShape.PayloadRefName }}
	{{- if HasNonBlobPayloadMembers $.parentShape }}
		Payload: eventstreamtest.MarshalEventPayload(payloadMarshaler, inputEvents[{{ $.idx }}]),
	{{- else if $payloadMemName }}
		{{- $shapeType := (index $.parentShape.MemberRefs $payloadMemName).Shape.Type }}
		{{- if eq $shapeType "blob" }}
			Payload: inputEvents[{{ $.idx }}].({{ $.parentShape.GoType }}).{{ $payloadMemName }},
		{{- else if eq $shapeType "string" }}
			Payload: []byte(*inputEvents[{{ $.idx }}].({{ $.parentShape.GoType }}).{{ $payloadMemName }}),
		{{- end }}
	{{- end }}
{{- end }}
`))
