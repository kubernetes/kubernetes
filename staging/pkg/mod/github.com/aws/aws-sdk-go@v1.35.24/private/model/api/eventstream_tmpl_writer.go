// +build codegen

package api

import "text/template"

var eventStreamShapeWriterTmpl = template.Must(template.New("eventStreamShapeWriterTmpl").
	Funcs(template.FuncMap{}).
	Parse(`
{{- $es := $.EventStream }}

// {{ $es.StreamWriterAPIName }} provides the interface for writing events to the stream.
// The default implementation for this interface will be {{ $.ShapeName }}.
//
// The writer's Close method must allow multiple concurrent calls.
//
// These events are:
// {{ range $_, $event := $es.Events }}
//     * {{ $event.Shape.ShapeName }}
{{- end }}
type {{ $es.StreamWriterAPIName }} interface {
	// Sends writes events to the stream blocking until the event has been
	// written. An error is returned if the write fails.
	Send(aws.Context, {{ $es.EventGroupName }}) error

	// Close will stop the writer writing to the event stream.
	Close() error

	// Returns any error that has occurred while writing to the event stream.
	Err() error
}

type {{ $es.StreamWriterImplName }} struct {
	*eventstreamapi.StreamWriter
}

func (w *{{ $es.StreamWriterImplName }}) Send(ctx aws.Context, event {{ $es.EventGroupName }}) error {
	return w.StreamWriter.Send(ctx, event)
}

func {{ $es.StreamEventTypeGetterName }}(event eventstreamapi.Marshaler) (string, error) {
	switch event.(type) {
		{{- range $_, $event := $es.Events }}
			case *{{ $event.Shape.ShapeName }}:
				return {{ printf "%q" $event.Name }}, nil
		{{- end }}
		{{- range $_, $event := $es.Exceptions }}
			case *{{ $event.Shape.ShapeName }}:
				return {{ printf "%q" $event.Name }}, nil
		{{- end }}
	default:
		return "", awserr.New(
			request.ErrCodeSerialization,
			fmt.Sprintf("unknown event type, %T, for {{ $es.Name }}", event),
			nil,
		)
	}
}
`))
