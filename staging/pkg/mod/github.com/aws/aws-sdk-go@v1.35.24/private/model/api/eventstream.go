// +build codegen

package api

import (
	"bytes"
	"fmt"
	"text/template"
)

// EventStreamAPI provides details about the event stream async API and
// associated EventStream shapes.
type EventStreamAPI struct {
	API          *API
	Operation    *Operation
	Name         string
	InputStream  *EventStream
	OutputStream *EventStream
	RequireHTTP2 bool

	// The eventstream generated code was generated with an older model that
	// does not scale with bi-directional models. This drives the need to
	// expose the output shape's event stream member as an exported member.
	Legacy bool
}

func (es *EventStreamAPI) StreamInputEventTypeGetterName() string {
	return "eventTypeFor" + es.Name + "InputEvent"
}
func (es *EventStreamAPI) StreamOutputUnmarshalerForEventName() string {
	return "eventTypeFor" + es.Name + "OutputEvent"
}

// EventStream represents a single eventstream group (input/output) and the
// modeled events that are known for the stream.
type EventStream struct {
	Name       string
	Shape      *Shape
	Events     []*Event
	Exceptions []*Event
}

func (es *EventStream) EventGroupName() string {
	return es.Name + "Event"
}

func (es *EventStream) StreamWriterAPIName() string {
	return es.Name + "Writer"
}

func (es *EventStream) StreamWriterImplName() string {
	return "write" + es.Name
}

func (es *EventStream) StreamEventTypeGetterName() string {
	return "eventTypeFor" + es.Name + "Event"
}

func (es *EventStream) StreamReaderAPIName() string {
	return es.Name + "Reader"
}

func (es *EventStream) StreamReaderImplName() string {
	return "read" + es.Name
}
func (es *EventStream) StreamReaderImplConstructorName() string {
	return "newRead" + es.Name
}

func (es *EventStream) StreamUnmarshalerForEventName() string {
	return "unmarshalerFor" + es.Name + "Event"
}

func (es *EventStream) StreamUnknownEventName() string {
	return es.Name + "UnknownEvent"
}

// Event is a single EventStream event that can be sent or received in an
// EventStream.
type Event struct {
	Name    string
	Shape   *Shape
	For     *EventStream
	Private bool
}

// ShapeDoc returns the docstring for the EventStream API.
func (esAPI *EventStreamAPI) ShapeDoc() string {
	tmpl := template.Must(template.New("eventStreamShapeDoc").Parse(`
{{- $.Name }} provides handling of EventStreams for
the {{ $.Operation.ExportedName }} API.

{{- if $.OutputStream }}

Use this type to receive {{ $.OutputStream.Name }} events. The events
can be read from the stream.

The events that can be received are:
{{- range $_, $event := $.OutputStream.Events }}
    * {{ $event.Shape.ShapeName }}
{{- end }}

{{- end }}

{{- if $.InputStream }}

Use this type to send {{ $.InputStream.Name }} events. The events
can be written to the stream.

The events that can be sent are:
{{ range $_, $event := $.InputStream.Events -}}
    * {{ $event.Shape.ShapeName }}
{{- end }}

{{- end }}`))

	var w bytes.Buffer
	if err := tmpl.Execute(&w, esAPI); err != nil {
		panic(fmt.Sprintf("failed to generate eventstream shape template for %v, %v",
			esAPI.Operation.ExportedName, err))
	}

	return commentify(w.String())
}

func hasEventStream(topShape *Shape) bool {
	for _, ref := range topShape.MemberRefs {
		if ref.Shape.IsEventStream {
			return true
		}
	}

	return false
}

func eventStreamAPIShapeRefDoc(refName string) string {
	return commentify(fmt.Sprintf("Use %s to use the API's stream.", refName))
}

func (a *API) setupEventStreams() error {
	streams := EventStreams{}

	for opName, op := range a.Operations {
		inputRef := getEventStreamMember(op.InputRef.Shape)
		outputRef := getEventStreamMember(op.OutputRef.Shape)

		if inputRef == nil && outputRef == nil {
			continue
		}
		if inputRef != nil && outputRef == nil {
			return fmt.Errorf("event stream input only stream not supported for protocol %s, %s, %v",
				a.NiceName(), opName, a.Metadata.Protocol)
		}
		switch a.Metadata.Protocol {
		case `rest-json`, `rest-xml`, `json`:
		default:
			return UnsupportedAPIModelError{
				Err: fmt.Errorf("EventStream not supported for protocol %s, %s, %v",
					a.NiceName(), opName, a.Metadata.Protocol),
			}
		}

		var inputStream *EventStream
		if inputRef != nil {
			inputStream = streams.GetStream(op.InputRef.Shape, inputRef.Shape)
			inputStream.Shape.IsInputEventStream = true
		}

		var outputStream *EventStream
		if outputRef != nil {
			outputStream = streams.GetStream(op.OutputRef.Shape, outputRef.Shape)
			outputStream.Shape.IsOutputEventStream = true
		}

		requireHTTP2 := op.API.Metadata.ProtocolSettings.HTTP2 == "eventstream" &&
			inputStream != nil && outputStream != nil

		a.HasEventStream = true
		op.EventStreamAPI = &EventStreamAPI{
			API:          a,
			Operation:    op,
			Name:         op.ExportedName + "EventStream",
			InputStream:  inputStream,
			OutputStream: outputStream,
			Legacy:       isLegacyEventStream(op),
			RequireHTTP2: requireHTTP2,
		}
		op.OutputRef.Shape.OutputEventStreamAPI = op.EventStreamAPI

		if s, ok := a.Shapes[op.EventStreamAPI.Name]; ok {
			newName := op.EventStreamAPI.Name + "Data"
			if _, ok := a.Shapes[newName]; ok {
				panic(fmt.Sprintf(
					"%s: attempting to rename %s to %s, but shape with that name already exists",
					a.NiceName(), op.EventStreamAPI.Name, newName))
			}
			s.Rename(newName)
		}
	}

	return nil
}

// EventStreams is a map of streams for the API shared across all operations.
// Ensurs that no stream is duplicated.
type EventStreams map[*Shape]*EventStream

// GetStream returns an EventStream for the operations top level shape, and
// member reference to the stream shape.
func (es *EventStreams) GetStream(topShape *Shape, streamShape *Shape) *EventStream {
	var stream *EventStream
	if v, ok := (*es)[streamShape]; ok {
		stream = v
	} else {
		stream = setupEventStream(streamShape)
		(*es)[streamShape] = stream
	}

	if topShape.API.Metadata.Protocol == "json" {
		topShape.EventFor = append(topShape.EventFor, stream)
	}

	return stream
}

var legacyEventStream = map[string]map[string]struct{}{
	"s3": {
		"SelectObjectContent": struct{}{},
	},
	"kinesis": {
		"SubscribeToShard": struct{}{},
	},
}

func isLegacyEventStream(op *Operation) bool {
	if s, ok := legacyEventStream[op.API.PackageName()]; ok {
		if _, ok = s[op.ExportedName]; ok {
			return true
		}
	}
	return false
}

func (e EventStreamAPI) OutputMemberName() string {
	if e.Legacy {
		return "EventStream"
	}

	return "eventStream"
}

func getEventStreamMember(topShape *Shape) *ShapeRef {
	for _, ref := range topShape.MemberRefs {
		if !ref.Shape.IsEventStream {
			continue
		}
		return ref
	}

	return nil
}

func setupEventStream(s *Shape) *EventStream {
	eventStream := &EventStream{
		Name:  s.ShapeName,
		Shape: s,
	}
	s.EventStream = eventStream

	for _, eventRefName := range s.MemberNames() {
		eventRef := s.MemberRefs[eventRefName]
		if !(eventRef.Shape.IsEvent || eventRef.Shape.Exception) {
			panic(fmt.Sprintf("unexpected non-event member reference %s.%s",
				s.ShapeName, eventRefName))
		}

		updateEventPayloadRef(eventRef.Shape)

		eventRef.Shape.EventFor = append(eventRef.Shape.EventFor, eventStream)

		// Exceptions and events are two different lists to allow the SDK
		// to easily generate code with the two handled differently.
		event := &Event{
			Name:  eventRefName,
			Shape: eventRef.Shape,
			For:   eventStream,
		}
		if eventRef.Shape.Exception {
			eventStream.Exceptions = append(eventStream.Exceptions, event)
		} else {
			eventStream.Events = append(eventStream.Events, event)
		}
	}

	return eventStream
}

func updateEventPayloadRef(parent *Shape) {
	refName := parent.PayloadRefName()
	if len(refName) == 0 {
		return
	}

	payloadRef := parent.MemberRefs[refName]
	if payloadRef.Shape.Type == "blob" {
		return
	}

	if len(payloadRef.LocationName) != 0 {
		return
	}

	payloadRef.LocationName = refName
}
