package eventstreamapi

import (
	"fmt"
	"io"
	"sync"

	"github.com/aws/aws-sdk-go/aws"
)

// StreamWriter provides concurrent safe writing to an event stream.
type StreamWriter struct {
	eventWriter *EventWriter
	stream      chan eventWriteAsyncReport

	done      chan struct{}
	closeOnce sync.Once
	err       *OnceError

	streamCloser io.Closer
}

// NewStreamWriter returns a StreamWriter for the event writer, and stream
// closer provided.
func NewStreamWriter(eventWriter *EventWriter, streamCloser io.Closer) *StreamWriter {
	w := &StreamWriter{
		eventWriter:  eventWriter,
		streamCloser: streamCloser,
		stream:       make(chan eventWriteAsyncReport),
		done:         make(chan struct{}),
		err:          NewOnceError(),
	}
	go w.writeStream()

	return w
}

// Close terminates the writers ability to write new events to the stream. Any
// future call to Send will fail with an error.
func (w *StreamWriter) Close() error {
	w.closeOnce.Do(w.safeClose)
	return w.Err()
}

func (w *StreamWriter) safeClose() {
	close(w.done)
}

// ErrorSet returns a channel which will be closed
// if an error occurs.
func (w *StreamWriter) ErrorSet() <-chan struct{} {
	return w.err.ErrorSet()
}

// Err returns any error that occurred while attempting to write an event to the
// stream.
func (w *StreamWriter) Err() error {
	return w.err.Err()
}

// Send writes a single event to the stream returning an error if the write
// failed.
//
// Send may be called concurrently. Events will be written to the stream
// safely.
func (w *StreamWriter) Send(ctx aws.Context, event Marshaler) error {
	if err := w.Err(); err != nil {
		return err
	}

	resultCh := make(chan error)
	wrapped := eventWriteAsyncReport{
		Event:  event,
		Result: resultCh,
	}

	select {
	case w.stream <- wrapped:
	case <-ctx.Done():
		return ctx.Err()
	case <-w.done:
		return fmt.Errorf("stream closed, unable to send event")
	}

	select {
	case err := <-resultCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-w.done:
		return fmt.Errorf("stream closed, unable to send event")
	}
}

func (w *StreamWriter) writeStream() {
	defer w.Close()

	for {
		select {
		case wrapper := <-w.stream:
			err := w.eventWriter.WriteEvent(wrapper.Event)
			wrapper.ReportResult(w.done, err)
			if err != nil {
				w.err.SetError(err)
				return
			}

		case <-w.done:
			if err := w.streamCloser.Close(); err != nil {
				w.err.SetError(err)
			}
			return
		}
	}
}

type eventWriteAsyncReport struct {
	Event  Marshaler
	Result chan<- error
}

func (e eventWriteAsyncReport) ReportResult(cancel <-chan struct{}, err error) bool {
	select {
	case e.Result <- err:
		return true
	case <-cancel:
		return false
	}
}
