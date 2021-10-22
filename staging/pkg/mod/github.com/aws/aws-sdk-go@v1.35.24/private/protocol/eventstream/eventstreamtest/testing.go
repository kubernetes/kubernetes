// +build go1.10

package eventstreamtest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
	"golang.org/x/net/http2"
)

const (
	errClientDisconnected = "client disconnected"
	errStreamClosed       = "http2: stream closed"
)

// ServeEventStream provides serving EventStream messages from a HTTP server to
// the client. The events are sent sequentially to the client without delay.
type ServeEventStream struct {
	T             *testing.T
	BiDirectional bool

	Events       []eventstream.Message
	ClientEvents []eventstream.Message

	ForceCloseAfter time.Duration

	requestsIdx int
}

func (s ServeEventStream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.(http.Flusher).Flush()

	if s.BiDirectional {
		s.serveBiDirectionalStream(w, r)
	} else {
		s.serveReadOnlyStream(w, r)
	}
}

func (s *ServeEventStream) serveReadOnlyStream(w http.ResponseWriter, r *http.Request) {
	encoder := eventstream.NewEncoder(flushWriter{w})

	for _, event := range s.Events {
		encoder.Encode(event)
	}
}

func (s *ServeEventStream) serveBiDirectionalStream(w http.ResponseWriter, r *http.Request) {
	var wg sync.WaitGroup

	ctx := context.Background()
	if s.ForceCloseAfter > 0 {
		var cancelFunc func()
		ctx, cancelFunc = context.WithTimeout(context.Background(), s.ForceCloseAfter)
		defer cancelFunc()
	}

	var (
		err error
		m   sync.Mutex
	)

	wg.Add(1)
	go func() {
		defer wg.Done()
		readErr := s.readEvents(ctx, r)
		if readErr != nil {
			m.Lock()
			if err == nil {
				err = readErr
			}
			m.Unlock()
		}
	}()

	writeErr := s.writeEvents(ctx, w)
	if writeErr != nil {
		m.Lock()
		if err != nil {
			err = writeErr
		}
		m.Unlock()
	}
	wg.Wait()

	if err != nil && isError(err) {
		s.T.Error(err.Error())
	}
}

func isError(err error) bool {
	switch err.(type) {
	case http2.StreamError:
		return false
	}

	for _, s := range []string{errClientDisconnected, errStreamClosed} {
		if strings.Contains(err.Error(), s) {
			return false
		}
	}

	return true
}

func (s ServeEventStream) readEvents(ctx context.Context, r *http.Request) error {
	signBuffer := make([]byte, 1024)
	messageBuffer := make([]byte, 1024)
	decoder := eventstream.NewDecoder(r.Body)

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}
		// unwrap signing envelope
		signedMessage, err := decoder.Decode(signBuffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		// empty payload is expected for the last signing message
		if len(signedMessage.Payload) == 0 {
			break
		}

		// get service event message from payload
		msg, err := eventstream.Decode(bytes.NewReader(signedMessage.Payload), messageBuffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if len(s.ClientEvents) > 0 {
			i := s.requestsIdx
			s.requestsIdx++

			if e, a := s.ClientEvents[i], msg; !reflect.DeepEqual(e, a) {
				return fmt.Errorf("expected %v, got %v", e, a)
			}
		}
	}

	return nil
}

func (s *ServeEventStream) writeEvents(ctx context.Context, w http.ResponseWriter) error {
	encoder := eventstream.NewEncoder(flushWriter{w})

	var event eventstream.Message
	pendingEvents := s.Events

	for len(pendingEvents) > 0 {
		event, pendingEvents = pendingEvents[0], pendingEvents[1:]
		select {
		case <-ctx.Done():
			return nil
		default:
			err := encoder.Encode(event)
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return fmt.Errorf("expected no error encoding event, got %v", err)
			}
		}
	}

	return nil
}

// SetupEventStreamSession creates a HTTP server SDK session for communicating
// with that server to be used for EventStream APIs. If HTTP/2 is enabled the
// server/client will only attempt to use HTTP/2.
func SetupEventStreamSession(
	t *testing.T, handler http.Handler, h2 bool,
) (sess *session.Session, cleanupFn func(), err error) {
	server := httptest.NewUnstartedServer(handler)

	client := setupServer(server, h2)

	cleanupFn = func() {
		server.Close()
	}

	sess, err = session.NewSession(unit.Session.Config, &aws.Config{
		Endpoint:               &server.URL,
		DisableParamValidation: aws.Bool(true),
		HTTPClient:             client,
		//		LogLevel:               aws.LogLevel(aws.LogDebugWithEventStreamBody),
	})
	if err != nil {
		return nil, nil, err
	}

	return sess, cleanupFn, nil
}

type flushWriter struct {
	w io.Writer
}

func (fw flushWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if f, ok := fw.w.(http.Flusher); ok {
		f.Flush()
	}
	return
}

// MarshalEventPayload marshals a SDK API shape into its associated wire
// protocol payload.
func MarshalEventPayload(
	payloadMarshaler protocol.PayloadMarshaler,
	v interface{},
) []byte {
	var w bytes.Buffer
	err := payloadMarshaler.MarshalPayload(&w, v)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal event %T, %v, %v", v, v, err))
	}

	return w.Bytes()
}

// Prevent circular dependencies on eventstreamapi redefine these here.
const (
	messageTypeHeader    = `:message-type` // Identifies type of message.
	eventMessageType     = `event`
	exceptionMessageType = `exception`
)

// EventMessageTypeHeader is an event message type header for specifying an
// event is an message type.
var EventMessageTypeHeader = eventstream.Header{
	Name:  messageTypeHeader,
	Value: eventstream.StringValue(eventMessageType),
}

// EventExceptionTypeHeader is an event exception type header for specifying an
// event is an exception type.
var EventExceptionTypeHeader = eventstream.Header{
	Name:  messageTypeHeader,
	Value: eventstream.StringValue(exceptionMessageType),
}
