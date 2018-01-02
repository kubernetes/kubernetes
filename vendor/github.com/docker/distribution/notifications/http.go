package notifications

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// httpSink implements a single-flight, http notification endpoint. This is
// very lightweight in that it only makes an attempt at an http request.
// Reliability should be provided by the caller.
type httpSink struct {
	url string

	mu        sync.Mutex
	closed    bool
	client    *http.Client
	listeners []httpStatusListener

	// TODO(stevvooe): Allow one to configure the media type accepted by this
	// sink and choose the serialization based on that.
}

// newHTTPSink returns an unreliable, single-flight http sink. Wrap in other
// sinks for increased reliability.
func newHTTPSink(u string, timeout time.Duration, headers http.Header, transport *http.Transport, listeners ...httpStatusListener) *httpSink {
	if transport == nil {
		transport = http.DefaultTransport.(*http.Transport)
	}
	return &httpSink{
		url:       u,
		listeners: listeners,
		client: &http.Client{
			Transport: &headerRoundTripper{
				Transport: transport,
				headers:   headers,
			},
			Timeout: timeout,
		},
	}
}

// httpStatusListener is called on various outcomes of sending notifications.
type httpStatusListener interface {
	success(status int, events ...Event)
	failure(status int, events ...Event)
	err(err error, events ...Event)
}

// Accept makes an attempt to notify the endpoint, returning an error if it
// fails. It is the caller's responsibility to retry on error. The events are
// accepted or rejected as a group.
func (hs *httpSink) Write(events ...Event) error {
	hs.mu.Lock()
	defer hs.mu.Unlock()
	defer hs.client.Transport.(*headerRoundTripper).CloseIdleConnections()

	if hs.closed {
		return ErrSinkClosed
	}

	envelope := Envelope{
		Events: events,
	}

	// TODO(stevvooe): It is not ideal to keep re-encoding the request body on
	// retry but we are going to do it to keep the code simple. It is likely
	// we could change the event struct to manage its own buffer.

	p, err := json.MarshalIndent(envelope, "", "   ")
	if err != nil {
		for _, listener := range hs.listeners {
			listener.err(err, events...)
		}
		return fmt.Errorf("%v: error marshaling event envelope: %v", hs, err)
	}

	body := bytes.NewReader(p)
	resp, err := hs.client.Post(hs.url, EventsMediaType, body)
	if err != nil {
		for _, listener := range hs.listeners {
			listener.err(err, events...)
		}

		return fmt.Errorf("%v: error posting: %v", hs, err)
	}
	defer resp.Body.Close()

	// The notifier will treat any 2xx or 3xx response as accepted by the
	// endpoint.
	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 400:
		for _, listener := range hs.listeners {
			listener.success(resp.StatusCode, events...)
		}

		// TODO(stevvooe): This is a little accepting: we may want to support
		// unsupported media type responses with retries using the correct
		// media type. There may also be cases that will never work.

		return nil
	default:
		for _, listener := range hs.listeners {
			listener.failure(resp.StatusCode, events...)
		}
		return fmt.Errorf("%v: response status %v unaccepted", hs, resp.Status)
	}
}

// Close the endpoint
func (hs *httpSink) Close() error {
	hs.mu.Lock()
	defer hs.mu.Unlock()

	if hs.closed {
		return fmt.Errorf("httpsink: already closed")
	}

	hs.closed = true
	return nil
}

func (hs *httpSink) String() string {
	return fmt.Sprintf("httpSink{%s}", hs.url)
}

type headerRoundTripper struct {
	*http.Transport // must be transport to support CancelRequest
	headers         http.Header
}

func (hrt *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	var nreq http.Request
	nreq = *req
	nreq.Header = make(http.Header)

	merge := func(headers http.Header) {
		for k, v := range headers {
			nreq.Header[k] = append(nreq.Header[k], v...)
		}
	}

	merge(req.Header)
	merge(hrt.headers)

	return hrt.Transport.RoundTrip(&nreq)
}
