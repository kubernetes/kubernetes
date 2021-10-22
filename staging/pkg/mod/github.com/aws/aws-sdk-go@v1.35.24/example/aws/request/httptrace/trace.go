package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net/http/httptrace"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

// RequestLatency provides latencies for the SDK API request and its attempts.
type RequestLatency struct {
	Latency  time.Duration
	Validate time.Duration
	Build    time.Duration

	Attempts []RequestAttemptLatency
}

// RequestAttemptLatency provides latencies for an individual request attempt.
type RequestAttemptLatency struct {
	Latency time.Duration
	Err     error

	Sign time.Duration
	Send time.Duration

	HTTP HTTPLatency

	Unmarshal      time.Duration
	UnmarshalError time.Duration

	WillRetry bool
	Retry     time.Duration
}

// HTTPLatency provides latencies for an HTTP request.
type HTTPLatency struct {
	Latency    time.Duration
	ConnReused bool

	GetConn time.Duration

	DNS     time.Duration
	Connect time.Duration
	TLS     time.Duration

	WriteHeader           time.Duration
	WriteRequest          time.Duration
	WaitResponseFirstByte time.Duration
	ReadHeader            time.Duration
	ReadBody              time.Duration
}

// RequestTrace provides the structure to store SDK request attempt latencies.
// Use TraceRequest as a API operation request option to capture trace metrics
// for the individual request.
type RequestTrace struct {
	Start, Finish time.Time

	ValidateStart, ValidateDone time.Time
	BuildStart, BuildDone       time.Time

	ReadResponseBody bool

	Attempts []*RequestAttemptTrace
}

// Latency returns the latencies of the request trace components.
func (t RequestTrace) Latency() RequestLatency {
	var attempts []RequestAttemptLatency
	for _, a := range t.Attempts {
		attempts = append(attempts, a.Latency())
	}

	latency := RequestLatency{
		Latency:  safeTimeDelta(t.Start, t.Finish),
		Validate: safeTimeDelta(t.ValidateStart, t.ValidateDone),
		Build:    safeTimeDelta(t.BuildStart, t.BuildDone),
		Attempts: attempts,
	}

	return latency
}

// TraceRequest is a SDK request Option that will add request handlers to an
// individual request to track request latencies per attempt. Must be used only
// for a single request call per RequestTrace value.
func (t *RequestTrace) TraceRequest(r *request.Request) {
	t.Start = time.Now()
	r.Handlers.Complete.PushBack(t.onComplete)

	r.Handlers.Validate.PushFront(t.onValidateStart)
	r.Handlers.Validate.PushBack(t.onValidateDone)

	r.Handlers.Build.PushFront(t.onBuildStart)
	r.Handlers.Build.PushBack(t.onBuildDone)

	var attempt *RequestAttemptTrace

	// Signing and Start attempt
	r.Handlers.Sign.PushFront(func(rr *request.Request) {
		attempt = &RequestAttemptTrace{Start: time.Now()}
		attempt.SignStart = attempt.Start
	})
	r.Handlers.Sign.PushBack(func(rr *request.Request) {
		attempt.SignDone = time.Now()
	})

	// Ensure that the http trace added to the request always uses the original
	// context instead of each following attempt's context to prevent conflict
	// with previous http traces used.
	origContext := r.Context()

	// Send
	r.Handlers.Send.PushFront(func(rr *request.Request) {
		attempt.SendStart = time.Now()
		attempt.HTTPTrace = NewHTTPTrace(origContext)
		rr.SetContext(attempt.HTTPTrace)
	})
	r.Handlers.Send.PushBack(func(rr *request.Request) {
		attempt.SendDone = time.Now()
		defer func() {
			attempt.HTTPTrace.Finish = time.Now()
		}()

		if rr.Error != nil {
			return
		}

		attempt.HTTPTrace.ReadHeaderDone = time.Now()
		if t.ReadResponseBody {
			attempt.HTTPTrace.ReadBodyStart = time.Now()
			var w bytes.Buffer
			if _, err := io.Copy(&w, rr.HTTPResponse.Body); err != nil {
				rr.Error = err
				return
			}
			rr.HTTPResponse.Body.Close()
			rr.HTTPResponse.Body = ioutil.NopCloser(&w)

			attempt.HTTPTrace.ReadBodyDone = time.Now()
		}
	})

	// Unmarshal
	r.Handlers.Unmarshal.PushFront(func(rr *request.Request) {
		attempt.UnmarshalStart = time.Now()
	})
	r.Handlers.Unmarshal.PushBack(func(rr *request.Request) {
		attempt.UnmarshalDone = time.Now()
	})

	// Unmarshal Error
	r.Handlers.UnmarshalError.PushFront(func(rr *request.Request) {
		attempt.UnmarshalErrorStart = time.Now()
	})
	r.Handlers.UnmarshalError.PushBack(func(rr *request.Request) {
		attempt.UnmarshalErrorDone = time.Now()
	})

	// Retry handling and delay
	r.Handlers.Retry.PushFront(func(rr *request.Request) {
		attempt.RetryStart = time.Now()
		attempt.Err = rr.Error
	})
	r.Handlers.AfterRetry.PushBack(func(rr *request.Request) {
		attempt.RetryDone = time.Now()
		attempt.WillRetry = rr.WillRetry()
	})

	// Complete Attempt
	r.Handlers.CompleteAttempt.PushBack(func(rr *request.Request) {
		attempt.Finish = time.Now()
		t.Attempts = append(t.Attempts, attempt)
	})
}

func (t *RequestTrace) String() string {
	var w strings.Builder

	l := t.Latency()
	writeDurField(&w, "Latency", l.Latency)
	writeDurField(&w, "Validate", l.Validate)
	writeDurField(&w, "Build", l.Build)
	writeField(&w, "Attempts", "%d", len(t.Attempts))

	for i, a := range t.Attempts {
		fmt.Fprintf(&w, "\n\tAttempt: %d, %s", i, a)
	}

	return w.String()
}

func (t *RequestTrace) onComplete(*request.Request) {
	t.Finish = time.Now()
}
func (t *RequestTrace) onValidateStart(*request.Request) { t.ValidateStart = time.Now() }
func (t *RequestTrace) onValidateDone(*request.Request)  { t.ValidateDone = time.Now() }
func (t *RequestTrace) onBuildStart(*request.Request)    { t.BuildStart = time.Now() }
func (t *RequestTrace) onBuildDone(*request.Request)     { t.BuildDone = time.Now() }

// RequestAttemptTrace provides a structure for storing trace information on
// the SDK's request attempt.
type RequestAttemptTrace struct {
	Start, Finish time.Time
	Err           error

	SignStart, SignDone time.Time

	SendStart, SendDone time.Time
	HTTPTrace           *HTTPTrace

	UnmarshalStart, UnmarshalDone           time.Time
	UnmarshalErrorStart, UnmarshalErrorDone time.Time

	WillRetry             bool
	RetryStart, RetryDone time.Time
}

// Latency returns the latencies of the request attempt trace components.
func (t *RequestAttemptTrace) Latency() RequestAttemptLatency {
	return RequestAttemptLatency{
		Latency: safeTimeDelta(t.Start, t.Finish),
		Err:     t.Err,

		Sign: safeTimeDelta(t.SignStart, t.SignDone),
		Send: safeTimeDelta(t.SendStart, t.SendDone),

		HTTP: t.HTTPTrace.Latency(),

		Unmarshal:      safeTimeDelta(t.UnmarshalStart, t.UnmarshalDone),
		UnmarshalError: safeTimeDelta(t.UnmarshalErrorStart, t.UnmarshalErrorDone),

		WillRetry: t.WillRetry,
		Retry:     safeTimeDelta(t.RetryStart, t.RetryDone),
	}
}

func (t *RequestAttemptTrace) String() string {
	var w strings.Builder

	l := t.Latency()
	writeDurField(&w, "Latency", l.Latency)
	writeDurField(&w, "Sign", l.Sign)
	writeDurField(&w, "Send", l.Send)

	writeDurField(&w, "Unmarshal", l.Unmarshal)
	writeDurField(&w, "UnmarshalError", l.UnmarshalError)

	writeField(&w, "WillRetry", "%t", l.WillRetry)
	writeDurField(&w, "Retry", l.Retry)

	fmt.Fprintf(&w, "\n\t\tHTTP: %s", t.HTTPTrace)
	if t.Err != nil {
		fmt.Fprintf(&w, "\n\t\tError: %v", t.Err)
	}

	return w.String()
}

// HTTPTrace provides the trace time stamps of the HTTP request's segments.
type HTTPTrace struct {
	context.Context

	Start, Finish time.Time

	GetConnStart, GetConnDone time.Time
	Reused                    bool

	DNSStart, DNSDone                   time.Time
	ConnectStart, ConnectDone           time.Time
	TLSHandshakeStart, TLSHandshakeDone time.Time
	WriteHeaderDone                     time.Time
	WriteRequestDone                    time.Time
	FirstResponseByte                   time.Time

	ReadHeaderStart, ReadHeaderDone time.Time
	ReadBodyStart, ReadBodyDone     time.Time
}

// NewHTTPTrace returns a initialized HTTPTrace for an
// httptrace.ClientTrace, based on the context passed.
func NewHTTPTrace(ctx context.Context) *HTTPTrace {
	t := &HTTPTrace{
		Start: time.Now(),
	}

	trace := &httptrace.ClientTrace{
		GetConn:              t.getConn,
		GotConn:              t.gotConn,
		PutIdleConn:          t.putIdleConn,
		GotFirstResponseByte: t.gotFirstResponseByte,
		Got100Continue:       t.got100Continue,
		DNSStart:             t.dnsStart,
		DNSDone:              t.dnsDone,
		ConnectStart:         t.connectStart,
		ConnectDone:          t.connectDone,
		TLSHandshakeStart:    t.tlsHandshakeStart,
		TLSHandshakeDone:     t.tlsHandshakeDone,
		WroteHeaders:         t.wroteHeaders,
		Wait100Continue:      t.wait100Continue,
		WroteRequest:         t.wroteRequest,
	}

	t.Context = httptrace.WithClientTrace(ctx, trace)

	return t
}

// Latency returns the latencies for an HTTP request.
func (t *HTTPTrace) Latency() HTTPLatency {
	latency := HTTPLatency{
		Latency:    safeTimeDelta(t.Start, t.Finish),
		ConnReused: t.Reused,

		WriteHeader:           safeTimeDelta(t.GetConnDone, t.WriteHeaderDone),
		WriteRequest:          safeTimeDelta(t.GetConnDone, t.WriteRequestDone),
		WaitResponseFirstByte: safeTimeDelta(t.WriteRequestDone, t.FirstResponseByte),
		ReadHeader:            safeTimeDelta(t.ReadHeaderStart, t.ReadHeaderDone),
		ReadBody:              safeTimeDelta(t.ReadBodyStart, t.ReadBodyDone),
	}

	if !t.Reused {
		latency.GetConn = safeTimeDelta(t.GetConnStart, t.GetConnDone)
		latency.DNS = safeTimeDelta(t.DNSStart, t.DNSDone)
		latency.Connect = safeTimeDelta(t.ConnectStart, t.ConnectDone)
		latency.TLS = safeTimeDelta(t.TLSHandshakeStart, t.TLSHandshakeDone)
	} else {
		latency.GetConn = safeTimeDelta(t.Start, t.GetConnDone)
	}

	return latency
}

func (t *HTTPTrace) String() string {
	var w strings.Builder

	l := t.Latency()
	writeDurField(&w, "Latency", l.Latency)
	writeField(&w, "ConnReused", "%t", l.ConnReused)
	writeDurField(&w, "GetConn", l.GetConn)

	writeDurField(&w, "WriteHeader", l.WriteHeader)
	writeDurField(&w, "WriteRequest", l.WriteRequest)
	writeDurField(&w, "WaitResponseFirstByte", l.WaitResponseFirstByte)
	writeDurField(&w, "ReadHeader", l.ReadHeader)
	writeDurField(&w, "ReadBody", l.ReadBody)

	if !t.Reused {
		fmt.Fprintf(&w, "\n\t\t\tConn: ")
		writeDurField(&w, "DNS", l.DNS)
		writeDurField(&w, "Connect", l.Connect)
		writeDurField(&w, "TLS", l.TLS)
	}

	return w.String()
}

func (t *HTTPTrace) getConn(hostPort string) {
	t.GetConnStart = time.Now()
}
func (t *HTTPTrace) gotConn(info httptrace.GotConnInfo) {
	t.GetConnDone = time.Now()
	t.Reused = info.Reused
}
func (t *HTTPTrace) putIdleConn(err error) {}
func (t *HTTPTrace) gotFirstResponseByte() {
	t.FirstResponseByte = time.Now()
	t.ReadHeaderStart = t.FirstResponseByte
}
func (t *HTTPTrace) got100Continue() {}
func (t *HTTPTrace) dnsStart(info httptrace.DNSStartInfo) {
	t.DNSStart = time.Now()
}
func (t *HTTPTrace) dnsDone(info httptrace.DNSDoneInfo) {
	t.DNSDone = time.Now()
}
func (t *HTTPTrace) connectStart(network, addr string) {
	t.ConnectStart = time.Now()
}
func (t *HTTPTrace) connectDone(network, addr string, err error) {
	t.ConnectDone = time.Now()
}
func (t *HTTPTrace) tlsHandshakeStart() {
	t.TLSHandshakeStart = time.Now()
}
func (t *HTTPTrace) tlsHandshakeDone(state tls.ConnectionState, err error) {
	t.TLSHandshakeDone = time.Now()
}
func (t *HTTPTrace) wroteHeaders() {
	t.WriteHeaderDone = time.Now()
}
func (t *HTTPTrace) wait100Continue() {}
func (t *HTTPTrace) wroteRequest(info httptrace.WroteRequestInfo) {
	t.WriteRequestDone = time.Now()
}

func safeTimeDelta(start, end time.Time) time.Duration {
	if start.IsZero() || end.IsZero() || start.After(end) {
		return 0
	}

	return end.Sub(start)
}

func writeField(w io.Writer, field string, format string, args ...interface{}) error {
	_, err := fmt.Fprintf(w, "%s: "+format+", ", append([]interface{}{field}, args...)...)
	return err
}

func writeDurField(w io.Writer, field string, dur time.Duration) error {
	if dur == 0 {
		return nil
	}

	_, err := fmt.Fprintf(w, "%s: %s, ", field, dur)
	return err
}
