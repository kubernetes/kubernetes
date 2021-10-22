// +build go1.13,integration,perftest

package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http/httptrace"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

type RequestTrace struct {
	Operation string
	ID        string

	context.Context

	start, finish time.Time

	errs       Errors
	attempts   []RequestAttempt
	curAttempt RequestAttempt
}

func NewRequestTrace(ctx context.Context, op, id string) *RequestTrace {
	rt := &RequestTrace{
		Operation: op,
		ID:        id,
		start:     time.Now(),
		attempts:  []RequestAttempt{},
		curAttempt: RequestAttempt{
			ID: id,
		},
	}

	trace := &httptrace.ClientTrace{
		GetConn:              rt.getConn,
		GotConn:              rt.gotConn,
		PutIdleConn:          rt.putIdleConn,
		GotFirstResponseByte: rt.gotFirstResponseByte,
		Got100Continue:       rt.got100Continue,
		DNSStart:             rt.dnsStart,
		DNSDone:              rt.dnsDone,
		ConnectStart:         rt.connectStart,
		ConnectDone:          rt.connectDone,
		TLSHandshakeStart:    rt.tlsHandshakeStart,
		TLSHandshakeDone:     rt.tlsHandshakeDone,
		WroteHeaders:         rt.wroteHeaders,
		Wait100Continue:      rt.wait100Continue,
		WroteRequest:         rt.wroteRequest,
	}

	rt.Context = httptrace.WithClientTrace(ctx, trace)

	return rt
}

func (rt *RequestTrace) AppendError(err error) {
	rt.errs = append(rt.errs, err)
}
func (rt *RequestTrace) OnSendAttempt(r *request.Request) {
	rt.curAttempt.SendStart = time.Now()
}
func (rt *RequestTrace) OnCompleteAttempt(r *request.Request) {
	rt.curAttempt.Start = r.AttemptTime
	rt.curAttempt.Finish = time.Now()
	rt.curAttempt.Err = r.Error

	if r.Error != nil {
		rt.AppendError(r.Error)
	}

	rt.attempts = append(rt.attempts, rt.curAttempt)
	rt.curAttempt = RequestAttempt{
		ID:         rt.curAttempt.ID,
		AttemptNum: rt.curAttempt.AttemptNum + 1,
	}
}
func (rt *RequestTrace) OnComplete(r *request.Request) {
	rt.finish = time.Now()
	// Last attempt includes reading the response body
	if len(rt.attempts) > 0 {
		rt.attempts[len(rt.attempts)-1].Finish = rt.finish
	}
}

func (rt *RequestTrace) Err() error {
	if len(rt.errs) != 0 {
		return rt.errs
	}
	return nil
}
func (rt *RequestTrace) TotalLatency() time.Duration {
	return rt.finish.Sub(rt.start)
}
func (rt *RequestTrace) Attempts() []RequestAttempt {
	return rt.attempts
}
func (rt *RequestTrace) Retries() int {
	return len(rt.attempts) - 1
}

func (rt *RequestTrace) getConn(hostPort string) {}
func (rt *RequestTrace) gotConn(info httptrace.GotConnInfo) {
	rt.curAttempt.Reused = info.Reused
}
func (rt *RequestTrace) putIdleConn(err error) {}
func (rt *RequestTrace) gotFirstResponseByte() {
	rt.curAttempt.FirstResponseByte = time.Now()
}
func (rt *RequestTrace) got100Continue() {}
func (rt *RequestTrace) dnsStart(info httptrace.DNSStartInfo) {
	rt.curAttempt.DNSStart = time.Now()
}
func (rt *RequestTrace) dnsDone(info httptrace.DNSDoneInfo) {
	rt.curAttempt.DNSDone = time.Now()
}
func (rt *RequestTrace) connectStart(network, addr string) {
	rt.curAttempt.ConnectStart = time.Now()
}
func (rt *RequestTrace) connectDone(network, addr string, err error) {
	rt.curAttempt.ConnectDone = time.Now()
}
func (rt *RequestTrace) tlsHandshakeStart() {
	rt.curAttempt.TLSHandshakeStart = time.Now()
}
func (rt *RequestTrace) tlsHandshakeDone(state tls.ConnectionState, err error) {
	rt.curAttempt.TLSHandshakeDone = time.Now()
}
func (rt *RequestTrace) wroteHeaders() {
	rt.curAttempt.WroteHeaders = time.Now()
}
func (rt *RequestTrace) wait100Continue() {
	rt.curAttempt.Read100Continue = time.Now()
}
func (rt *RequestTrace) wroteRequest(info httptrace.WroteRequestInfo) {
	rt.curAttempt.RequestWritten = time.Now()
}

type RequestAttempt struct {
	Start, Finish time.Time
	SendStart     time.Time
	Err           error

	Reused     bool
	ID         string
	AttemptNum int

	DNSStart, DNSDone                   time.Time
	ConnectStart, ConnectDone           time.Time
	TLSHandshakeStart, TLSHandshakeDone time.Time
	WroteHeaders                        time.Time
	RequestWritten                      time.Time
	Read100Continue                     time.Time
	FirstResponseByte                   time.Time
}

func (a RequestAttempt) String() string {
	const sep = ", "

	var w strings.Builder
	w.WriteString(a.ID + "-" + strconv.Itoa(a.AttemptNum) + sep)
	w.WriteString("Latency:" + durToMSString(a.Finish.Sub(a.Start)) + sep)
	w.WriteString("SDKPreSend:" + durToMSString(a.SendStart.Sub(a.Start)) + sep)

	writeStart := a.SendStart
	fmt.Fprintf(&w, "ConnReused:%t"+sep, a.Reused)
	if !a.Reused {
		w.WriteString("DNS:" + durToMSString(a.DNSDone.Sub(a.DNSStart)) + sep)
		w.WriteString("Connect:" + durToMSString(a.ConnectDone.Sub(a.ConnectStart)) + sep)
		w.WriteString("TLS:" + durToMSString(a.TLSHandshakeDone.Sub(a.TLSHandshakeStart)) + sep)
		writeStart = a.TLSHandshakeDone
	}

	writeHeader := a.WroteHeaders.Sub(writeStart)
	w.WriteString("WriteHeader:" + durToMSString(writeHeader) + sep)
	if !a.Read100Continue.IsZero() {
		// With 100-continue
		w.WriteString("Read100Cont:" + durToMSString(a.Read100Continue.Sub(a.WroteHeaders)) + sep)
		w.WriteString("WritePayload:" + durToMSString(a.FirstResponseByte.Sub(a.RequestWritten)) + sep)

		w.WriteString("RespRead:" + durToMSString(a.Finish.Sub(a.RequestWritten)) + sep)
	} else {
		// No 100-continue
		w.WriteString("WritePayload:" + durToMSString(a.RequestWritten.Sub(a.WroteHeaders)) + sep)

		if !a.FirstResponseByte.IsZero() {
			w.WriteString("RespFirstByte:" + durToMSString(a.FirstResponseByte.Sub(a.RequestWritten)) + sep)
			w.WriteString("RespRead:" + durToMSString(a.Finish.Sub(a.FirstResponseByte)) + sep)
		}
	}

	return w.String()
}

func durToMSString(v time.Duration) string {
	ms := float64(v) / float64(time.Millisecond)
	return fmt.Sprintf("%0.6f", ms)
}
