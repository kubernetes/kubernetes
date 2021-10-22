// +build integration,perftest

package main

import (
	"context"
	"crypto/tls"
	"net/http/httptrace"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

type RequestTrace struct {
	ID int64
	context.Context

	start, finish time.Time

	errs     Errors
	attempts []RequestAttempt

	curAttempt RequestAttempt
}

func NewRequestTrace(ctx context.Context, id int64) *RequestTrace {
	rt := &RequestTrace{
		ID:       id,
		start:    time.Now(),
		attempts: []RequestAttempt{},
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
func (rt *RequestTrace) OnCompleteAttempt(r *request.Request) {
	rt.curAttempt.Start = r.AttemptTime
	rt.curAttempt.Finish = time.Now()
	rt.curAttempt.Err = r.Error

	rt.attempts = append(rt.attempts, rt.curAttempt)
	rt.curAttempt = RequestAttempt{}
}
func (rt *RequestTrace) OnSendAttempt(r *request.Request) {
	rt.curAttempt.SendStart = time.Now()
}
func (rt *RequestTrace) OnCompleteRequest(r *request.Request) {}
func (rt *RequestTrace) RequestDone() {
	rt.finish = time.Now()
	// Last attempt includes reading the response body
	rt.attempts[len(rt.attempts)-1].Finish = rt.finish
}

func (rt *RequestTrace) Err() error {
	return rt.errs
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
func (rt *RequestTrace) wroteHeaders()    {}
func (rt *RequestTrace) wait100Continue() {}
func (rt *RequestTrace) wroteRequest(info httptrace.WroteRequestInfo) {
	rt.curAttempt.RequestWritten = time.Now()
}

type RequestAttempt struct {
	Start, Finish time.Time
	SendStart     time.Time
	Err           error

	Reused bool

	DNSStart, DNSDone                   time.Time
	ConnectStart, ConnectDone           time.Time
	TLSHandshakeStart, TLSHandshakeDone time.Time
	RequestWritten                      time.Time
	FirstResponseByte                   time.Time
}
