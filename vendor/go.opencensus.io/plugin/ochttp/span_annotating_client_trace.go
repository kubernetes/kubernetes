// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ochttp

import (
	"crypto/tls"
	"net/http"
	"net/http/httptrace"
	"strings"

	"go.opencensus.io/trace"
)

type spanAnnotator struct {
	sp *trace.Span
}

// TODO: Remove NewSpanAnnotator at the next release.

// NewSpanAnnotator returns a httptrace.ClientTrace which annotates
// all emitted httptrace events on the provided Span.
// Deprecated: Use NewSpanAnnotatingClientTrace instead
func NewSpanAnnotator(r *http.Request, s *trace.Span) *httptrace.ClientTrace {
	return NewSpanAnnotatingClientTrace(r, s)
}

// NewSpanAnnotatingClientTrace returns a httptrace.ClientTrace which annotates
// all emitted httptrace events on the provided Span.
func NewSpanAnnotatingClientTrace(_ *http.Request, s *trace.Span) *httptrace.ClientTrace {
	sa := spanAnnotator{sp: s}

	return &httptrace.ClientTrace{
		GetConn:              sa.getConn,
		GotConn:              sa.gotConn,
		PutIdleConn:          sa.putIdleConn,
		GotFirstResponseByte: sa.gotFirstResponseByte,
		Got100Continue:       sa.got100Continue,
		DNSStart:             sa.dnsStart,
		DNSDone:              sa.dnsDone,
		ConnectStart:         sa.connectStart,
		ConnectDone:          sa.connectDone,
		TLSHandshakeStart:    sa.tlsHandshakeStart,
		TLSHandshakeDone:     sa.tlsHandshakeDone,
		WroteHeaders:         sa.wroteHeaders,
		Wait100Continue:      sa.wait100Continue,
		WroteRequest:         sa.wroteRequest,
	}
}

func (s spanAnnotator) getConn(hostPort string) {
	attrs := []trace.Attribute{
		trace.StringAttribute("httptrace.get_connection.host_port", hostPort),
	}
	s.sp.Annotate(attrs, "GetConn")
}

func (s spanAnnotator) gotConn(info httptrace.GotConnInfo) {
	attrs := []trace.Attribute{
		trace.BoolAttribute("httptrace.got_connection.reused", info.Reused),
		trace.BoolAttribute("httptrace.got_connection.was_idle", info.WasIdle),
	}
	if info.WasIdle {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.got_connection.idle_time", info.IdleTime.String()))
	}
	s.sp.Annotate(attrs, "GotConn")
}

// PutIdleConn implements a httptrace.ClientTrace hook
func (s spanAnnotator) putIdleConn(err error) {
	var attrs []trace.Attribute
	if err != nil {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.put_idle_connection.error", err.Error()))
	}
	s.sp.Annotate(attrs, "PutIdleConn")
}

func (s spanAnnotator) gotFirstResponseByte() {
	s.sp.Annotate(nil, "GotFirstResponseByte")
}

func (s spanAnnotator) got100Continue() {
	s.sp.Annotate(nil, "Got100Continue")
}

func (s spanAnnotator) dnsStart(info httptrace.DNSStartInfo) {
	attrs := []trace.Attribute{
		trace.StringAttribute("httptrace.dns_start.host", info.Host),
	}
	s.sp.Annotate(attrs, "DNSStart")
}

func (s spanAnnotator) dnsDone(info httptrace.DNSDoneInfo) {
	var addrs []string
	for _, addr := range info.Addrs {
		addrs = append(addrs, addr.String())
	}
	attrs := []trace.Attribute{
		trace.StringAttribute("httptrace.dns_done.addrs", strings.Join(addrs, " , ")),
	}
	if info.Err != nil {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.dns_done.error", info.Err.Error()))
	}
	s.sp.Annotate(attrs, "DNSDone")
}

func (s spanAnnotator) connectStart(network, addr string) {
	attrs := []trace.Attribute{
		trace.StringAttribute("httptrace.connect_start.network", network),
		trace.StringAttribute("httptrace.connect_start.addr", addr),
	}
	s.sp.Annotate(attrs, "ConnectStart")
}

func (s spanAnnotator) connectDone(network, addr string, err error) {
	attrs := []trace.Attribute{
		trace.StringAttribute("httptrace.connect_done.network", network),
		trace.StringAttribute("httptrace.connect_done.addr", addr),
	}
	if err != nil {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.connect_done.error", err.Error()))
	}
	s.sp.Annotate(attrs, "ConnectDone")
}

func (s spanAnnotator) tlsHandshakeStart() {
	s.sp.Annotate(nil, "TLSHandshakeStart")
}

func (s spanAnnotator) tlsHandshakeDone(_ tls.ConnectionState, err error) {
	var attrs []trace.Attribute
	if err != nil {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.tls_handshake_done.error", err.Error()))
	}
	s.sp.Annotate(attrs, "TLSHandshakeDone")
}

func (s spanAnnotator) wroteHeaders() {
	s.sp.Annotate(nil, "WroteHeaders")
}

func (s spanAnnotator) wait100Continue() {
	s.sp.Annotate(nil, "Wait100Continue")
}

func (s spanAnnotator) wroteRequest(info httptrace.WroteRequestInfo) {
	var attrs []trace.Attribute
	if info.Err != nil {
		attrs = append(attrs,
			trace.StringAttribute("httptrace.wrote_request.error", info.Err.Error()))
	}
	s.sp.Annotate(attrs, "WroteRequest")
}
