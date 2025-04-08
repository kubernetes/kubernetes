// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"context"
	"time"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors"
	"github.com/prometheus/client_golang/prometheus"
)

type reporter struct {
	clientMetrics   *ClientMetrics
	serverMetrics   *ServerMetrics
	typ             interceptors.GRPCType
	service, method string
	kind            Kind
	exemplar        prometheus.Labels
}

func (r *reporter) PostCall(err error, rpcDuration time.Duration) {
	// get status code from error
	status := FromError(err)
	code := status.Code()

	// perform handling of metrics from code
	switch r.kind {
	case KindServer:
		r.incrementWithExemplar(r.serverMetrics.serverHandledCounter, string(r.typ), r.service, r.method, code.String())
		if r.serverMetrics.serverHandledHistogram != nil {
			r.observeWithExemplar(r.serverMetrics.serverHandledHistogram, rpcDuration.Seconds(), string(r.typ), r.service, r.method)
		}

	case KindClient:
		r.incrementWithExemplar(r.clientMetrics.clientHandledCounter, string(r.typ), r.service, r.method, code.String())
		if r.clientMetrics.clientHandledHistogram != nil {
			r.observeWithExemplar(r.clientMetrics.clientHandledHistogram, rpcDuration.Seconds(), string(r.typ), r.service, r.method)
		}
	}
}

func (r *reporter) PostMsgSend(_ any, _ error, sendDuration time.Duration) {
	switch r.kind {
	case KindServer:
		r.incrementWithExemplar(r.serverMetrics.serverStreamMsgSent, string(r.typ), r.service, r.method)
	case KindClient:
		r.incrementWithExemplar(r.clientMetrics.clientStreamMsgSent, string(r.typ), r.service, r.method)
		if r.clientMetrics.clientStreamSendHistogram != nil {
			r.observeWithExemplar(r.clientMetrics.clientStreamSendHistogram, sendDuration.Seconds(), string(r.typ), r.service, r.method)
		}
	}
}

func (r *reporter) PostMsgReceive(_ any, _ error, recvDuration time.Duration) {
	switch r.kind {
	case KindServer:
		r.incrementWithExemplar(r.serverMetrics.serverStreamMsgReceived, string(r.typ), r.service, r.method)
	case KindClient:
		r.incrementWithExemplar(r.clientMetrics.clientStreamMsgReceived, string(r.typ), r.service, r.method)
		if r.clientMetrics.clientStreamRecvHistogram != nil {
			r.observeWithExemplar(r.clientMetrics.clientStreamRecvHistogram, recvDuration.Seconds(), string(r.typ), r.service, r.method)
		}
	}
}

type reportable struct {
	clientMetrics *ClientMetrics
	serverMetrics *ServerMetrics

	opts []Option
}

func (rep *reportable) ServerReporter(ctx context.Context, meta interceptors.CallMeta) (interceptors.Reporter, context.Context) {
	return rep.reporter(ctx, rep.serverMetrics, nil, meta, KindServer)
}

func (rep *reportable) ClientReporter(ctx context.Context, meta interceptors.CallMeta) (interceptors.Reporter, context.Context) {
	return rep.reporter(ctx, nil, rep.clientMetrics, meta, KindClient)
}

func (rep *reportable) reporter(ctx context.Context, sm *ServerMetrics, cm *ClientMetrics, meta interceptors.CallMeta, kind Kind) (interceptors.Reporter, context.Context) {
	var c config
	c.apply(rep.opts)
	r := &reporter{
		clientMetrics: cm,
		serverMetrics: sm,
		typ:           meta.Typ,
		service:       meta.Service,
		method:        meta.Method,
		kind:          kind,
	}
	if c.exemplarFn != nil {
		r.exemplar = c.exemplarFn(ctx)
	}

	switch kind {
	case KindClient:
		r.incrementWithExemplar(r.clientMetrics.clientStartedCounter, string(r.typ), r.service, r.method)
	case KindServer:
		r.incrementWithExemplar(r.serverMetrics.serverStartedCounter, string(r.typ), r.service, r.method)
	}
	return r, ctx
}

func (r *reporter) incrementWithExemplar(c *prometheus.CounterVec, lvals ...string) {
	c.WithLabelValues(lvals...).(prometheus.ExemplarAdder).AddWithExemplar(1, r.exemplar)
}

func (r *reporter) observeWithExemplar(h *prometheus.HistogramVec, value float64, lvals ...string) {
	h.WithLabelValues(lvals...).(prometheus.ExemplarObserver).ObserveWithExemplar(value, r.exemplar)
}
