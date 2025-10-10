/*
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package orca

import (
	"context"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/orca/internal"
	"google.golang.org/grpc/status"

	v3orcapb "github.com/cncf/xds/go/xds/data/orca/v3"
	v3orcaservicegrpc "github.com/cncf/xds/go/xds/service/orca/v3"
	v3orcaservicepb "github.com/cncf/xds/go/xds/service/orca/v3"
	"google.golang.org/protobuf/types/known/durationpb"
)

type producerBuilder struct{}

// Build constructs and returns a producer and its cleanup function
func (*producerBuilder) Build(cci any) (balancer.Producer, func()) {
	p := &producer{
		client:    v3orcaservicegrpc.NewOpenRcaServiceClient(cci.(grpc.ClientConnInterface)),
		intervals: make(map[time.Duration]int),
		listeners: make(map[OOBListener]struct{}),
		backoff:   internal.DefaultBackoffFunc,
	}
	return p, func() {
		p.mu.Lock()
		if p.stop != nil {
			p.stop()
			p.stop = nil
		}
		p.mu.Unlock()
		<-p.stopped
	}
}

var producerBuilderSingleton = &producerBuilder{}

// OOBListener is used to receive out-of-band load reports as they arrive.
type OOBListener interface {
	// OnLoadReport is called when a load report is received.
	OnLoadReport(*v3orcapb.OrcaLoadReport)
}

// OOBListenerOptions contains options to control how an OOBListener is called.
type OOBListenerOptions struct {
	// ReportInterval specifies how often to request the server to provide a
	// load report.  May be provided less frequently if the server requires a
	// longer interval, or may be provided more frequently if another
	// subscriber requests a shorter interval.
	ReportInterval time.Duration
}

// RegisterOOBListener registers an out-of-band load report listener on a Ready
// sc.  Any OOBListener may only be registered once per subchannel at a time.
// The returned stop function must be called when no longer needed.  Do not
// register a single OOBListener more than once per SubConn.
func RegisterOOBListener(sc balancer.SubConn, l OOBListener, opts OOBListenerOptions) (stop func()) {
	pr, closeFn := sc.GetOrBuildProducer(producerBuilderSingleton)
	p := pr.(*producer)

	p.registerListener(l, opts.ReportInterval)

	// If stop is called multiple times, prevent it from having any effect on
	// subsequent calls.
	return sync.OnceFunc(func() {
		p.unregisterListener(l, opts.ReportInterval)
		closeFn()
	})
}

type producer struct {
	client v3orcaservicegrpc.OpenRcaServiceClient

	// backoff is called between stream attempts to determine how long to delay
	// to avoid overloading a server experiencing problems.  The attempt count
	// is incremented when stream errors occur and is reset when the stream
	// reports a result.
	backoff func(int) time.Duration
	stopped chan struct{} // closed when the run goroutine exits

	mu          sync.Mutex
	intervals   map[time.Duration]int    // map from interval time to count of listeners requesting that time
	listeners   map[OOBListener]struct{} // set of registered listeners
	minInterval time.Duration
	stop        func() // stops the current run goroutine
}

// registerListener adds the listener and its requested report interval to the
// producer.
func (p *producer) registerListener(l OOBListener, interval time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.listeners[l] = struct{}{}
	p.intervals[interval]++
	if len(p.listeners) == 1 || interval < p.minInterval {
		p.minInterval = interval
		p.updateRunLocked()
	}
}

// registerListener removes the listener and its requested report interval to
// the producer.
func (p *producer) unregisterListener(l OOBListener, interval time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	delete(p.listeners, l)
	p.intervals[interval]--
	if p.intervals[interval] == 0 {
		delete(p.intervals, interval)

		if p.minInterval == interval {
			p.recomputeMinInterval()
			p.updateRunLocked()
		}
	}
}

// recomputeMinInterval sets p.minInterval to the minimum key's value in
// p.intervals.
func (p *producer) recomputeMinInterval() {
	first := true
	for interval := range p.intervals {
		if first || interval < p.minInterval {
			p.minInterval = interval
			first = false
		}
	}
}

// updateRunLocked is called whenever the run goroutine needs to be started /
// stopped / restarted due to: 1. the initial listener being registered, 2. the
// final listener being unregistered, or 3. the minimum registered interval
// changing.
func (p *producer) updateRunLocked() {
	if p.stop != nil {
		p.stop()
		p.stop = nil
	}
	if len(p.listeners) > 0 {
		var ctx context.Context
		ctx, p.stop = context.WithCancel(context.Background())
		p.stopped = make(chan struct{})
		go p.run(ctx, p.stopped, p.minInterval)
	}
}

// run manages the ORCA OOB stream on the subchannel.
func (p *producer) run(ctx context.Context, done chan struct{}, interval time.Duration) {
	defer close(done)

	runStream := func() error {
		resetBackoff, err := p.runStream(ctx, interval)
		if status.Code(err) == codes.Unimplemented {
			// Unimplemented; do not retry.
			logger.Error("Server doesn't support ORCA OOB load reporting protocol; not listening for load reports.")
			return err
		}
		// Retry for all other errors.
		if code := status.Code(err); code != codes.Unavailable && code != codes.Canceled {
			// TODO: Unavailable and Canceled should also ideally log an error,
			// but for now we receive them when shutting down the ClientConn
			// (Unavailable if the stream hasn't started yet, and Canceled if it
			// happens mid-stream).  Once we can determine the state or ensure
			// the producer is stopped before the stream ends, we can log an
			// error when it's not a natural shutdown.
			logger.Error("Received unexpected stream error:", err)
		}
		if resetBackoff {
			return backoff.ErrResetBackoff
		}
		return nil
	}
	backoff.RunF(ctx, runStream, p.backoff)
}

// runStream runs a single stream on the subchannel and returns the resulting
// error, if any, and whether or not the run loop should reset the backoff
// timer to zero or advance it.
func (p *producer) runStream(ctx context.Context, interval time.Duration) (resetBackoff bool, err error) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := p.client.StreamCoreMetrics(streamCtx, &v3orcaservicepb.OrcaLoadReportRequest{
		ReportInterval: durationpb.New(interval),
	})
	if err != nil {
		return false, err
	}

	for {
		report, err := stream.Recv()
		if err != nil {
			return resetBackoff, err
		}
		resetBackoff = true
		p.mu.Lock()
		for l := range p.listeners {
			l.OnLoadReport(report)
		}
		p.mu.Unlock()
	}
}
