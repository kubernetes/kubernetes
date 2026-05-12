/*
 *
 * Copyright 2024 gRPC authors.
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
 *
 */

package health

import (
	"context"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/status"
)

func init() {
	producerBuilderSingleton = &producerBuilder{}
	internal.RegisterClientHealthCheckListener = registerClientSideHealthCheckListener
}

type producerBuilder struct{}

var producerBuilderSingleton *producerBuilder

// Build constructs and returns a producer and its cleanup function.
func (*producerBuilder) Build(cci any) (balancer.Producer, func()) {
	p := &healthServiceProducer{
		cc:     cci.(grpc.ClientConnInterface),
		cancel: func() {},
	}
	return p, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		p.cancel()
	}
}

type healthServiceProducer struct {
	// The following fields are initialized at build time and read-only after
	// that and therefore do not need to be guarded by a mutex.
	cc grpc.ClientConnInterface

	mu     sync.Mutex
	cancel func()
}

// registerClientSideHealthCheckListener accepts a listener to provide server
// health state via the health service.
func registerClientSideHealthCheckListener(ctx context.Context, sc balancer.SubConn, serviceName string, listener func(balancer.SubConnState)) func() {
	pr, closeFn := sc.GetOrBuildProducer(producerBuilderSingleton)
	p := pr.(*healthServiceProducer)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cancel()
	if listener == nil {
		return closeFn
	}

	ctx, cancel := context.WithCancel(ctx)
	p.cancel = cancel

	go p.startHealthCheck(ctx, sc, serviceName, listener)
	return closeFn
}

func (p *healthServiceProducer) startHealthCheck(ctx context.Context, sc balancer.SubConn, serviceName string, listener func(balancer.SubConnState)) {
	newStream := func(method string) (any, error) {
		return p.cc.NewStream(ctx, &grpc.StreamDesc{ServerStreams: true}, method)
	}

	setConnectivityState := func(state connectivity.State, err error) {
		listener(balancer.SubConnState{
			ConnectivityState: state,
			ConnectionError:   err,
		})
	}

	// Call the function through the internal variable as tests use it for
	// mocking.
	err := internal.HealthCheckFunc(ctx, newStream, setConnectivityState, serviceName)
	if err == nil {
		return
	}
	if status.Code(err) == codes.Unimplemented {
		logger.Errorf("Subchannel health check is unimplemented at server side, thus health check is disabled for SubConn %p", sc)
	} else {
		logger.Errorf("Health checking failed for SubConn %p: %v", sc, err)
	}
}
