/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package factory

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"path"
	"strings"
	"sync"
	"time"

	grpcprom "github.com/grpc-ecosystem/go-grpc-prometheus"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/runtime"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics/legacyregistry"
	tracing "k8s.io/component-base/tracing"
)

const (
	// The short keepalive timeout and interval have been chosen to aggressively
	// detect a failed etcd server without introducing much overhead.
	keepaliveTime    = 30 * time.Second
	keepaliveTimeout = 10 * time.Second

	// dialTimeout is the timeout for failing to establish a connection.
	// It is set to 20 seconds as times shorter than that will cause TLS connections to fail
	// on heavily loaded arm64 CPUs (issue #64649)
	dialTimeout = 20 * time.Second
)

func init() {
	// grpcprom auto-registers (via an init function) their client metrics, since we are opting out of
	// using the global prometheus registry and using our own wrapped global registry,
	// we need to explicitly register these metrics to our global registry here.
	// For reference: https://github.com/kubernetes/kubernetes/pull/81387
	legacyregistry.RawMustRegister(grpcprom.DefaultClientMetrics)
}

type activeETCD3Client struct {
	client *clientv3.Client
	refs   int

	compactionCancel   func()
	compactionInterval time.Duration
	metricPollCancel   func()
	metricPollInterval time.Duration
}

type etcd3ClientCache struct {
	mx     sync.Mutex
	active map[string]*activeETCD3Client
}

func newETCD3ClientCache() *etcd3ClientCache {
	return &etcd3ClientCache{active: make(map[string]*activeETCD3Client)}
}

func etcd3ClientCacheKey(tc storagebackend.TransportConfig) string {
	return fmt.Sprintf("%v", tc) // gives: {[server1 server2] keyFile certFile caFile}
}

// For returns a compacted etcd v3 client for the supplied transport config. One
// client and compactor are started for each transport. Callers must call the
// returned destroy function once they are done with the client.
func (e *etcd3ClientCache) For(tc storagebackend.TransportConfig, compactionInterval, metricPollInterval time.Duration) (*clientv3.Client, error) {
	e.mx.Lock()
	defer e.mx.Unlock()

	k := etcd3ClientCacheKey(tc)
	if a, ok := e.active[k]; ok {
		// We want to compact and poll at the shortest interval specified by any caller.
		if a.compactionInterval > compactionInterval {
			a.compactionCancel()
			ctx, cancel := context.WithCancel(context.Background())
			etcd3.StartCompactor(ctx, a.client, compactionInterval)
			a.compactionInterval = compactionInterval
			a.compactionCancel = cancel
		}
		if a.metricPollInterval > metricPollInterval {
			a.metricPollCancel()
			ctx, cancel := context.WithCancel(context.Background())
			etcd3.StartDBSizeMonitor(ctx, a.client, metricPollInterval)
			a.metricPollInterval = metricPollInterval
			a.metricPollCancel = cancel
		}
		a.refs++
		return a.client, nil
	}

	c, err := newETCD3Client(tc)
	if err != nil {
		return nil, fmt.Errorf("cannot create new etcd3 client: %w", err)
	}

	// Decorate the KV instance so we can track etcd latency per request.
	// TODO(negz): Does it hurt that this will track latency for compaction?
	c.KV = etcd3.NewETCDLatencyTracker(c.KV)

	cctx, ccancel := context.WithCancel(context.Background())
	etcd3.StartCompactor(cctx, c, compactionInterval)

	mctx, mcancel := context.WithCancel(context.Background())
	etcd3.StartDBSizeMonitor(mctx, c, metricPollInterval)

	e.active[k] = &activeETCD3Client{
		client:             c,
		refs:               1,
		compactionCancel:   ccancel,
		compactionInterval: compactionInterval,
		metricPollCancel:   mcancel,
		metricPollInterval: metricPollInterval,
	}
	return c, nil
}

func (e *etcd3ClientCache) Done(tc storagebackend.TransportConfig) {
	e.mx.Lock()
	defer e.mx.Unlock()

	k := etcd3ClientCacheKey(tc)
	if _, ok := e.active[k]; !ok {
		return
	}

	e.active[k].refs--
	if e.active[k].refs > 0 {
		return
	}
	e.active[k].compactionCancel()
	e.active[k].metricPollCancel()
	e.active[k].client.Close()
	delete(e.active, k)
}

func newETCD3HealthCheck(c storagebackend.Config, stopCh <-chan struct{}) (func() error, error) {
	timeout := storagebackend.DefaultHealthcheckTimeout
	if c.HealthcheckTimeout != time.Duration(0) {
		timeout = c.HealthcheckTimeout
	}
	return newETCD3Check(c, timeout, stopCh)
}

func newETCD3ReadyCheck(c storagebackend.Config, stopCh <-chan struct{}) (func() error, error) {
	timeout := storagebackend.DefaultReadinessTimeout
	if c.ReadycheckTimeout != time.Duration(0) {
		timeout = c.ReadycheckTimeout
	}
	return newETCD3Check(c, timeout, stopCh)
}

func newETCD3Check(c storagebackend.Config, timeout time.Duration, stopCh <-chan struct{}) (func() error, error) {
	// constructing the etcd v3 client blocks and times out if etcd is not available.
	// retry in a loop in the background until we successfully create the client, storing the client or error encountered

	lock := sync.Mutex{}
	var client *clientv3.Client
	clientErr := fmt.Errorf("etcd client connection not yet established")

	go wait.PollUntil(time.Second, func() (bool, error) {
		newClient, err := newETCD3Client(c.Transport)

		lock.Lock()
		defer lock.Unlock()

		// Ensure that server is already not shutting down.
		select {
		case <-stopCh:
			if err == nil {
				newClient.Close()
			}
			return true, nil
		default:
		}

		if err != nil {
			clientErr = err
			return false, nil
		}
		client = newClient
		clientErr = nil
		return true, nil
	}, stopCh)

	// Close the client on shutdown.
	go func() {
		defer utilruntime.HandleCrash()
		<-stopCh

		lock.Lock()
		defer lock.Unlock()
		if client != nil {
			client.Close()
			clientErr = fmt.Errorf("server is shutting down")
		}
	}()

	return func() error {
		// Given that client is closed on shutdown we hold the lock for
		// the entire period of healthcheck call to ensure that client will
		// not be closed during healthcheck.
		// Given that healthchecks has a 2s timeout, worst case of blocking
		// shutdown for additional 2s seems acceptable.
		lock.Lock()
		defer lock.Unlock()
		if clientErr != nil {
			return clientErr
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		// See https://github.com/etcd-io/etcd/blob/c57f8b3af865d1b531b979889c602ba14377420e/etcdctl/ctlv3/command/ep_command.go#L118
		_, err := client.Get(ctx, path.Join("/", c.Prefix, "health"))
		if err == nil {
			return nil
		}
		return fmt.Errorf("error getting data from etcd: %w", err)
	}, nil
}

var newETCD3Client = func(c storagebackend.TransportConfig) (*clientv3.Client, error) {
	tlsInfo := transport.TLSInfo{
		CertFile:      c.CertFile,
		KeyFile:       c.KeyFile,
		TrustedCAFile: c.TrustedCAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, err
	}
	// NOTE: Client relies on nil tlsConfig
	// for non-secure connections, update the implicit variable
	if len(c.CertFile) == 0 && len(c.KeyFile) == 0 && len(c.TrustedCAFile) == 0 {
		tlsConfig = nil
	}
	networkContext := egressselector.Etcd.AsNetworkContext()
	var egressDialer utilnet.DialFunc
	if c.EgressLookup != nil {
		egressDialer, err = c.EgressLookup(networkContext)
		if err != nil {
			return nil, err
		}
	}
	dialOptions := []grpc.DialOption{
		grpc.WithBlock(), // block until the underlying connection is up
		// use chained interceptors so that the default (retry and backoff) interceptors are added.
		// otherwise they will be overwritten by the metric interceptor.
		//
		// these optional interceptors will be placed after the default ones.
		// which seems to be what we want as the metrics will be collected on each attempt (retry)
		grpc.WithChainUnaryInterceptor(grpcprom.UnaryClientInterceptor),
		grpc.WithChainStreamInterceptor(grpcprom.StreamClientInterceptor),
	}
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.APIServerTracing) {
		tracingOpts := []otelgrpc.Option{
			otelgrpc.WithPropagators(tracing.Propagators()),
			otelgrpc.WithTracerProvider(c.TracerProvider),
		}
		// Even with Noop  TracerProvider, the otelgrpc still handles context propagation.
		// See https://github.com/open-telemetry/opentelemetry-go/tree/main/example/passthrough
		dialOptions = append(dialOptions,
			grpc.WithUnaryInterceptor(otelgrpc.UnaryClientInterceptor(tracingOpts...)),
			grpc.WithStreamInterceptor(otelgrpc.StreamClientInterceptor(tracingOpts...)))
	}
	if egressDialer != nil {
		dialer := func(ctx context.Context, addr string) (net.Conn, error) {
			if strings.Contains(addr, "//") {
				// etcd client prior to 3.5 passed URLs to dialer, normalize to address
				u, err := url.Parse(addr)
				if err != nil {
					return nil, err
				}
				addr = u.Host
			}
			return egressDialer(ctx, "tcp", addr)
		}
		dialOptions = append(dialOptions, grpc.WithContextDialer(dialer))
	}

	cfg := clientv3.Config{
		DialTimeout:          dialTimeout,
		DialKeepAliveTime:    keepaliveTime,
		DialKeepAliveTimeout: keepaliveTimeout,
		DialOptions:          dialOptions,
		Endpoints:            c.ServerList,
		TLS:                  tlsConfig,
	}

	return clientv3.New(cfg)
}

func newETCD3Storage(client *etcd3ClientCache, tc storagebackend.ConfigForResource, newFunc func() runtime.Object) (storage.Interface, DestroyFunc, error) {
	c, err := client.For(tc.Transport, tc.CompactionInterval, tc.DBMetricPollInterval)
	if err != nil {
		return nil, nil, err
	}

	var once sync.Once
	destroyFunc := func() {
		// we know that storage destroy funcs are called multiple times (due to reuse in subresources).
		// Hence, we only destroy once.
		// TODO: fix duplicated storage destroy calls higher level
		once.Do(func() {
			client.Done(tc.Transport)
		})
	}
	transformer := tc.Transformer
	if transformer == nil {
		transformer = value.IdentityTransformer
	}
	return etcd3.New(c, tc.Codec, newFunc, tc.Prefix, tc.GroupResource, transformer, tc.Paging, tc.LeaseManagerConfig), destroyFunc, nil
}
