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
	"sync/atomic"
	"time"

	grpcprom "github.com/grpc-ecosystem/go-grpc-prometheus"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/runtime"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/traces"
	"k8s.io/klog/v2"
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

	dbMetricsMonitorJitter = 0.5
)

func init() {
	// grpcprom auto-registers (via an init function) their client metrics, since we are opting out of
	// using the global prometheus registry and using our own wrapped global registry,
	// we need to explicitly register these metrics to our global registry here.
	// For reference: https://github.com/kubernetes/kubernetes/pull/81387
	legacyregistry.RawMustRegister(grpcprom.DefaultClientMetrics)
	dbMetricsMonitors = make(map[string]struct{})
}

func newETCD3HealthCheck(c storagebackend.Config) (func() error, error) {
	// constructing the etcd v3 client blocks and times out if etcd is not available.
	// retry in a loop in the background until we successfully create the client, storing the client or error encountered

	clientValue := &atomic.Value{}

	clientErrMsg := &atomic.Value{}
	clientErrMsg.Store("etcd client connection not yet established")

	go wait.PollUntil(time.Second, func() (bool, error) {
		client, err := newETCD3Client(c.Transport)
		if err != nil {
			clientErrMsg.Store(err.Error())
			return false, nil
		}
		clientValue.Store(client)
		clientErrMsg.Store("")
		return true, nil
	}, wait.NeverStop)

	return func() error {
		if errMsg := clientErrMsg.Load().(string); len(errMsg) > 0 {
			return fmt.Errorf(errMsg)
		}
		client := clientValue.Load().(*clientv3.Client)
		healthcheckTimeout := storagebackend.DefaultHealthcheckTimeout
		if c.HealthcheckTimeout != time.Duration(0) {
			healthcheckTimeout = c.HealthcheckTimeout
		}
		ctx, cancel := context.WithTimeout(context.Background(), healthcheckTimeout)
		defer cancel()
		// See https://github.com/etcd-io/etcd/blob/c57f8b3af865d1b531b979889c602ba14377420e/etcdctl/ctlv3/command/ep_command.go#L118
		_, err := client.Get(ctx, path.Join("/", c.Prefix, "health"))
		if err == nil {
			return nil
		}
		return fmt.Errorf("error getting data from etcd: %v", err)
	}, nil
}

func newETCD3Client(c storagebackend.TransportConfig) (*clientv3.Client, error) {
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
			otelgrpc.WithPropagators(traces.Propagators()),
		}
		if c.TracerProvider != nil {
			tracingOpts = append(tracingOpts, otelgrpc.WithTracerProvider(*c.TracerProvider))
		}
		// Even if there is no TracerProvider, the otelgrpc still handles context propagation.
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

type runningCompactor struct {
	interval time.Duration
	cancel   context.CancelFunc
	client   *clientv3.Client
	refs     int
}

var (
	// compactorsMu guards access to compactors map
	compactorsMu sync.Mutex
	compactors   = map[string]*runningCompactor{}
	// dbMetricsMonitorsMu guards access to dbMetricsMonitors map
	dbMetricsMonitorsMu sync.Mutex
	dbMetricsMonitors   map[string]struct{}
)

// startCompactorOnce start one compactor per transport. If the interval get smaller on repeated calls, the
// compactor is replaced. A destroy func is returned. If all destroy funcs with the same transport are called,
// the compactor is stopped.
func startCompactorOnce(c storagebackend.TransportConfig, interval time.Duration) (func(), error) {
	compactorsMu.Lock()
	defer compactorsMu.Unlock()

	key := fmt.Sprintf("%v", c) // gives: {[server1 server2] keyFile certFile caFile}
	if compactor, foundBefore := compactors[key]; !foundBefore || compactor.interval > interval {
		compactorClient, err := newETCD3Client(c)
		if err != nil {
			return nil, err
		}

		if foundBefore {
			// replace compactor
			compactor.cancel()
			compactor.client.Close()
		} else {
			// start new compactor
			compactor = &runningCompactor{}
			compactors[key] = compactor
		}

		ctx, cancel := context.WithCancel(context.Background())

		compactor.interval = interval
		compactor.cancel = cancel
		compactor.client = compactorClient

		etcd3.StartCompactor(ctx, compactorClient, interval)
	}

	compactors[key].refs++

	return func() {
		compactorsMu.Lock()
		defer compactorsMu.Unlock()

		compactor := compactors[key]
		compactor.refs--
		if compactor.refs == 0 {
			compactor.cancel()
			compactor.client.Close()
			delete(compactors, key)
		}
	}, nil
}

func newETCD3Storage(c storagebackend.ConfigForResource, newFunc func() runtime.Object) (storage.Interface, DestroyFunc, error) {
	stopCompactor, err := startCompactorOnce(c.Transport, c.CompactionInterval)
	if err != nil {
		return nil, nil, err
	}

	client, err := newETCD3Client(c.Transport)
	if err != nil {
		stopCompactor()
		return nil, nil, err
	}

	stopDBSizeMonitor, err := startDBSizeMonitorPerEndpoint(client, c.DBMetricPollInterval)
	if err != nil {
		return nil, nil, err
	}

	var once sync.Once
	destroyFunc := func() {
		// we know that storage destroy funcs are called multiple times (due to reuse in subresources).
		// Hence, we only destroy once.
		// TODO: fix duplicated storage destroy calls higher level
		once.Do(func() {
			stopCompactor()
			stopDBSizeMonitor()
			client.Close()
		})
	}
	transformer := c.Transformer
	if transformer == nil {
		transformer = value.IdentityTransformer
	}
	return etcd3.New(client, c.Codec, newFunc, c.Prefix, c.GroupResource, transformer, c.Paging, c.LeaseManagerConfig), destroyFunc, nil
}

// startDBSizeMonitorPerEndpoint starts a loop to monitor etcd database size and update the
// corresponding metric etcd_db_total_size_in_bytes for each etcd server endpoint.
func startDBSizeMonitorPerEndpoint(client *clientv3.Client, interval time.Duration) (func(), error) {
	if interval == 0 {
		return func() {}, nil
	}
	dbMetricsMonitorsMu.Lock()
	defer dbMetricsMonitorsMu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	for _, ep := range client.Endpoints() {
		if _, found := dbMetricsMonitors[ep]; found {
			continue
		}
		dbMetricsMonitors[ep] = struct{}{}
		endpoint := ep
		klog.V(4).Infof("Start monitoring storage db size metric for endpoint %s with polling interval %v", endpoint, interval)
		go wait.JitterUntilWithContext(ctx, func(context.Context) {
			epStatus, err := client.Maintenance.Status(ctx, endpoint)
			if err != nil {
				klog.V(4).Infof("Failed to get storage db size for ep %s: %v", endpoint, err)
				metrics.UpdateEtcdDbSize(endpoint, -1)
			} else {
				metrics.UpdateEtcdDbSize(endpoint, epStatus.DbSize)
			}
		}, interval, dbMetricsMonitorJitter, true)
	}

	return func() {
		cancel()
	}, nil
}
