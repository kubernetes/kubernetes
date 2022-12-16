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

package storagebackend

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"sync"
	"time"

	grpcprom "github.com/grpc-ecosystem/go-grpc-prometheus"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	oteltrace "go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	flowcontrolrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

const (
	StorageTypeUnset = ""
	StorageTypeETCD2 = "etcd2"
	StorageTypeETCD3 = "etcd3"

	DefaultCompactInterval      = 5 * time.Minute
	DefaultDBMetricPollInterval = 30 * time.Second
	DefaultHealthcheckTimeout   = 2 * time.Second
	DefaultReadinessTimeout     = 2 * time.Second
)

// TransportConfig holds all connection related info,  i.e. equal TransportConfig means equal servers we talk to.
type TransportConfig struct {
	// ServerList is the list of storage servers to connect with.
	ServerList []string
	// TLS credentials
	KeyFile       string
	CertFile      string
	TrustedCAFile string
	// function to determine the egress dialer. (i.e. konnectivity server dialer)
	EgressLookup egressselector.Lookup
	// The TracerProvider can add tracing the connection
	TracerProvider oteltrace.TracerProvider

	// complete guards getClient and makes sure it is only set once via Complete (see that method for more details).
	complete  bool
	getClient func() (*clientv3.Client, error)
}

// Config is configuration for creating a storage backend.
type Config struct {
	// Type defines the type of storage backend. Default ("") is "etcd3".
	Type string
	// Prefix is the prefix to all keys passed to storage.Interface methods.
	Prefix string
	// Transport holds all connection related info, i.e. equal TransportConfig means equal servers we talk to.
	Transport TransportConfig
	// Paging indicates whether the server implementation should allow paging (if it is
	// supported). This is generally configured by feature gating, or by a specific
	// resource type not wishing to allow paging, and is not intended for end users to
	// set.
	Paging bool

	Codec runtime.Codec
	// EncodeVersioner is the same groupVersioner used to build the
	// storage encoder. Given a list of kinds the input object might belong
	// to, the EncodeVersioner outputs the gvk the object will be
	// converted to before persisted in etcd.
	EncodeVersioner runtime.GroupVersioner
	// Transformer allows the value to be transformed prior to persisting into etcd.
	Transformer value.Transformer

	// CompactionInterval is an interval of requesting compaction from apiserver.
	// If the value is 0, no compaction will be issued.
	CompactionInterval time.Duration
	// CountMetricPollPeriod specifies how often should count metric be updated
	CountMetricPollPeriod time.Duration
	// DBMetricPollInterval specifies how often should storage backend metric be updated.
	DBMetricPollInterval time.Duration
	// HealthcheckTimeout specifies the timeout used when checking health
	HealthcheckTimeout time.Duration
	// ReadycheckTimeout specifies the timeout used when checking readiness
	ReadycheckTimeout time.Duration

	LeaseManagerConfig etcd3.LeaseManagerConfig

	// StorageObjectCountTracker is used to keep track of the total
	// number of objects in the storage per resource.
	StorageObjectCountTracker flowcontrolrequest.StorageObjectCountTracker
}

// ConfigForResource is a Config specialized to a particular `schema.GroupResource`
type ConfigForResource struct {
	// Config is the resource-independent configuration
	Config

	// GroupResource is the relevant one
	GroupResource schema.GroupResource
}

// ForResource specializes to the given resource
func (config *Config) ForResource(resource schema.GroupResource) *ConfigForResource {
	return &ConfigForResource{
		Config:        *config,
		GroupResource: resource,
	}
}

func NewDefaultConfig(prefix string, codec runtime.Codec) *Config {
	return &Config{
		Paging:               true,
		Prefix:               prefix,
		Codec:                codec,
		CompactionInterval:   DefaultCompactInterval,
		DBMetricPollInterval: DefaultDBMetricPollInterval,
		HealthcheckTimeout:   DefaultHealthcheckTimeout,
		ReadycheckTimeout:    DefaultReadinessTimeout,
		LeaseManagerConfig:   etcd3.NewDefaultLeaseManagerConfig(),
		Transport:            TransportConfig{TracerProvider: oteltrace.NewNoopTracerProvider()},
	}
}

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

// Complete can only be called once and must be called before making any calls to Client.
// Callers must not mutate TransportConfig after calling Complete.  For the exceptional
// cases where such functionality is needed, use the ShallowCopyAndResetComplete method.
// This method does not attempt to create an etcd client - the client is lazily initialized
// on the first call to Client.
func (c *TransportConfig) Complete(ctx context.Context) error {
	if c.complete {
		return fmt.Errorf("TransportConfig.Complete called more than once")
	}

	// we need TransportConfig.Client to be go routine safe and we need to only
	// create a single etcd client per transport config, so we use sync.Once to
	// guard the local state created in this method.  capturing this state via
	// a closure allows us to lazily initialize the client which makes it far
	// easier to keep existing unit tests passing (because most of them do not
	// actually want to create an etcd client).  this also prevents us from
	// creating a client that is not used (i.e. if Complete is called but then
	// Client is not called such as code paths involving ShallowCopyAndResetComplete).
	var (
		once   sync.Once
		client *clientv3.Client
		err    error
	)
	c.getClient = func() (*clientv3.Client, error) {
		once.Do(func() {
			client, err = getClientWithRetries(ctx, *c)
		})
		return client, err
	}
	c.complete = true

	return nil
}

// Client returns the etcd client associated with this transport.
// It is safe to call concurrently and will always return the same etcd client.
// It blocks forever until it can successfully create a client.
func (c *TransportConfig) Client() (*clientv3.Client, error) {
	if !c.complete {
		return nil, fmt.Errorf("TransportConfig.Client called without completion")
	}

	return c.getClient()
}

// ShallowCopyAndResetComplete allows an already completed TransportConfig to be re-used in a different context.
// For example, in situations where you need to mutate an already completed TransportConfig to change ServerList
// and then create a new etcd client.  This method should be used with care as it allows an unbounded amount of
// etcd clients to be created.
func (c *TransportConfig) ShallowCopyAndResetComplete() TransportConfig {
	out := *c
	out.complete = false
	out.getClient = nil
	return out
}

func getClientWithRetries(ctx context.Context, c TransportConfig) (*clientv3.Client, error) {
	// constructing the etcd v3 client blocks and times out if etcd is not available
	// retry in a loop until we successfully create the client
	// otherwise this loop terminates when the context is cancelled (i.e. server shutdown)

	var (
		client *clientv3.Client
		err    error
	)
	if pollErr := wait.PollImmediateUntilWithContext(ctx, time.Second, func(ctx context.Context) (bool, error) {
		client, err = getClient(ctx, c)
		if err != nil {
			klog.ErrorS(err, "failed to get etcd client")
		}
		return err == nil, nil
	}); pollErr != nil {
		return nil, fmt.Errorf("failed to get etcd client, pollErr=%v: %w", pollErr, err)
	}

	return client, err
}

func getClient(ctx context.Context, c TransportConfig) (*clientv3.Client, error) {
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

	client, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}

	// decorate the KV instance so we can track etcd latency per request.
	client.KV = etcd3.NewETCDLatencyTracker(client.KV)

	// lifecycle the etcd client via ctx
	go func() {
		defer utilruntime.HandleCrash()

		<-ctx.Done()
		_ = client.Close()
	}()

	return client, nil
}
