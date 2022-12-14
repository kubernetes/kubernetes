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
	"path"
	"sync"
	"time"

	grpcprom "github.com/grpc-ecosystem/go-grpc-prometheus"
	clientv3 "go.etcd.io/etcd/client/v3"
	"golang.org/x/time/rate"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

const (
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

func newETCD3HealthCheck(c storagebackend.Config, stopCh <-chan struct{}) (func() error, error) {
	timeout := storagebackend.DefaultHealthcheckTimeout
	if c.HealthcheckTimeout != 0 {
		timeout = c.HealthcheckTimeout
	}
	return newETCD3Check(c, timeout, stopCh)
}

func newETCD3ReadyCheck(c storagebackend.Config, stopCh <-chan struct{}) (func() error, error) {
	timeout := storagebackend.DefaultReadinessTimeout
	if c.ReadycheckTimeout != 0 {
		timeout = c.ReadycheckTimeout
	}
	return newETCD3Check(c, timeout, stopCh)
}

// atomic error acts as a cache for atomically store an error
// the error is only updated if the timestamp is more recent than
// current stored error.
type atomicLastError struct {
	mu        sync.RWMutex
	err       error
	timestamp time.Time
}

func (a *atomicLastError) Store(err error, t time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.timestamp.IsZero() || a.timestamp.Before(t) {
		a.err = err
		a.timestamp = t
	}
}

func (a *atomicLastError) Load() error {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.err
}

func newETCD3Check(c storagebackend.Config, timeout time.Duration, stopCh <-chan struct{}) (func() error, error) {
	// constructing the etcd v3 client blocks until it succeeds so we fetch the client in the background

	lock := sync.RWMutex{}
	var client *clientv3.Client
	clientErr := fmt.Errorf("etcd client connection not yet established")

	go func() {
		defer utilruntime.HandleCrash()

		newClient, newClientErr := newETCD3Client(c.Transport)

		// do nothing if the server is already not shutting down
		select {
		case <-stopCh:
			return
		default:
		}

		lock.Lock()
		defer lock.Unlock()

		client = newClient
		clientErr = newClientErr
	}()

	// limit to a request every half of the configured timeout with a maximum burst of one
	// rate limited requests will receive the last request sent error (note: not the last received response)
	limiter := rate.NewLimiter(rate.Every(timeout/2), 1)
	// initial state is the clientErr
	lastError := &atomicLastError{err: fmt.Errorf("etcd client connection not yet established")}

	return func() error {
		lock.RLock()
		defer lock.RUnlock()

		if clientErr != nil {
			return clientErr
		}
		if limiter.Allow() == false {
			return lastError.Load()
		}
		ctx, cancel := context.WithTimeout(client.Ctx(), timeout)
		defer cancel()
		// See https://github.com/etcd-io/etcd/blob/c57f8b3af865d1b531b979889c602ba14377420e/etcdctl/ctlv3/command/ep_command.go#L118
		now := time.Now()
		_, err := client.Get(ctx, path.Join("/", c.Prefix, "health"))
		if err != nil {
			err = fmt.Errorf("error getting data from etcd: %w", err)
		}
		lastError.Store(err, now)
		return err
	}, nil
}

var newETCD3Client = func(c storagebackend.TransportConfig) (*clientv3.Client, error) {
	return c.Client()
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

	key := fmt.Sprintf("%v", c) // gives: {[server1 server2] keyFile certFile caFile}  // TODO does this make sense?
	if compactor, foundBefore := compactors[key]; !foundBefore || compactor.interval > interval {
		compactorClient, err := newETCD3Client(c)
		if err != nil {
			return nil, err
		}

		if foundBefore {
			// replace compactor
			compactor.cancel()
		} else {
			// start new compactor
			compactor = &runningCompactor{}
			compactors[key] = compactor
		}

		ctx, cancel := context.WithCancel(compactorClient.Ctx())

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

	stopDBSizeMonitor := startDBSizeMonitorPerEndpoint(client, c.DBMetricPollInterval)

	var once sync.Once
	destroyFunc := func() {
		// we know that storage destroy funcs are called multiple times (due to reuse in subresources).
		// Hence, we only destroy once.
		// TODO: fix duplicated storage destroy calls higher level
		once.Do(func() {
			stopCompactor()
			stopDBSizeMonitor()
		})
	}
	transformer := c.Transformer
	if transformer == nil {
		transformer = identity.NewEncryptCheckTransformer()
	}
	return etcd3.New(client, c.Codec, newFunc, c.Prefix, c.GroupResource, transformer, c.Paging, c.LeaseManagerConfig), destroyFunc, nil
}

// startDBSizeMonitorPerEndpoint starts a loop to monitor etcd database size and update the
// corresponding metric etcd_db_total_size_in_bytes for each etcd server endpoint.
func startDBSizeMonitorPerEndpoint(client *clientv3.Client, interval time.Duration) func() {
	if interval == 0 {
		return func() {}
	}
	dbMetricsMonitorsMu.Lock()
	defer dbMetricsMonitorsMu.Unlock()

	ctx, cancel := context.WithCancel(client.Ctx())
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

	return cancel
}
