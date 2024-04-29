/*
Copyright 2024 The Kubernetes Authors.

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

package transport

import (
	"bytes"
	"context"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	dynamicRootCATransportControllerName = "DynamicRootCA"
	queueKey                             = "dynamic-root-ca-key"
)

func newDynamicRootCATransport(transport *http.Transport) *dynamicRootCATransport {
	p := &atomic.Pointer[http.Transport]{}
	p.Store(transport)
	return &dynamicRootCATransport{
		container: transportWaitGroup{transport: transport, wg: &sync.WaitGroup{}},
	}
}

func newRootCASyncer(stopCtx context.Context, dt *dynamicRootCATransport, caFile string, caBytes []byte) *rootCASyncer {
	return &rootCASyncer{
		stopCtx:                stopCtx,
		dynamicRootCATransport: dt,
		caFile:                 caFile,
		caBytes:                caBytes,
	}
}

// the controller uses this abstraction to periodically
// refresh the reloadable transport object.
type syncer interface {
	sync() error
}

func newDynamicRootCATransportController(syncer syncer) *dynamicRootCATransportController {
	// TODO: the default has an exponential failure rate limiter with a base
	// delay of 5ms which may be more aggressive than what we need here
	queue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), dynamicRootCATransportControllerName)
	controller := &dynamicRootCATransportController{queue: queue, syncer: syncer}
	controller.queueAdderFn = func(stopCtx context.Context) {
		// same interval as dynamic client cert rotation
		err := wait.PollUntilContextCancel(stopCtx, CertCallbackRefreshDuration, true, func(_ context.Context) (bool, error) {
			queue.Add(queueKey)
			return false, nil
		})
		klog.V(3).InfoS("Controller queue add function has finished", "name", dynamicRootCATransportControllerName, "err", err)
	}
	return controller
}

type transportWaitGroup struct {
	transport *http.Transport
	wg        *sync.WaitGroup
}

type dynamicRootCATransport struct {
	lock      sync.RWMutex
	container transportWaitGroup
}

func (dt *dynamicRootCATransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var container transportWaitGroup
	dt.lock.RLock()
	container = dt.container
	container.wg.Add(1)
	dt.lock.RUnlock()

	defer container.wg.Done()
	return container.transport.RoundTrip(req)
}

func (dt *dynamicRootCATransport) WrappedRoundTripper() http.RoundTripper {
	var transport *http.Transport
	dt.lock.RLock()
	transport = dt.container.transport
	dt.lock.RUnlock()

	// TODO: side effects for exposing the internal transport object?
	return transport
}

func (dt *dynamicRootCATransport) refresh(rootCAs *x509.CertPool) transportWaitGroup {
	dt.lock.Lock()
	defer dt.lock.Unlock()

	t := dt.container.transport.Clone()
	t.TLSClientConfig.RootCAs = rootCAs

	container := dt.container
	dt.container = transportWaitGroup{transport: t, wg: &sync.WaitGroup{}}
	return container
}

type rootCASyncer struct {
	*dynamicRootCATransport
	stopCtx context.Context

	lock    sync.Mutex
	caBytes []byte
	caFile  string
}

// sync refreshes the transport object with a new root CA certificate,
// if the file has been written to with a new CA.
func (r *rootCASyncer) sync() error {
	r.lock.Lock()
	defer r.lock.Unlock()

	newCABytes, err := os.ReadFile(r.caFile)
	if err != nil {
		return fmt.Errorf("failed to read CA file: %q - err: %w", r.caFile, err)
	}
	if len(newCABytes) == 0 {
		klog.InfoS("CA file is empty", "file", r.caFile)
		return nil
	}
	if bytes.Equal(newCABytes, r.caBytes) {
		return nil
	}

	rootCAs, err := rootCertPool(newCABytes)
	if err != nil {
		return fmt.Errorf("failed to build cert pool from CA file: %q - err: %w", r.caFile, err)
	}

	c := r.refresh(rootCAs)
	r.caBytes = newCABytes
	klog.InfoS("Root CA has been reloaded", "controller", dynamicRootCATransportControllerName, "file", r.caFile)

	go r.clean(c)
	return nil
}

func (r *rootCASyncer) clean(c transportWaitGroup) <-chan struct{} {
	defer utilruntime.HandleCrash()

	doneCh, roundTripperDoneCh := make(chan struct{}), make(chan struct{})
	defer close(doneCh)
	go func() {
		defer utilruntime.HandleCrash()
		defer close(roundTripperDoneCh)
		c.wg.Wait()
	}()

	// first close the idle connections
	c.transport.CloseIdleConnections()
	select {
	case <-roundTripperDoneCh:
		c.transport.CloseIdleConnections()
		klog.InfoS("All HTTP.Transport.RoundTripper(s) have returned", "controller", dynamicRootCATransportControllerName, "file", r.caFile, "transport", fmt.Sprintf("%p", c.transport))
	case <-r.stopCtx.Done():
	}

	return doneCh
}

type dynamicRootCATransportController struct {
	queue workqueue.RateLimitingInterface
	// this function is responsible for adding keys to the queue,
	// it will be executed asynchronusly in a new goroutine.
	// it makes unit tests flake free and not dependent on time.Sleep
	queueAdderFn func(stopCtx context.Context)
	// this is the work function that refreshes the transport
	syncer syncer
}

func (c *dynamicRootCATransportController) Run(stopCtx context.Context) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.V(3).InfoS("Controller is starting", "name", dynamicRootCATransportControllerName)
	defer klog.V(3).InfoS("Controller is shutting down", "name", dynamicRootCATransportControllerName)

	go wait.Until(c.runWorker, time.Second, stopCtx.Done())
	go wait.Until(func() { c.queueAdderFn(stopCtx) }, time.Second, stopCtx.Done())

	<-stopCtx.Done()
}

func (c *dynamicRootCATransportController) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *dynamicRootCATransportController) processNextWorkItem() bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	if err := c.syncer.sync(); err != nil {
		utilruntime.HandleError(fmt.Errorf("[%s]: %v failed with: %w", dynamicRootCATransportControllerName, dsKey, err))
		c.queue.AddRateLimited(dsKey)
		return true
	}

	c.queue.Forget(dsKey)
	return true
}
