/*
Copyright 2025 The Kubernetes Authors.

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
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"github.com/containerd/log"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
)

const caRotationWorkItemKey = "ca-rotation"

// CARotationRefreshDuration is exposed so that integration tests can crank up the reload speed.
var CARotationRefreshDuration = 5 * time.Minute

// atomicTransportHolder holds a transport that can be atomically updated
// when CA files change, enabling graceful CA rotation without cache complexity
type atomicTransportHolder struct {
	transport     atomic.Pointer[http.Transport]
	caFile        string
	config        *Config
	currentCAData []byte // Track the actual CA data currently in use

	// queue only ever has one item, but it has nice error handling backoff/retry semantics
	queue workqueue.TypedRateLimitingInterface[string]
}

// RoundTrip implements http.RoundTripper interface
func (h *atomicTransportHolder) RoundTrip(req *http.Request) (*http.Response, error) {
	return h.transport.Load().RoundTrip(req)
}

// newAtomicTransportHolder creates a new holder for CA file reloading scenarios
func newAtomicTransportHolder(config *Config, initialTransport *http.Transport) *atomicTransportHolder {
	holder := &atomicTransportHolder{
		caFile: config.TLS.CAFile,
		config: config,
		queue: workqueue.NewTypedRateLimitingQueue(
			workqueue.DefaultTypedControllerRateLimiter[string](),
		),
	}
	holder.transport.Store(initialTransport)

	// Initialize currentCAData with the CA data that was actually loaded into the transport
	if len(config.TLS.CAData) > 0 {
		holder.currentCAData = config.TLS.CAData
	} else if len(config.TLS.CAFile) > 0 {
		// Read the initial CA data from file
		if caData, err := os.ReadFile(config.TLS.CAFile); err == nil {
			holder.currentCAData = caData
		}
	}

	return holder
}

// run starts the controller and blocks until stopCh is closed.
func (h *atomicTransportHolder) run(stopCh <-chan struct{}, rotationRefreshDuration time.Duration) {
	defer h.queue.ShutDown()

	log.L.Info("Starting CA rotation controller")
	defer log.L.Info("Shutting down CA rotation controller")

	go wait.Until(h.runWorker, time.Second, stopCh)

	go wait.PollUntilContextCancel(wait.ContextForChannel(stopCh), rotationRefreshDuration, true, func(_ context.Context) (bool, error) {
		h.queue.Add(caRotationWorkItemKey)
		return false, nil
	})

	<-stopCh
}

func (h *atomicTransportHolder) runWorker() {
	for h.processNextWorkItem() {
	}
}

func (h *atomicTransportHolder) processNextWorkItem() bool {
	obj, quit := h.queue.Get()
	if quit {
		return false
	}
	defer h.queue.Done(obj)

	err := h.checkCAFileAndRotate()
	if err == nil {
		h.queue.Forget(obj)
		return true
	}

	h.queue.AddRateLimited(obj)

	return true
}

// checkCAFileAndRotate checks if the CA file has changed and rotates the transport if needed
func (h *atomicTransportHolder) checkCAFileAndRotate() error {
	log.L.Info("Checking CA file content", "caFile", h.caFile)

	// Load new CA data from file
	newCAData, err := os.ReadFile(h.caFile)
	if err != nil {
		log.L.Error(err, "Failed to load CA data from file", "caFile", h.caFile)
		return err
	}

	// Skip rotation if CA data is empty (file might be in the middle of being written)
	if len(newCAData) == 0 {
		log.L.Info("CA file is empty, skipping rotation", "caFile", h.caFile)
		return nil
	}

	// Compare with current CA data actually in use
	if bytes.Equal(h.currentCAData, newCAData) {
		log.L.Info("CA content unchanged, skipping transport rotation", "caFile", h.caFile)
		return nil
	}

	log.L.Info("CA content changed, updating transport", "caFile", h.caFile)

	// Load new CA pool
	newCAs, err := rootCertPool(newCAData)
	if err != nil {
		log.L.Error(err, "Failed to load CA pool from CA file", "caFile", h.caFile)
		return err
	}

	// Clone the current transport and update its RootCAs
	newTransport := h.transport.Load().Clone()
	newTransport.TLSClientConfig.RootCAs = newCAs
	oldTransport := h.transport.Swap(newTransport)

	// Update our tracking of current CA data
	h.currentCAData = newCAData

	if oldTransport != nil {
		// Close idle connections on the old transport to encourage migration
		oldTransport.CloseIdleConnections()
	}

	log.L.Infof("Transport updated for CA rotation, caFile: %v", h.caFile)

	return nil
}
