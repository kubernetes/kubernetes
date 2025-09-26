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
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// CARefreshDuration is exposed so that tests can crank up the refresh frequency.
var CARefreshDuration = 5 * time.Minute

// atomicTransportHolder holds a transport that can be atomically updated
// when CA files change, enabling graceful CA rotation without cache complexity
type atomicTransportHolder struct {
	transport     atomic.Pointer[http.Transport]
	caFile        string
	config        *Config
	currentCAData []byte // Track the actual CA data currently in use
	// clock is used to allow for testing time-based logic.
	clock       clock.Clock
	mu          sync.Mutex
	lastChecked time.Time
}

// RoundTrip implements http.RoundTripper interface
func (h *atomicTransportHolder) RoundTrip(req *http.Request) (*http.Response, error) {
	if err := h.refreshTransportIfNeeded(); err != nil {
		klog.Error(err, "Failed to refresh root CAs", "caFile", h.caFile)
	}
	return h.transport.Load().RoundTrip(req)
}

// newAtomicTransportHolder creates a new holder for CA file reloading scenarios
func newAtomicTransportHolder(config *Config, initialTransport *http.Transport, c clock.Clock) *atomicTransportHolder {
	holder := &atomicTransportHolder{
		caFile:      config.TLS.CAFile,
		config:      config,
		clock:       c,
		lastChecked: c.Now(),
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

// refreshTransportIfNeeded checks if the CA file has changed and rotates the transport if needed
func (h *atomicTransportHolder) refreshTransportIfNeeded() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.clock.Since(h.lastChecked) < CARefreshDuration {
		return nil
	}

	h.lastChecked = h.clock.Now()

	klog.InfoS("Checking CA file content", "caFile", h.caFile)

	// Load new CA data from file
	newCAData, err := os.ReadFile(h.caFile)
	if err != nil {
		return err
	}

	if len(newCAData) == 0 || bytes.Equal(h.currentCAData, newCAData) {
		klog.InfoS("CA file unchanged or empty, skipping transport rotation", "caFile", h.caFile)
		return nil
	}

	klog.InfoS("CA content changed, updating transport", "caFile", h.caFile)

	// Load new CA pool
	newCAs, err := rootCertPool(newCAData)
	if err != nil {
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

	klog.InfoS("Transport updated for CA rotation", "caFile", h.caFile)
	return nil
}
