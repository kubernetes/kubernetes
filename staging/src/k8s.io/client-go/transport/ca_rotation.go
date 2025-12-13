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
	"crypto/tls"
	"net/http"
	"os"
	"sync"
	"time"

	"k8s.io/client-go/tools/metrics"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

var caRefreshDuration = 5 * time.Minute

// atomicTransportHolder holds a transport that can be atomically updated
// when CA files change, enabling graceful CA rotation without cache complexity
type atomicTransportHolder struct {
	caFile        string
	config        *Config
	currentCAData []byte // Track the actual CA data currently in use
	// clock is used to allow for testing time-based logic.
	clock clock.Clock
	// mu covers transport and transportLastUpdated
	mu                   sync.RWMutex
	transport            *http.Transport
	transportLastChecked time.Time
}

// RoundTrip implements http.RoundTripper interface
func (h *atomicTransportHolder) RoundTrip(req *http.Request) (*http.Response, error) {
	transport := h.getTransport()
	return transport.RoundTrip(req)
}

func (h *atomicTransportHolder) getTransport() *http.Transport {
	if tr := h.getTransportIfFresh(); tr != nil {
		return tr
	}
	return h.tryRefreshTransport()
}

func (h *atomicTransportHolder) getTransportIfFresh() *http.Transport {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.clock.Since(h.transportLastChecked) < caRefreshDuration {
		return h.transport
	}
	return nil
}

func (h *atomicTransportHolder) tryRefreshTransport() *http.Transport {
	h.mu.Lock()
	defer h.mu.Unlock()

	// If some other goroutine already checked/updated the CA
	if h.clock.Since(h.transportLastChecked) < caRefreshDuration {
		return h.transport
	}

	klog.V(4).InfoS("Checking CA file content", "caFile", h.caFile)

	// Load new CA data from file
	newCAData, err := os.ReadFile(h.caFile)
	// Return old transport on read error
	if err != nil {
		metrics.TransportCAReloads.Increment("failure", "read_error")
		return h.transport
	}
	if len(newCAData) == 0 || bytes.Equal(h.currentCAData, newCAData) {
		klog.V(4).InfoS("CA file unchanged or empty, skipping transport rotation", "caFile", h.caFile)
		h.transportLastChecked = h.clock.Now()
		metrics.TransportCAReloads.Increment("success", "unchanged")
		return h.transport
	}
	klog.V(4).InfoS("CA content changed, updating transport", "caFile", h.caFile)

	// Load new CA pool
	newCAs, err := rootCertPool(newCAData)
	// Return old transport on parse error
	if err != nil {
		metrics.TransportCAReloads.Increment("failure", "ca_parse_error")
		return h.transport
	}
	newTransport := h.transport.Clone()
	if newTransport.TLSClientConfig == nil {
		newTransport.TLSClientConfig = &tls.Config{}
	}
	newTransport.TLSClientConfig.RootCAs = newCAs
	oldTransport := h.transport
	h.transport = newTransport
	h.transportLastChecked = h.clock.Now()

	// Update our tracking of current CA data
	h.currentCAData = newCAData

	if oldTransport != nil {
		// Close idle connections on the old transport to encourage migration
		oldTransport.CloseIdleConnections()
	}

	klog.V(4).InfoS("Transport updated for CA rotation", "caFile", h.caFile)
	metrics.TransportCAReloads.Increment("success", "updated")
	return newTransport
}

// newAtomicTransportHolder creates a new holder for CA file reloading scenarios
func newAtomicTransportHolder(config *Config, initialTransport *http.Transport, c clock.Clock) *atomicTransportHolder {
	holder := &atomicTransportHolder{
		caFile:               config.TLS.CAFile,
		config:               config,
		clock:                c,
		transportLastChecked: c.Now(),
		transport:            initialTransport,
	}

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
