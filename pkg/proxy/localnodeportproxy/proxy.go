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

package localnodeportproxy

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/metrics"
)

// dialTimeout is the timeout for connecting to a backend endpoint.
const dialTimeout = 5 * time.Second

// NodePortSpec describes a single NodePort that needs a localhost proxy.
// Callers build a slice of these during their own sync loop and hand it
// to SyncNodePorts.
type NodePortSpec struct {
	ServicePortName proxy.ServicePortName
	Protocol        v1.Protocol
	NodePort        int
	// Endpoints is the set of "ip:port" backends the listener round-robins over.
	Endpoints []string
	// SessionAffinityType mirrors Service.spec.sessionAffinity. When set to
	// ClientIP, consecutive connections from localhost are pinned to the same
	// endpoint for StickyMaxAgeSeconds.
	SessionAffinityType v1.ServiceAffinity
	// StickyMaxAgeSeconds is the affinity timeout in seconds, used only when
	// SessionAffinityType is ClientIP.
	StickyMaxAgeSeconds int
}

// affinityTimeout returns the effective affinity window for the spec, or 0
// when affinity is disabled.
func (s *NodePortSpec) affinityTimeout() time.Duration {
	if s.SessionAffinityType != v1.ServiceAffinityClientIP {
		return 0
	}
	return time.Duration(s.StickyMaxAgeSeconds) * time.Second
}

// LocalNodePortProxy manages userspace L4 proxy listeners on localhost
// for NodePort services in nftables mode, which has no route_localnet
// equivalent for serving localhost NodePorts in kernel space.
type LocalNodePortProxy struct {
	ctx      context.Context
	logger   klog.Logger
	listenIP string
	family   string

	// mu guards active against concurrent Sync/Shutdown calls.
	mu     sync.Mutex
	active map[string]*nodePortListener
}

// NewLocalNodePortProxy creates a new proxy for the given IP family.
func NewLocalNodePortProxy(ctx context.Context, ipFamily v1.IPFamily) *LocalNodePortProxy {
	listenIP := "127.0.0.1"
	if ipFamily == v1.IPv6Protocol {
		listenIP = "::1"
	}
	return &LocalNodePortProxy{
		ctx:      ctx,
		logger:   klog.FromContext(ctx),
		listenIP: listenIP,
		family:   string(ipFamily),
		active:   make(map[string]*nodePortListener),
	}
}

// SyncNodePorts reconciles the set of active localhost NodePort proxies with
// the desired state.
func (p *LocalNodePortProxy) SyncNodePorts(desired []NodePortSpec) {
	byKey := make(map[string]*NodePortSpec, len(desired))
	for i := range desired {
		spec := &desired[i]
		if len(spec.Endpoints) == 0 {
			continue
		}
		key := nodePortKey(spec.Protocol, spec.NodePort)
		byKey[key] = spec
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Remove stale listeners
	for key, l := range p.active {
		if _, ok := byKey[key]; !ok {
			p.logger.V(2).Info("Removing localhost nodeport proxy", "protocol", l.protocol, "port", l.port)
			l.shutdown()
			delete(p.active, key)
		}
	}

	// Add or update
	for key, spec := range byKey {
		if existing, ok := p.active[key]; ok {
			existing.update(spec)
			continue
		}
		if spec.Protocol != v1.ProtocolTCP {
			p.logger.V(2).Info("Skipping non-TCP localhost nodeport proxy",
				"service", spec.ServicePortName, "protocol", spec.Protocol, "port", spec.NodePort)
			continue
		}
		l, err := p.newNodePortListener(spec)
		if err != nil {
			p.logger.Error(err, "Failed to create localhost nodeport proxy", "protocol", spec.Protocol, "port", spec.NodePort)
			metrics.LocalhostNodePortListenerCreationFailuresTotal.WithLabelValues(p.family).Inc()
			continue
		}
		p.active[key] = l
		p.logger.V(2).Info("Created localhost nodeport proxy", "protocol", spec.Protocol, "port", spec.NodePort, "endpoints", len(spec.Endpoints))
	}

	metrics.LocalhostNodePortListeners.WithLabelValues(p.family).Set(float64(len(p.active)))
}

func nodePortKey(protocol v1.Protocol, port int) string {
	return fmt.Sprintf("%s/%d", strings.ToLower(string(protocol)), port)
}

// Shutdown tears down all active listeners and closes any in-flight connections.
func (p *LocalNodePortProxy) Shutdown() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for key, l := range p.active {
		l.shutdown()
		delete(p.active, key)
	}
	metrics.LocalhostNodePortListeners.WithLabelValues(p.family).Set(0)
}

func (p *LocalNodePortProxy) newNodePortListener(spec *NodePortSpec) (*nodePortListener, error) {
	addr := net.JoinHostPort(p.listenIP, fmt.Sprintf("%d", spec.NodePort))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on %s: %w", addr, err)
	}

	ctx, cancel := context.WithCancel(p.ctx)
	l := &nodePortListener{
		protocol:        spec.Protocol,
		port:            spec.NodePort,
		logger:          p.logger,
		endpoints:       spec.Endpoints,
		affinityTimeout: spec.affinityTimeout(),
		listener:        listener,
		cancel:          cancel,
		pick:            rand.IntN,
	}
	go l.acceptLoop(ctx)
	return l, nil
}

type nodePortListener struct {
	// Immutable after construction.
	protocol v1.Protocol
	port     int
	logger   klog.Logger
	listener net.Listener
	cancel   context.CancelFunc
	// pick returns an index into endpoints. It defaults to rand.IntN and is a
	// field so tests can substitute a deterministic picker at construction.
	pick func(int) int

	// mu guards the endpoint-selection state below
	mu        sync.Mutex
	endpoints []string
	// affinityTimeout is 0 when SessionAffinity is disabled; otherwise, it is
	// the duration for which a picked endpoint stays pinned for localhost
	// traffic. Since the source IP for all localhost traffic is 127.0.0.1 or
	// ::1, ClientIP affinity effectively pins all traffic through this
	// listener to a single endpoint until the pin goes stale.
	affinityTimeout time.Duration
	pinnedEndpoint  string // "ip:port" of the currently pinned endpoint, empty if none
	pinnedLastUsed  time.Time
}

func (l *nodePortListener) acceptLoop(ctx context.Context) {
	for {
		conn, err := l.listener.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return
			}
			l.logger.Error(err, "Accept error on localhost nodeport proxy", "protocol", l.protocol, "port", l.port)
			continue
		}
		go l.handleTCPConn(ctx, conn)
	}
}

func (l *nodePortListener) handleTCPConn(ctx context.Context, clientConn net.Conn) {
	defer clientConn.Close() //nolint:errcheck

	ep := l.pickEndpoint()
	if ep == "" {
		l.logger.V(4).Info("No endpoints available for localhost nodeport proxy", "protocol", l.protocol, "port", l.port)
		return
	}

	backendConn, err := net.DialTimeout("tcp", ep, dialTimeout)
	if err != nil {
		l.logger.Error(err, "Failed to connect to backend", "protocol", l.protocol, "port", l.port, "endpoint", ep)
		return
	}
	defer backendConn.Close() //nolint:errcheck

	// Force both sides closed on listener shutdown so io.Copy returns and
	// the handler goroutine doesn't leak on a long-lived/idle connection.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = clientConn.Close()
			_ = backendConn.Close()
		case <-done:
		}
	}()

	var wg sync.WaitGroup
	wg.Go(func() {
		_, _ = io.Copy(backendConn, clientConn)
		if tc, ok := backendConn.(*net.TCPConn); ok {
			_ = tc.CloseWrite()
		}
	})
	wg.Go(func() {
		_, _ = io.Copy(clientConn, backendConn)
		if tc, ok := clientConn.(*net.TCPConn); ok {
			_ = tc.CloseWrite()
		}
	})
	wg.Wait()
}

func (l *nodePortListener) pickEndpoint() string {
	l.mu.Lock()
	defer l.mu.Unlock()

	if len(l.endpoints) == 0 {
		return ""
	}
	now := time.Now()
	if l.affinityTimeout > 0 && l.pinnedEndpoint != "" && now.Sub(l.pinnedLastUsed) <= l.affinityTimeout {
		for _, ep := range l.endpoints {
			if ep == l.pinnedEndpoint {
				l.pinnedLastUsed = now
				return ep
			}
		}
		// Pinned endpoint no longer in the set; fall through and pick a new one.
	}
	ep := l.endpoints[l.pick(len(l.endpoints))]
	if l.affinityTimeout > 0 {
		l.pinnedEndpoint = ep
		l.pinnedLastUsed = now
	}
	return ep
}

func (l *nodePortListener) update(spec *NodePortSpec) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.endpoints = spec.Endpoints
	newTimeout := spec.affinityTimeout()
	if l.affinityTimeout != newTimeout {
		// Affinity config changed; drop any stale pin.
		l.pinnedEndpoint = ""
		l.affinityTimeout = newTimeout
	}
}

func (l *nodePortListener) shutdown() {
	l.cancel()
	_ = l.listener.Close()
}
