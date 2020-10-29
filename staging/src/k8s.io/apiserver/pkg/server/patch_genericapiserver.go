/*
Copyright 2020 The Kubernetes Authors.

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

package server

import (
	"net"
	"net/http"
	"strings"
	"sync"
	goatomic "sync/atomic"

	"go.uber.org/atomic"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

// terminationLoggingListener wraps the given listener to mark late connections
// as such, identified by the remote address. In parallel, we have a filter that
// logs bad requests through these connections. We need this filter to get
// access to the http path in order to filter out healthz or readyz probes that
// are allowed at any point during termination.
//
// Connections are late after the lateStopCh has been closed.
type terminationLoggingListener struct {
	net.Listener
	lateStopCh <-chan struct{}
}

type eventfFunc func(eventType, reason, messageFmt string, args ...interface{})

var (
	lateConnectionRemoteAddrsLock sync.RWMutex
	lateConnectionRemoteAddrs     = map[string]bool{}

	unexpectedRequestsEventf goatomic.Value
)

func (l *terminationLoggingListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}

	select {
	case <-l.lateStopCh:
		lateConnectionRemoteAddrsLock.Lock()
		defer lateConnectionRemoteAddrsLock.Unlock()
		lateConnectionRemoteAddrs[c.RemoteAddr().String()] = true
	default:
	}

	return c, nil
}

// WithLateConnectionFilter logs every non-probe request that comes through a late connection identified by remote address.
func WithLateConnectionFilter(handler http.Handler) http.Handler {
	var lateRequestReceived atomic.Bool

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lateConnectionRemoteAddrsLock.RLock()
		late := lateConnectionRemoteAddrs[r.RemoteAddr]
		lateConnectionRemoteAddrsLock.RUnlock()

		if late {
			if pth := "/" + strings.TrimLeft(r.URL.Path, "/"); pth != "/readyz" && pth != "/healthz" && pth != "/livez" {
				if isLocal(r) {
					klog.V(4).Infof("Loopback request to %q (user agent %q) through connection created very late in the graceful termination process (more than 80%% has passed). This client probably does not watch /readyz and might get failures when termination is over.", r.URL.Path, r.UserAgent())
				} else {
					klog.Warningf("Request to %q (source IP %s, user agent %q) through a connection created very late in the graceful termination process (more than 80%% has passed), possibly a sign for a broken load balancer setup.", r.URL.Path, r.RemoteAddr, r.UserAgent())

					// create only one event to avoid event spam.
					var eventf eventfFunc
					eventf, _ = unexpectedRequestsEventf.Load().(eventfFunc)
					if swapped := lateRequestReceived.CAS(false, true); swapped && eventf != nil {
						eventf(corev1.EventTypeWarning, "LateConnections", "The apiserver received connections (e.g. from %q, user agent %q) very late in the graceful termination process, possibly a sign for a broken load balancer setup.", r.RemoteAddr, r.UserAgent())
					}
				}
			}
		}

		handler.ServeHTTP(w, r)
	})
}

// WithNonReadyRequestLogging rejects the request until the process has been ready once.
func WithNonReadyRequestLogging(handler http.Handler, hasBeenReadySignal lifecycleSignal) http.Handler {
	if hasBeenReadySignal == nil {
		return handler
	}

	var nonReadyRequestReceived atomic.Bool

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-hasBeenReadySignal.Signaled():
			handler.ServeHTTP(w, r)
			return
		default:
		}

		// ignore connections to local IP. Those clients better know what they are doing.
		if pth := "/" + strings.TrimLeft(r.URL.Path, "/"); pth != "/readyz" && pth != "/healthz" && pth != "/livez" {
			if isLocal(r) {
				if !isKubeApiserverLoopBack(r) {
					klog.V(2).Infof("Loopback request to %q (user agent %q) before server is ready. This client probably does not watch /readyz and might get inconsistent answers.", r.URL.Path, r.UserAgent())
				}
			} else {
				klog.Warningf("Request to %q (source IP %s, user agent %q) before server is ready, possibly a sign for a broken load balancer setup.", r.URL.Path, r.RemoteAddr, r.UserAgent())

				// create only one event to avoid event spam.
				var eventf eventfFunc
				eventf, _ = unexpectedRequestsEventf.Load().(eventfFunc)
				if swapped := nonReadyRequestReceived.CAS(false, true); swapped && eventf != nil {
					eventf(corev1.EventTypeWarning, "NonReadyRequests", "The kube-apiserver received requests (e.g. from %q, user agent %q, accessing %s) before it was ready, possibly a sign for a broken load balancer setup.", r.RemoteAddr, r.UserAgent(), r.URL.Path)
				}
			}
		}

		handler.ServeHTTP(w, r)
	})
}

func isLocal(req *http.Request) bool {
	host, _, err := net.SplitHostPort(req.RemoteAddr)
	if err != nil {
		// ignore error and keep going
	} else if ip := netutils.ParseIPSloppy(host); ip != nil {
		return ip.IsLoopback()
	}

	return false
}

func isKubeApiserverLoopBack(req *http.Request) bool {
	return strings.HasPrefix(req.UserAgent(), "kube-apiserver/")
}
