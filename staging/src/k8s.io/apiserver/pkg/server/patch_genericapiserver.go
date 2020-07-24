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

	"go.uber.org/atomic"

	"k8s.io/klog"

	corev1 "k8s.io/api/core/v1"
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

var (
	lateConnectionRemoteAddrsLock sync.RWMutex
	lateConnectionRemoteAddrs     map[string]bool = map[string]bool{}
	lateConnectionEventf          func(eventType, reason, messageFmt string, args ...interface{})
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
			// ignore connections to local IP. Those clients better know what they are doing.
			local := false
			host, _, err := net.SplitHostPort(r.RemoteAddr)
			if err != nil {
				// ignore error and keep going
			} else if ip := net.ParseIP(host); ip != nil {
				local = ip.IsLoopback()
			}

			if pth := "/" + strings.TrimLeft(r.URL.Path, "/"); pth != "/readyz" && pth != "/healthz" {
				if local {
					klog.V(4).Infof("Request from loopback client %s to %q (user agent %q) through connection created very late in the graceful termination process (more than 80%% has passed). This client probably does not watch /readyz and might get failures when termination is over.", r.RemoteAddr, r.URL.Path, r.UserAgent())
				} else {
					klog.Warningf("Request from %s to %q (user agent %q) through connection created very late in the graceful termination process (more than 80%% has passed), possibly a sign for a broken load balancer setup.", r.RemoteAddr, r.URL.Path, r.UserAgent())

					// create only one event to avoid event spam.
					if swapped := lateRequestReceived.CAS(false, true); swapped && lateConnectionEventf != nil {
						lateConnectionEventf(corev1.EventTypeWarning, "LateConnections", "The apiserver received connections (e.g. from %q, user agent %q) very late in the graceful termination process, possibly a sign for a broken load balancer setup.", r.RemoteAddr, r.UserAgent())
					}
				}
			}
		}

		handler.ServeHTTP(w, r)
	})
}
