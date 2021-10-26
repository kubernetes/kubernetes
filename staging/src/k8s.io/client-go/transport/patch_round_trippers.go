/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

const (
	blockPeriod              = 20 * time.Second     // time for the reconcile loop to remove and unready endpoint
	defaultKubernetesService = "kubernetes.default" // API server Service/DNS name
)

// alternative contains the details of an alternative service
type alternative struct {
	local     bool      // the alternative service address is present on the system
	blocked   bool      // the alternative service is blocked
	blockedTs time.Time // timestamp the alternative service was blocked (zero means forever)
}

// alternativeServiceRoundTripper is a round tripper that allow to use alternative services exposed by the api server
// The alternative services exposed are the Endpoints published for the kubernetes.default service.
type alternativeServiceRoundTripper struct {
	mu    sync.Mutex
	cache map[string]alternative // cache with the alternative services keyed by host[:port]
	last  string                 // last connected host so it sticks to it

	once           sync.Once // used to set the TLS ServerName on the transport only once
	serverName     string    // TLS ServerName for the alternative services
	allowLocalhost bool      // allow to use alternative services when using localhost
	clock          clock.Clock

	rt http.RoundTripper // wrapped round tripper
}

type AlternativeServicesOptions func(*alternativeServiceRoundTripper) *alternativeServiceRoundTripper

// WithAlternativeServices preseed the cache with alternatives services
func WithAlternativeServices(hosts []string) AlternativeServicesOptions {
	return func(a *alternativeServiceRoundTripper) *alternativeServiceRoundTripper {
		a.addAlternativeServices(hosts)
		return a
	}
}

// WithAlternativeServerName sets the TLS ServerName to be used for Alternative Services
func WithAlternativeServerName(serverName string) AlternativeServicesOptions {
	return func(a *alternativeServiceRoundTripper) *alternativeServiceRoundTripper {
		a.serverName = serverName
		return a
	}
}

// WithLocalhostAllowed allows to use alternative services if the destination host is localhost (disabled by default)
func WithLocalhostAllowed() AlternativeServicesOptions {
	return func(a *alternativeServiceRoundTripper) *alternativeServiceRoundTripper {
		a.allowLocalhost = true
		return a
	}
}

// NewAlternativeServiceRoundTripper returns a roundtipper that allows to connect to the Kubernetes
// api server, providing client side HA using Alt-Svc headers.
func NewAlternativeServiceRoundTripper(rt http.RoundTripper) http.RoundTripper {
	return &alternativeServiceRoundTripper{
		cache:      make(map[string]alternative),
		clock:      clock.RealClock{},
		serverName: defaultKubernetesService,
		rt:         rt,
	}
}

// NewAlternativeServiceRoundTripperWithOptions returns a roundtipper that allows to connect to the Kubernetes
// api server, providing client side HA using Alt-Svc headers. Allows to pass options.
func NewAlternativeServiceRoundTripperWithOptions(rt http.RoundTripper, options ...AlternativeServicesOptions) http.RoundTripper {
	a := &alternativeServiceRoundTripper{
		cache:      make(map[string]alternative),
		clock:      clock.RealClock{},
		serverName: defaultKubernetesService,
		rt:         rt,
	}
	for _, opt := range options {
		a = opt(a)
	}
	return a
}

func (rt *alternativeServiceRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	req := utilnet.CloneRequest(request)
	hostOrig := req.URL.Host
	originalTLSServerName := ""

	// https only, alternative services can be exploited to hijack services.
	if req.URL == nil || req.URL.Scheme != "https" {
		return rt.rt.RoundTrip(req)
	}

	// avoid loops, don't process request that are already using an alternative service.
	if len(req.Header.Get("Alt-Used")) > 0 {
		return rt.rt.RoundTrip(req)
	}

	if !rt.allowLocalhost {
		hostname := req.URL.Hostname()
		// localhost addresses doesn't use alternative services
		if netutils.ParseIPSloppy(hostname).IsLoopback() || hostname == "localhost" {
			return rt.rt.RoundTrip(req)
		}
	}

	// retry failed connections against alternative services
retry:
	// check if there is a preferred alternative service and use it
	host := rt.getAltSvc(hostOrig)
	alternativeService := host != hostOrig
	// if is not an alternative service round trip the original request directly
	// otherwise, mutate the destination host, the Host header and the SNI ServerName
	if !alternativeService {
		req = utilnet.CloneRequest(request)
		// clear last host connected
		rt.resetLast()
	} else {
		klog.V(2).InfoS("Using alternative service", "alternative", host, "host", hostOrig)
		// replace the destination host
		req.URL.Host = host
		// set the host header to the corresponding ServerName
		req.Host = rt.serverName
		// RFC7838 Section 5 to avoid loops
		req.Header.Set("Alt-Used", host)
		// use kubernetes.default as ServerName for SNI authentication
		// This is the same used for InCluster configuration.
		rt.once.Do(func() {
			tlsCfg, err := utilnet.TLSClientConfig(rt.rt)
			if err == nil {
				originalTLSServerName = tlsCfg.ServerName
				klog.V(2).InfoS("Setting SNI ServerName", "oldServerName", originalTLSServerName, "newServerName", rt.serverName)
				tlsCfg.ServerName = rt.serverName
			}
		})
	}
	// RoundTrip the request
	resp, err := rt.rt.RoundTrip(req)
	// Error handling
	if err != nil {
		// clear last host connected
		rt.resetLast()
		// return the error if the connection was against the original host
		if !alternativeService {
			return resp, err
		}
		klog.ErrorS(err, "Error using alternative service", "alternative", host, "host", hostOrig)
		// Network errors will block the host, so it won't be retried during the blockPeriod.
		if isNetworkError(err) {
			klog.ErrorS(err, "Network error using alternative service, blocking alternative service", "alternative", host, "blockPeriod", blockPeriod)
			rt.blockHost(host, false)
			goto retry
		}
		// Certificate errors will block the host forever and restore the ServerName.
		if isCertError(err) {
			klog.ErrorS(err, "Certificate error using alternative service, blocking alternative service forever", "alternative", host)
			rt.blockHost(host, true)
			tlsCfg, err := utilnet.TLSClientConfig(rt.rt)
			if err == nil && tlsCfg.ServerName == rt.serverName {
				tlsCfg.ServerName = originalTLSServerName
			}
			goto retry
		}
		// just return for other errors
		return resp, err
	}
	// connection succeed, use the same host next time
	if alternativeService {
		rt.setLast(host)
	}
	// process the alt-svc header to update the cache with the alternative services
	altSvc := resp.Header.Get("Alt-Svc")
	if len(altSvc) > 0 {
		klog.V(4).InfoS("Alternative services found", "Alt-Svc", altSvc)
		alternativeServices, err := parseAltSvcHeader(altSvc, host)
		if err != nil {
			klog.ErrorS(err, "Error parsing Alt-Svc header", "Alt-Svc", alternativeServices)
		} else {
			rt.addAlternativeServices(alternativeServices)
		}
	}
	return resp, err
}

func (rt *alternativeServiceRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *alternativeServiceRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// getAltSvc returns an alternative host, if exist, otherwise it returns the same host
// The logic is as follow:
// - empty cache does nothing
// - if we have used a host from the cache previously, keep using the same host
// - if there is a host in the cache that is in the same host, then use it
// - if the original requested host is in the cache, use it
// - use any available host from the cache
func (rt *alternativeServiceRoundTripper) getAltSvc(host string) string {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	// sticky
	if len(rt.last) > 0 {
		return rt.last
	}
	// iterate over the cache
	now := rt.clock.Now()
	hosts := []string{}
	for h, entry := range rt.cache {
		// check if the entry is blocked
		if entry.blocked {
			if entry.blockedTs.IsZero() ||
				now.Sub(entry.blockedTs) < blockPeriod {
				continue
			}
			// unblock it if necessary
			entry.blocked = false
			entry.blockedTs = time.Time{}
			rt.cache[h] = entry
		}
		// prefer local addresses
		if entry.local {
			return h
		} else {
			hosts = append(hosts, h)
		}
	}
	// no hosts available, reset and use the original one
	if len(hosts) == 0 {
		return host
	}
	// prefer the requested host
	if contains(hosts, host) {
		return host
	}
	return hosts[0]
}

func contains(list []string, element string) bool {
	if len(list) == 0 {
		return false
	}
	for _, x := range list {
		if x == element {
			return true
		}
	}
	return false
}

// resetLast resets the last connected host
func (rt *alternativeServiceRoundTripper) resetLast() {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.last = ""
}

// resetLast resets the last connected host
func (rt *alternativeServiceRoundTripper) setLast(host string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.last = host
}

// addAlternativeServices updates the cache with a new list of alternative services
// checking if they are local. Cache is completely replaced with the new entries.
func (rt *alternativeServiceRoundTripper) addAlternativeServices(hosts []string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	newCache := map[string]alternative{}
	for _, h := range hosts {
		a := alternative{}
		// if host already exists keep the information
		if v, ok := rt.cache[h]; ok {
			a = v
		} else {
			a.local = isLocal(h)
		}
		newCache[h] = a
	}
	rt.cache = newCache
}

// blockHost sets the host information to blocked and adds a timestamp.
// A host can be blocked forever using the zero timestamp.
func (rt *alternativeServiceRoundTripper) blockHost(host string, forever bool) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	timestamp := rt.clock.Now()
	if forever {
		// zero means forever
		timestamp = time.Time{}
	}
	if v, ok := rt.cache[host]; ok {
		v.blocked = true
		v.blockedTs = timestamp
		rt.cache[host] = v
	}
}

// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }

// isLocal checks if the host is local, by checking if the IP address is present in the host.
func isLocal(urlHost string) bool {
	var host string
	var err error
	if hasPort(urlHost) {
		host, _, err = net.SplitHostPort(urlHost)
		if err != nil {
			return false
		}
	} else {
		host = urlHost
	}

	ips, err := net.LookupIP(host)
	if err != nil {
		return false
	}
	localIPs := getLocalAddressSet()
	for _, ip := range ips {
		if localIPs.Has(ip) {
			return true
		}
	}
	return false
}

// getLocalAddrs returns a set with all network addresses on the local system.
func getLocalAddressSet() netutils.IPSet {
	localAddrs := netutils.IPSet{}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		klog.ErrorS(err, "Error getting local addresses")
		return localAddrs
	}

	for _, addr := range addrs {
		ip, _, err := netutils.ParseCIDRSloppy(addr.String())
		if err != nil {
			klog.ErrorS(err, "Error getting local addresses", "address", addr.String())
			continue
		}
		localAddrs.Insert(ip)
	}
	return localAddrs
}

// parseAltSvcHeader parses an Alt-Svc header and returns the list of alternative services in format host[:port]
// Alt-Svc
// RFC 7838
//    Alt-Svc       = clear / 1#alt-value
// clear         = %s"clear"; "clear", case-sensitive
// alt-value     = alternative *( OWS ";" OWS parameter )
// alternative   = protocol-id "=" alt-authority
// protocol-id   = token ; percent-encoded ALPN protocol name
// alt-authority = quoted-string ; containing [ uri-host ] ":" port
// parameter     = token "=" ( token / quoted-string )
// Caching parameters:
// ma 			 = delta-seconds
// persist       = not clear on network changes
func parseAltSvcHeader(header, origHost string) ([]string, error) {
	if len(header) == 0 {
		return []string{}, nil
	}
	// tolerate whitespaces
	header = strings.TrimSpace(header)
	if header == "clear" {
		return []string{}, nil
	}

	var errors []error
	// comma separated list of alternative services
	alternatives := strings.Split(header, ",")
	hosts := make([]string, len(alternatives))
	for i, a := range alternatives {
		// semi colon separated list of options per alternative service
		alternative := strings.Split(a, ";")
		if len(alternative) == 0 {
			errors = append(errors, fmt.Errorf("no alternative service present"))
			continue
		}
		// Process first entry
		// alternative   = protocol-id "=" alt-authority
		h := strings.Split(strings.TrimSpace(alternative[0]), "=")
		if len(h) != 2 {
			errors = append(errors, fmt.Errorf("error parsing alternative service %s", alternative))
			continue
		}
		// only support http2
		if h[0] != "h2" {
			errors = append(errors, fmt.Errorf("unsupported protocol %s", h[0]))
			continue
		}

		// validate alt-authority (it is a quoted string)
		authority := strings.Trim(h[1], "\"")
		host, port, err := net.SplitHostPort(authority)
		if err != nil {
			errors = append(errors, err)
			continue
		}
		// Alt-Svc returns an empty host for the service we are connecting against
		if host == "" {
			if origHost == "" {
				return []string{}, fmt.Errorf("missing original host, can't obtain alternative service")
			}
			host = origHost
		}
		hosts[i] = net.JoinHostPort(host, port)
		// TODO the rest of the options in an Alt-Svc header are related to caching
		// that doesn't really apply in this environment
	}
	if len(errors) > 0 {
		return []string{}, utilerrors.NewAggregate(errors)
	}
	return hosts, nil
}

// isNetworkError return true if the error is a network error.
func isNetworkError(err error) bool {
	_, ok := err.(net.Error)
	if ok {
		return ok
	}
	return isIdleError(err) || isRequestCanceled(err)
}

// isCertError return true if the error is because the certificate is invalid.
func isCertError(err error) bool {
	return strings.Contains(err.Error(), "x509: certificate")
}

// isIdleError return true if the error is because the connection was lost.
func isIdleError(err error) bool {
	return strings.Contains(err.Error(), "http2: client connection lost")
}

// isIdleError return true if the error is because the connection was lost.
func isRequestCanceled(err error) bool {
	return strings.Contains(err.Error(), "net/http: request canceled")
}
