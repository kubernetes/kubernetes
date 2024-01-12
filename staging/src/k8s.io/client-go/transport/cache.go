/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/metrics"
	"k8s.io/klog/v2"
)

// TlsTransportCache caches TLS http.RoundTrippers different configurations. The
// same RoundTripper will be returned for configs with identical TLS options If
// the config has no custom TLS options, http.DefaultTransport is returned.
type tlsTransportCache struct {
	mu         sync.Mutex
	transports map[tlsCacheKey]*reloadableTransport
}

// DialerStopCh is stop channel that is passed down to dynamic cert dialer.
// It's exposed as variable for testing purposes to avoid testing for goroutine
// leakages.
var DialerStopCh = wait.NeverStop

const idleConnsPerHost = 25

var tlsCache = &tlsTransportCache{transports: make(map[tlsCacheKey]*reloadableTransport)}

type tlsCacheKey struct {
	insecure           bool
	caData             string
	certData           string
	keyData            string `datapolicy:"security-key"`
	certFile           string
	keyFile            string
	serverName         string
	nextProtos         string
	disableCompression bool
	// these functions are wrapped to allow them to be used as map keys
	getCert *GetCertHolder
	dial    *DialHolder
}

var _ utilnet.RoundTripperWrapper = &reloadableTransport{}

type reloadableTransport struct {
	rt *atomic.Pointer[http.Transport]
}

func (r *reloadableTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return r.rt.Load().RoundTrip(req)
}

func (r *reloadableTransport) WrappedRoundTripper() http.RoundTripper {
	return r.rt.Load()
}

func (t tlsCacheKey) String() string {
	keyText := "<none>"
	if len(t.keyData) > 0 {
		keyText = "<redacted>"
	}
	return fmt.Sprintf("insecure:%v, caData:%#v, certData:%#v, keyData:%s, serverName:%s, disableCompression:%t, getCert:%p, dial:%p",
		t.insecure, t.caData, t.certData, keyText, t.serverName, t.disableCompression, t.getCert, t.dial)
}

func (c *tlsTransportCache) get(config *Config) (http.RoundTripper, error) {
	key, canCache, err := tlsConfigKey(config)
	if err != nil {
		return nil, err
	}

	if canCache {
		// Ensure we only create a single transport for the given TLS options
		c.mu.Lock()
		defer c.mu.Unlock()
		defer metrics.TransportCacheEntries.Observe(len(c.transports))

		// See if we already have a custom transport for this config
		if t, ok := c.transports[key]; ok {
			metrics.TransportCreateCalls.Increment("hit")
			return t, nil
		}
		metrics.TransportCreateCalls.Increment("miss")
	} else {
		metrics.TransportCreateCalls.Increment("uncacheable")
	}

	// check here because TLSConfigFor likes to mutate the config
	shouldReloadCA := len(config.TLS.CAData) == 0 && len(config.TLS.CAFile) > 0

	// Get the TLS options for this client config
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		return nil, err
	}
	// The options didn't require a custom TLS config
	if tlsConfig == nil && config.DialHolder == nil && config.Proxy == nil {
		return http.DefaultTransport, nil
	}

	var dial func(ctx context.Context, network, address string) (net.Conn, error)
	if config.DialHolder != nil {
		dial = config.DialHolder.Dial
	} else {
		dial = (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext
	}

	// If we use are reloading files, we need to handle certificate rotation properly
	// TODO(jackkleeman): We can also add rotation here when config.HasCertCallback() is true
	if config.TLS.ReloadTLSFiles && tlsConfig != nil && tlsConfig.GetClientCertificate != nil {
		dynamicCertDialer := certRotatingDialer(tlsConfig.GetClientCertificate, dial)
		tlsConfig.GetClientCertificate = dynamicCertDialer.GetClientCertificate
		dial = dynamicCertDialer.connDialer.DialContext
		go dynamicCertDialer.Run(DialerStopCh)
	}

	proxy := http.ProxyFromEnvironment
	if config.Proxy != nil {
		proxy = config.Proxy
	}

	transport := buildReloadableTransport(utilnet.SetTransportDefaults(&http.Transport{
		Proxy:               proxy,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		MaxIdleConnsPerHost: idleConnsPerHost,
		DialContext:         dial,
		DisableCompression:  config.DisableCompression,
	}))

	if shouldReloadCA {
		caFile := config.TLS.CAFile
		caData := config.TLS.CAData
		baseRT := transport.rt.Load().Clone()
		go wait.UntilWithContext(wait.ContextForChannel(DialerStopCh), func(_ context.Context) {
			data, err := os.ReadFile(caFile)
			if err != nil {
				klog.ErrorS(err, "failed to read CA file", "file", caFile)
				return
			}

			if len(data) == 0 {
				klog.InfoS("CA file is empty", "file", caFile)
				return
			}

			if bytes.Equal(data, caData) {
				return
			}

			rootCAs, err := rootCertPool(data)
			if err != nil {
				klog.ErrorS(err, "failed to build pool from CA file", "file", caFile)
				return
			}

			newRT := baseRT.Clone()
			newRT.TLSClientConfig.RootCAs = rootCAs

			transport.rt.Store(newRT)
		}, time.Hour)
	}

	if canCache {
		// Cache a single transport for these options
		c.transports[key] = transport
	}

	return transport, nil
}

// tlsConfigKey returns a unique key for tls.Config objects returned from TLSConfigFor
func tlsConfigKey(c *Config) (tlsCacheKey, bool, error) {
	// Make sure ca/key/cert content is loaded
	if err := loadTLSFiles(c); err != nil {
		return tlsCacheKey{}, false, err
	}

	if c.Proxy != nil {
		// cannot determine equality for functions
		return tlsCacheKey{}, false, nil
	}

	k := tlsCacheKey{
		insecure:           c.TLS.Insecure,
		caData:             string(c.TLS.CAData),
		serverName:         c.TLS.ServerName,
		nextProtos:         strings.Join(c.TLS.NextProtos, ","),
		disableCompression: c.DisableCompression,
		getCert:            c.TLS.GetCertHolder,
		dial:               c.DialHolder,
	}

	if c.TLS.ReloadTLSFiles {
		k.certFile = c.TLS.CertFile
		k.keyFile = c.TLS.KeyFile
	} else {
		k.certData = string(c.TLS.CertData)
		k.keyData = string(c.TLS.KeyData)
	}

	return k, true, nil
}

func buildReloadableTransport(rt *http.Transport) *reloadableTransport {
	out := &reloadableTransport{rt: &atomic.Pointer[http.Transport]{}}
	out.rt.Store(rt)
	return out
}
