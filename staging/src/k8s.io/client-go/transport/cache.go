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
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
)

// TlsTransportCache caches TLS http.RoundTrippers different configurations. The
// same RoundTripper will be returned for configs with identical TLS options If
// the config has no custom TLS options, http.DefaultTransport is returned.
type tlsTransportCache struct {
	mu         sync.Mutex
	transports map[tlsCacheKey]*http.Transport
}

// DialerStopCh is stop channel that is passed down to dynamic cert dialer.
// It's exposed as variable for testing purposes to avoid testing for goroutine
// leakages.
var DialerStopCh = wait.NeverStop

const idleConnsPerHost = 25

var tlsCache = &tlsTransportCache{transports: make(map[tlsCacheKey]*http.Transport)}

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
	dialer             Dialer
	cert               Certificate
	proxier            Proxier
}

func (t tlsCacheKey) String() string {
	keyText := "<none>"
	if len(t.keyData) > 0 {
		keyText = "<redacted>"
	}
	return fmt.Sprintf("insecure:%v, caData:%#v, certData:%#v, keyData:%s, serverName:%s, disableCompression:%t dialer:%#v cert:%#v proxier:%#v", t.insecure, t.caData, t.certData, keyText, t.serverName, t.disableCompression, t.dialer, t.cert, t.proxier)
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

		// See if we already have a custom transport for this config
		if t, ok := c.transports[key]; ok {
			return t, nil
		}
	}

	// Get the TLS options for this client config
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		return nil, err
	}
	// The options didn't require a custom TLS config
	if tlsConfig == nil && config.Dial == nil && config.Proxy == nil {
		return http.DefaultTransport, nil
	}

	var dialFunc func(ctx context.Context, network, address string) (net.Conn, error)
	if config.Dialer != nil {
		dialFunc = config.Dialer.DialContext
	} else if config.Dial != nil {
		dialFunc = config.Dial
	} else {
		dialFunc = (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext
	}

	// If we use are reloading files, we need to handle certificate rotation properly
	// TODO(jackkleeman): We can also add rotation here when config.HasCertCallback() is true
	if config.TLS.ReloadTLSFiles {
		dynamicCertDialer := certRotatingDialer(tlsConfig.GetClientCertificate, dialFunc)
		tlsConfig.GetClientCertificate = dynamicCertDialer.GetClientCertificate
		dialFunc = dynamicCertDialer.connDialer.DialContext
		go dynamicCertDialer.Run(DialerStopCh)
	}

	var proxyFunc func(*http.Request) (*url.URL, error)
	if config.Proxier != nil {
		proxyFunc = config.Proxier.Proxy
	} else if config.Proxy != nil {
		proxyFunc = config.Proxy
	} else {
		proxyFunc = http.ProxyFromEnvironment
	}

	transport := utilnet.SetTransportDefaults(&http.Transport{
		Proxy:               proxyFunc,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		MaxIdleConnsPerHost: idleConnsPerHost,
		DialContext:         dialFunc,
		DisableCompression:  config.DisableCompression,
	})

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

	if (c.TLS.GetCert != nil && c.TLS.Cert == nil) || (c.Dial != nil && c.Dialer == nil) || (c.Proxy != nil && c.Proxier == nil) {
		// cannot determine equality for functions
		return tlsCacheKey{}, false, nil
	}

	k := tlsCacheKey{
		insecure:           c.TLS.Insecure,
		caData:             string(c.TLS.CAData),
		serverName:         c.TLS.ServerName,
		nextProtos:         strings.Join(c.TLS.NextProtos, ","),
		disableCompression: c.DisableCompression,
		dialer:             c.Dialer,
		cert:               c.TLS.Cert,
		proxier:            c.Proxier,
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
