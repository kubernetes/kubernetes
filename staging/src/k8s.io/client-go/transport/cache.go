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
	"fmt"
	"net"
	"net/http"
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

const idleConnsPerHost = 25

var tlsCache = &tlsTransportCache{transports: make(map[tlsCacheKey]*http.Transport)}

type tlsCacheKey struct {
	insecure           bool
	caData             string
	certData           string
	keyData            string
	certFile           string
	keyFile            string
	getCert            string
	serverName         string
	nextProtos         string
	dial               string
	disableCompression bool
	proxy              string
}

func (t tlsCacheKey) String() string {
	keyText := "<none>"
	if len(t.keyData) > 0 {
		keyText = "<redacted>"
	}
	return fmt.Sprintf("insecure:%v, caData:%#v, certData:%#v, keyData:%s, getCert: %s, serverName:%s, dial:%s disableCompression:%t, proxy: %s", t.insecure, t.caData, t.certData, keyText, t.getCert, t.serverName, t.dial, t.disableCompression, t.proxy)
}

func (c *tlsTransportCache) get(config *Config) (http.RoundTripper, error) {
	key, err := tlsConfigKey(config)
	if err != nil {
		return nil, err
	}

	// Ensure we only create a single transport for the given TLS options
	c.mu.Lock()
	defer c.mu.Unlock()

	// See if we already have a custom transport for this config
	if t, ok := c.transports[key]; ok {
		return t, nil
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

	dial := config.Dial
	if dial == nil {
		dial = (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext
	}

	// If we use are reloading files, we need to handle certificate rotation properly
	// TODO(jackkleeman): We can also add rotation here when config.HasCertCallback() is true
	if config.TLS.ReloadTLSFiles {
		dynamicCertDialer := certRotatingDialer(tlsConfig.GetClientCertificate, dial)
		tlsConfig.GetClientCertificate = dynamicCertDialer.GetClientCertificate
		dial = dynamicCertDialer.connDialer.DialContext
		go dynamicCertDialer.Run(wait.NeverStop)
	}

	proxy := http.ProxyFromEnvironment
	if config.Proxy != nil {
		proxy = config.Proxy
	}

	// Cache a single transport for these options
	c.transports[key] = utilnet.SetTransportDefaults(&http.Transport{
		Proxy:               proxy,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		MaxIdleConnsPerHost: idleConnsPerHost,
		DialContext:         dial,
		DisableCompression:  config.DisableCompression,
	})
	return c.transports[key], nil
}

// tlsConfigKey returns a unique key for tls.Config objects returned from TLSConfigFor
func tlsConfigKey(c *Config) (tlsCacheKey, error) {
	// Make sure ca/key/cert content is loaded
	if err := loadTLSFiles(c); err != nil {
		return tlsCacheKey{}, err
	}
	k := tlsCacheKey{
		insecure:           c.TLS.Insecure,
		caData:             string(c.TLS.CAData),
		getCert:            fmt.Sprintf("%p", c.TLS.GetCert),
		serverName:         c.TLS.ServerName,
		nextProtos:         strings.Join(c.TLS.NextProtos, ","),
		dial:               fmt.Sprintf("%p", c.Dial),
		disableCompression: c.DisableCompression,
		proxy:              fmt.Sprintf("%p", c.Proxy),
	}

	if c.TLS.ReloadTLSFiles {
		k.certFile = c.TLS.CertFile
		k.keyFile = c.TLS.KeyFile
	} else {
		k.certData = string(c.TLS.CertData)
		k.keyData = string(c.TLS.KeyData)
	}

	return k, nil
}
