/*
Copyright 2019 The Kubernetes Authors.

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
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"k8s.io/klog"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"k8s.io/apiserver/pkg/apis/apiserver"
)

type ContextDialer func(ctx context.Context, network, addr string) (net.Conn, error)

var m map[string]ContextDialer = nil
var mu sync.Mutex

var directDialer ContextDialer =  (&net.Dialer{
	Timeout:   30 * time.Second,
	KeepAlive: 30 * time.Second,
	DualStack: true,
}).DialContext

type NetworkContext struct {
	// ConnectivityServiceName is the unique name of the
	// ConnectivityServiceConfiguration which determines
	// the network we route the traffic to.
	ConnectivityServiceName string
}

func createConnectDialer(connection apiserver.Connection) (ContextDialer, error) {
	clientCert := connection.ClientCertFile
	clientKey := connection.ClientKeyFile
	caCert := connection.CABundle
	proxyURL, err := url.Parse(connection.URL)
	if err != nil {
		return nil, fmt.Errorf("invalid proxy server url %q: %v", connection.URL, err)
	}
	proxyAddress := proxyURL.Host

	clientCerts, err := tls.LoadX509KeyPair(clientCert, clientKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read key pair %s & %s, got %v", clientCert, clientKey, err)
	}
	certPool := x509.NewCertPool()
	certBytes, err := ioutil.ReadFile(caCert)
	if err != nil {
		return nil, fmt.Errorf("failed to read cert file %s, got %v", caCert, err)
	}
	ok := certPool.AppendCertsFromPEM(certBytes)
	if !ok {
		return nil, fmt.Errorf("failed to append CA cert to the cert pool")
	}
	contextDialer := func(ctx context.Context, network, addr string) (net.Conn, error) {
		klog.Warningf("Sending request to %q.", addr)
		proxyConn, err := tls.Dial("tcp", proxyAddress,
			&tls.Config{
				ServerName:   "kubernetes-master",
				Certificates: []tls.Certificate{clientCerts},
				RootCAs:      certPool,
			},
		)
		if err != nil {
			return nil, fmt.Errorf("dialing proxy %q failed: %v", proxyAddress, err)
		}
		fmt.Fprintf(proxyConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", addr, "127.0.0.1")
		br := bufio.NewReader(proxyConn)
		res, err := http.ReadResponse(br, nil)
		if err != nil {
			return nil, fmt.Errorf("reading HTTP response from CONNECT to %s via proxy %s failed: %v",
				addr, proxyAddress, err)
		}
		if res.StatusCode != 200 {
			return nil, fmt.Errorf("proxy error from %s while dialing %s: %v", proxyAddress, addr, res.Status)
		}

		// It's safe to discard the bufio.Reader here and return the
		// original TCP conn directly because we only use this for
		// TLS, and in TLS the client speaks first, so we know there's
		// no unbuffered data. But we can double-check.
		if br.Buffered() > 0 {
			return nil, fmt.Errorf("unexpected %d bytes of buffered data from CONNECT proxy %q",
				br.Buffered(), proxyAddress)
		}
		klog.Infof("About to proxy request to %s over %s.", addr, proxyAddress)
		return proxyConn, nil
	}
	return contextDialer, nil
}

func SetupConnectivityService(config *apiserver.ConnectivityServiceConfiguration) error {
	mu.Lock()
	defer mu.Unlock()
	if m != nil {
		return fmt.Errorf("attempt to reinitialize connectivity service")
	}
	serviceMap := make(map[string]ContextDialer)
	for _, service := range config.ConnectionServices {
		name := service.Name
		switch service.Connection.Type {
		case "http-connect":
			contextDialer, err := createConnectDialer(service.Connection)
			if err != nil {
				return fmt.Errorf("failed to create http-connect dialer: %v", err)
			}
			serviceMap[name] = contextDialer
		case "direct":
			serviceMap[name] = directDialer
		default:
			return fmt.Errorf("unrecognized service connection type %q", service.Connection.Type)
		}
	}
	m = serviceMap
	return nil
}

func getServiceMap() map[string]ContextDialer {
	mu.Lock()
	defer mu.Unlock()
	return m
}

func Lookup(networkContext NetworkContext) (ContextDialer, error) {
	serviceMap := getServiceMap()
	if serviceMap == nil {
		return directDialer, nil
	}
	return serviceMap[networkContext.ConnectivityServiceName], nil
}

