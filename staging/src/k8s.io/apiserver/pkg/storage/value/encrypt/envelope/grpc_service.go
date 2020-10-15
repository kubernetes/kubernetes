/*
Copyright 2017 The Kubernetes Authors.

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

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
	"k8s.io/klog/v2"
)

const (
	// Now supported unix domain socket and TCP connection with TLS credentials.
	// If endpoint's scheme is unix, then no TLS credentials will be used.
	unixProtocol = "unix"

	// Current version for the protocol interface definition.
	kmsapiVersion = "v1beta1"

	versionErrorf = "KMS provider api version %s is not supported, only %s is supported now"
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	kmsClient      kmsapi.KeyManagementServiceClient
	connection     *grpc.ClientConn
	callTimeout    time.Duration
	mux            sync.RWMutex
	versionChecked bool
}

// NewGRPCService returns an envelope.Service which use gRPC to communicate the remote KMS provider.
func NewGRPCService(endpoint string, callTimeout time.Duration, tlsConfig *apiserverconfig.KMSTLSClientConfig) (Service, error) {
	klog.V(4).Infof("Configure KMS provider with endpoint: %s", endpoint)

	var err error
	s := &gRPCService{callTimeout: callTimeout}
	if tlsConfig == nil {
		s.connection, err = s.unixSocketConnection(endpoint)
	} else {
		s.connection, err = s.tlsConnection(endpoint, tlsConfig)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, tlsConfig: %v, error: %v", endpoint, tlsConfig, err)
	}

	s.kmsClient = kmsapi.NewKeyManagementServiceClient(s.connection)
	return s, nil
}

// unixSocketConnection returns unix socket gRPC connection according to the given unix socket endpoint.
func (g *gRPCService) unixSocketConnection(endpoint string) (*grpc.ClientConn, error) {
	addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	conn, err := grpc.Dial(
		addr,
		grpc.WithInsecure(),
		grpc.WithUnaryInterceptor(g.interceptor),
		grpc.WithDefaultCallOptions(grpc.WaitForReady(true)),
		grpc.WithContextDialer(
			func(context.Context, string) (net.Conn, error) {
				// Ignoring addr and timeout arguments:
				// addr - comes from the closure
				c, err := net.DialUnix(unixProtocol, nil, &net.UnixAddr{Name: addr})
				if err != nil {
					klog.Errorf("failed to create connection to unix socket: %s, error: %v", addr, err)
				} else {
					klog.V(4).Infof("Successfully dialed Unix socket %v", addr)
				}
				return c, err
			}))

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, error: %v", endpoint, err)
	}

	return conn, nil
}

// tlsConnection returns TLS gRPC connection according to the given endpoint and client TLS configuration.
func (g *gRPCService) tlsConnection(endpoint string, clientTLSConfig *apiserverconfig.KMSTLSClientConfig) (*grpc.ClientConn, error) {
	clientCertificate, err := tls.LoadX509KeyPair(clientTLSConfig.CertFile, clientTLSConfig.KeyFile)
	if err != nil {
		return nil, err
	}

	certPool := x509.NewCertPool()
	serverCABytes, err := ioutil.ReadFile(clientTLSConfig.CAFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read server CA cert: %v", err)
	}

	ok := certPool.AppendCertsFromPEM(serverCABytes)
	if !ok {
		return nil, fmt.Errorf("failed to append server CA cert to ca-pool")
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{clientCertificate},
		RootCAs:      certPool,
	}

	if clientTLSConfig.ServerName != "" {
		tlsConfig.ServerName = clientTLSConfig.ServerName
	}

	conn, err := grpc.Dial(
		endpoint,
		grpc.WithUnaryInterceptor(g.interceptor),
		grpc.WithDefaultCallOptions(grpc.WaitForReady(true)),
		grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, error: %v", endpoint, err)
	}

	return conn, nil
}

// Parse the endpoint to extract schema, host or path.
func parseEndpoint(endpoint string) (string, error) {
	if len(endpoint) == 0 {
		return "", fmt.Errorf("remote KMS provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q for remote KMS provider, error: %v", endpoint, err)
	}

	if u.Scheme != unixProtocol {
		return "", fmt.Errorf("unsupported scheme %q for remote KMS provider", u.Scheme)
	}

	// Linux abstract namespace socket - no physical file required
	// Warning: Linux Abstract sockets have not concept of ACL (unlike traditional file based sockets).
	// However, Linux Abstract sockets are subject to Linux networking namespace, so will only be accessible to
	// containers within the same pod (unless host networking is used).
	if strings.HasPrefix(u.Path, "/@") {
		return strings.TrimPrefix(u.Path, "/"), nil
	}

	return u.Path, nil
}

func (g *gRPCService) checkAPIVersion(ctx context.Context) error {
	g.mux.Lock()
	defer g.mux.Unlock()

	if g.versionChecked {
		return nil
	}

	request := &kmsapi.VersionRequest{Version: kmsapiVersion}
	response, err := g.kmsClient.Version(ctx, request)
	if err != nil {
		return fmt.Errorf("failed get version from remote KMS provider: %v", err)
	}
	if response.Version != kmsapiVersion {
		return fmt.Errorf(versionErrorf, response.Version, kmsapiVersion)
	}
	g.versionChecked = true

	klog.V(4).Infof("Version of KMS provider is %s", response.Version)
	return nil
}

// Decrypt a given data string to obtain the original byte data.
func (g *gRPCService) Decrypt(cipher []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()

	request := &kmsapi.DecryptRequest{Cipher: cipher, Version: kmsapiVersion}
	response, err := g.kmsClient.Decrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Plain, nil
}

// Encrypt bytes to a string ciphertext.
func (g *gRPCService) Encrypt(plain []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()

	request := &kmsapi.EncryptRequest{Plain: plain, Version: kmsapiVersion}
	response, err := g.kmsClient.Encrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Cipher, nil
}

func (g *gRPCService) interceptor(
	ctx context.Context,
	method string,
	req interface{},
	reply interface{},
	cc *grpc.ClientConn,
	invoker grpc.UnaryInvoker,
	opts ...grpc.CallOption,
) error {
	if !kmsapi.IsVersionCheckMethod(method) {
		if err := g.checkAPIVersion(ctx); err != nil {
			return err
		}
	}

	return invoker(ctx, method, req, reply, cc, opts...)
}
