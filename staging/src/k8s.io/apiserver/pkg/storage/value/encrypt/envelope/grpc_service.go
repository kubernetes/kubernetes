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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"golang.org/x/net/context"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	// Supportied protocol schema for gRPC.
	tcpProtocol  = "tcp"
	unixProtocol = "unix"

	// Current version for the protocal interface definition.
	version = "v1beta1"
)

type gRPCService struct {
	// gRPC client instance
	kmsClient  kmsapi.KMSServiceClient
	connection *grpc.ClientConn
}

// NewGRPCService returns an envelope.Service which use gRPC to communicate the remote KMS provider.
func NewGRPCService(endpoint, serverCert, clientCert, clientKey string) (Service, error) {
	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	dialer := func(addr string, timeout time.Duration) (net.Conn, error) {
		return net.DialTimeout(protocol, addr, timeout)
	}

	// With or without TLS/SSL support
	tlsOption, err := getTLSDialOption(serverCert, clientCert, clientKey)
	if err != nil {
		return nil, err
	}

	conn, err := grpc.Dial(addr, tlsOption, grpc.WithDialer(dialer))
	if err != nil {
		return nil, fmt.Errorf("connect remote image service %s failed, error: %v", addr, err)
	}

	return &gRPCService{kmsClient: kmsapi.NewKMSServiceClient(conn), connection: conn}, nil
}

// Parse the endpoint to extract schema, host or path.
func parseEndpoint(endpoint string) (string, string, error) {
	if len(endpoint) == 0 {
		return "", "", fmt.Errorf("remote KMS provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", fmt.Errorf("invalid kms provider endpoint %q, error: %v", endpoint, err)
	}

	switch u.Scheme {
	case tcpProtocol:
		return tcpProtocol, u.Host, nil
	case unixProtocol:
		return unixProtocol, u.Path, nil
	default:
		return "", "", fmt.Errorf("invalid endpoint %q for remote KMS provider", endpoint)
	}
}

// Build the TLS/SSL options for gRPC client.
func getTLSDialOption(serverCert, clientCert, clientKey string) (grpc.DialOption, error) {
	// No TLS/SSL support.
	if len(serverCert) == 0 && len(clientCert) == 0 && len(clientKey) == 0 {
		return grpc.WithInsecure(), nil
	}

	// Set the CA that verify the certificate from the gRPC server.
	certPool := x509.NewCertPool()
	if len(serverCert) > 0 {
		ca, err := ioutil.ReadFile(serverCert)
		if err != nil {
			return nil, fmt.Errorf("kms provider invalid server cert, error: %v", err)
		}
		if !certPool.AppendCertsFromPEM(ca) {
			return nil, fmt.Errorf("can't append server cert for kms provider")
		}
	}

	// Set client authenticate certificate.
	certificates := make([]tls.Certificate, 0, 1)
	if len(clientCert) != 0 || len(clientKey) != 0 {
		if len(clientCert) == 0 || len(clientKey) == 0 {
			return nil, fmt.Errorf("both client cert and key must be provided for kms provider")
		}

		cert, err := tls.LoadX509KeyPair(clientCert, clientKey)
		if err != nil {
			return nil, fmt.Errorf("kms provider invalid client cert or key, error: %v", err)
		}
		certificates = append(certificates, cert)
	}

	tlsConfig := tls.Config{
		Certificates: certificates,
		RootCAs:      certPool,
	}
	transportCreds := credentials.NewTLS(&tlsConfig)
	return grpc.WithTransportCredentials(transportCreds), nil
}

// Decrypt a given data string to obtain the original byte data.
func (g *gRPCService) Decrypt(cipher string) ([]byte, error) {
	request := &kmsapi.DecryptRequest{Cipher: []byte(cipher), Version: version}
	response, err := g.kmsClient.Decrypt(context.Background(), request)
	if err != nil {
		return nil, err
	}
	return response.Plain, nil
}

// Encrypt bytes to a string ciphertext.
func (g *gRPCService) Encrypt(plain []byte) (string, error) {
	request := &kmsapi.EncryptRequest{Plain: plain, Version: version}
	response, err := g.kmsClient.Encrypt(context.Background(), request)
	if err != nil {
		return "", err
	}
	return string(response.Cipher), nil
}
