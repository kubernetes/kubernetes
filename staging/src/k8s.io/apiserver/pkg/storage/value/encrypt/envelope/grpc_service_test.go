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
	"encoding/base64"
	"fmt"
	"net"
	"reflect"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"golang.org/x/net/context"
	"k8s.io/client-go/util/cert"
)

const (
	// Port 0 to select any available port
	listenerAddr = "127.0.0.1:0"

	cafile     = "testdata/ca.crt"
	serverCert = "testdata/server.crt"
	serverKey  = "testdata/server.key"
	clientCert = "testdata/client.crt"
	clientKey  = "testdata/client.key"
)

func TestTcpEndpoint(t *testing.T) {
	// Start the gRPC server that listens on tcp socket.
	listener, err := tcpListner()
	if err != nil {
		t.Fatal(err)
	}

	server := startTestKmsProvider(listener)
	defer server.Stop()

	endpoint := tcpProtocol + "://" + listener.Addr().String()
	verifyService(t, endpoint, "", "", "")
}

func TestTlsEndpoint(t *testing.T) {
	// Start the gRPC server that listens on tcp socket.
	listener, err := tcpListner()
	if err != nil {
		t.Fatal(err)
	}

	tlsOption, err := tlsServerOption()
	if err != nil {
		t.Fatal(err)
	}

	server := startTestKmsProvider(listener, tlsOption)
	defer server.Stop()

	// There are 2 TLS case: no auth and client auth.
	endpoint := tcpProtocol + "://" + listener.Addr().String()
	certConfigs := []struct {
		name         string
		serverCACert string
		clientCert   string
		clientKey    string
	}{
		{"noAuth", cafile, "", ""},
		{"clientAuth", cafile, clientCert, clientKey},
	}
	for _, testCase := range certConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			verifyService(t, endpoint, testCase.serverCACert, testCase.clientCert, testCase.clientKey)
		})
	}
}

func TestInvalidConfiguration(t *testing.T) {
	// Start the gRPC server that listens on tcp socket.
	listener, err := tcpListner()
	if err != nil {
		t.Fatal(err)
	}

	tlsOption, err := tlsServerOption()
	if err != nil {
		t.Fatal(err)
	}

	server := startTestKmsProvider(listener, tlsOption)
	defer server.Stop()
	endpoint := tcpProtocol + "://" + listener.Addr().String()

	invalidConfigs := []struct {
		name         string
		endpoint     string
		serverCACert string
		clientCert   string
		clientKey    string
	}{
		{"emptyConfiguration", "", "", "", ""},
		{"invalidEndpoint", "http://localhost:80", "", "", ""},
		{"invalidServerCACert", endpoint, "non-exits.pem", "", ""},
		{"missClientKey", endpoint, cafile, clientCert, ""},
		{"invalidClientCert", endpoint, cafile, "non-exists.pem", clientKey},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := NewEnvelopeService(
				testCase.endpoint,
				testCase.serverCACert,
				testCase.clientCert,
				testCase.clientKey,
			)
			if err == nil {
				t.Fatalf("should fail to create envelope service for %s.", testCase.name)
			}
		})
	}
}

func verifyService(t *testing.T, endpoint, serverCACert, clientCert, clientKey string) {
	service, err := NewEnvelopeService(endpoint, serverCACert, clientCert, clientKey)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	data := []byte("test data")
	cipher, err := service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}

	result, err := service.Decrypt(cipher)
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}

	if !reflect.DeepEqual(data, result) {
		t.Errorf("expect: %v, but: %v", data, result)
	}
}

func destroyService(service Service) {
	gs := service.(*gRPCService)
	gs.connection.Close()
}

func tcpListner() (net.Listener, error) {
	listener, err := net.Listen(tcpProtocol, listenerAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on the tcp address, error: %v", err)
	}

	return listener, nil
}

func startTestKmsProvider(listener net.Listener, options ...grpc.ServerOption) *grpc.Server {
	server := grpc.NewServer(options...)
	RegisterKmsServiceServer(server, &base64Server{})
	go server.Serve(listener)
	return server
}

func tlsServerOption() (grpc.ServerOption, error) {
	certificate, err := tls.LoadX509KeyPair(serverCert, serverKey)
	if err != nil {
		return nil, fmt.Errorf("bad server cert or key, error: %v", err)
	}

	certPool, err := cert.NewPool(cafile)
	if err != nil {
		return nil, fmt.Errorf("bad ca file, error: %v", err)
	}

	tlsConfig := &tls.Config{
		ClientAuth:   tls.VerifyClientCertIfGiven,
		Certificates: []tls.Certificate{certificate},
		ClientCAs:    certPool,
	}
	return grpc.Creds(credentials.NewTLS(tlsConfig)), nil
}

// Fake gRPC sever for remote KMS provider.
type base64Server struct{}

func (b *base64Server) Decrypt(ctx context.Context, request *DecryptRequest) (*DecryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.DecodedLen(len(request.Cipher)))
	n, err := base64.StdEncoding.Decode(buf, request.Cipher)
	if err != nil {
		return nil, err
	}

	return &DecryptResponse{Plain: buf[:n]}, nil
}

func (b *base64Server) Encrypt(ctx context.Context, request *EncryptRequest) (*EncryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.EncodedLen(len(request.Plain)))
	base64.StdEncoding.Encode(buf, request.Plain)
	return &EncryptResponse{Cipher: buf}, nil
}
