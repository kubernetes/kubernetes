/*
 *
 * Copyright 2025 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package grpctransport provides an implementation of the
// clients.TransportBuilder interface using gRPC.
package grpctransport

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/keepalive"
)

var (
	logger = grpclog.Component("grpctransport")
)

// ServerIdentifierExtension holds settings for connecting to a gRPC server,
// such as an xDS management or an LRS server.
//
// It must be set by value (not pointer) in the
// clients.ServerIdentifier.Extensions field (See Example).
type ServerIdentifierExtension struct {
	// ConfigName is the name of the configuration to use for this transport.
	// It must be present as a key in the map of configs passed to NewBuilder.
	ConfigName string
}

// Builder creates gRPC-based Transports. It must be paired with ServerIdentifiers
// that contain an Extension field of type ServerIdentifierExtension.
type Builder struct {
	// configs is a map of configuration names to their respective Config.
	configs map[string]Config

	mu sync.Mutex
	// connections is a map of clients.ServerIdentifiers in use by the Builder
	// to connect to different servers.
	connections map[clients.ServerIdentifier]*grpc.ClientConn
	// refs tracks the number of active references to each connection.
	refs map[clients.ServerIdentifier]int
}

// Config defines the configuration for connecting to a gRPC server, including
// credentials and an optional custom new client function.
type Config struct {
	// Credentials is the credentials bundle to be used for the connection.
	Credentials credentials.Bundle
	// GRPCNewClient is an optional custom function to establish a gRPC connection.
	// If nil, grpc.NewClient will be used as the default.
	GRPCNewClient func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error)
}

// NewBuilder provides a builder for creating gRPC-based Transports using
// the credentials from provided map of credentials names to
// credentials.Bundle.
func NewBuilder(configs map[string]Config) *Builder {
	return &Builder{
		configs:     configs,
		connections: make(map[clients.ServerIdentifier]*grpc.ClientConn),
		refs:        make(map[clients.ServerIdentifier]int),
	}
}

// Build returns a gRPC-based clients.Transport.
//
// The Extension field of the ServerIdentifier must be a ServerIdentifierExtension.
func (b *Builder) Build(si clients.ServerIdentifier) (clients.Transport, error) {
	if si.ServerURI == "" {
		return nil, fmt.Errorf("grpctransport: ServerURI is not set in ServerIdentifier")
	}
	if si.Extensions == nil {
		return nil, fmt.Errorf("grpctransport: Extensions is not set in ServerIdentifier")
	}
	sce, ok := si.Extensions.(ServerIdentifierExtension)
	if !ok {
		return nil, fmt.Errorf("grpctransport: Extensions field is %T, but must be %T in ServerIdentifier", si.Extensions, ServerIdentifierExtension{})
	}

	config, ok := b.configs[sce.ConfigName]
	if !ok {
		return nil, fmt.Errorf("grpctransport: unknown config name %q specified in ServerIdentifierExtension", sce.ConfigName)
	}
	if config.Credentials == nil {
		return nil, fmt.Errorf("grpctransport: config %q has nil credentials bundle", sce.ConfigName)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if cc, ok := b.connections[si]; ok {
		if logger.V(2) {
			logger.Info("Reusing existing connection to the server for ServerIdentifier: %v", si)
		}
		b.refs[si]++
		tr := &grpcTransport{cc: cc}
		tr.cleanup = b.cleanupFunc(si, tr)
		return tr, nil
	}

	// Create a new gRPC client/channel for the server with the provided
	// credentials, server URI, and a byte codec to send and receive messages.
	// Also set a static keepalive configuration that is common across gRPC
	// language implementations.
	kpCfg := grpc.WithKeepaliveParams(keepalive.ClientParameters{
		Time:    5 * time.Minute,
		Timeout: 20 * time.Second,
	})
	dopts := []grpc.DialOption{kpCfg, grpc.WithCredentialsBundle(config.Credentials), grpc.WithDefaultCallOptions(grpc.ForceCodec(&byteCodec{}))}
	newClientFunc := grpc.NewClient
	if config.GRPCNewClient != nil {
		newClientFunc = config.GRPCNewClient
	}
	cc, err := newClientFunc(si.ServerURI, dopts...)
	if err != nil {
		return nil, fmt.Errorf("grpctransport: failed to create connection to server %q: %v", si.ServerURI, err)
	}
	tr := &grpcTransport{cc: cc}
	// Register a cleanup function that decrements the refs to the gRPC
	// transport each time Close() is called to close it and remove from
	// transports and connections map if last reference is being released.
	tr.cleanup = b.cleanupFunc(si, tr)

	// Add the newly created connection to the maps to re-use the transport
	// channel and track references.
	b.connections[si] = cc
	b.refs[si] = 1

	if logger.V(2) {
		logger.Info("Created a new transport to the server for ServerIdentifier: %v", si)
	}
	return tr, nil
}

func (b *Builder) cleanupFunc(si clients.ServerIdentifier, tr *grpcTransport) func() {
	return sync.OnceFunc(func() {
		b.mu.Lock()
		defer b.mu.Unlock()

		b.refs[si]--
		if b.refs[si] != 0 {
			return
		}

		tr.cc.Close()
		tr.cc = nil
		delete(b.connections, si)
		delete(b.refs, si)
	})
}

type grpcTransport struct {
	cc *grpc.ClientConn

	// cleanup is the function to be invoked for releasing the references to
	// the gRPC transport each time Close() is called.
	cleanup func()
}

// NewStream creates a new gRPC stream to the server for the specified method.
func (g *grpcTransport) NewStream(ctx context.Context, method string) (clients.Stream, error) {
	s, err := g.cc.NewStream(ctx, &grpc.StreamDesc{ClientStreams: true, ServerStreams: true}, method)
	if err != nil {
		return nil, err
	}
	return &stream{stream: s}, nil
}

// Close closes the gRPC channel to the server.
func (g *grpcTransport) Close() {
	g.cleanup()
}

type stream struct {
	stream grpc.ClientStream
}

// Send sends a message to the server.
func (s *stream) Send(msg []byte) error {
	return s.stream.SendMsg(msg)
}

// Recv receives a message from the server.
func (s *stream) Recv() ([]byte, error) {
	var typedRes []byte

	if err := s.stream.RecvMsg(&typedRes); err != nil {
		return nil, err
	}
	return typedRes, nil
}

// byteCodec here is still sending proto messages. It's just they are
// in []byte form.
type byteCodec struct{}

func (c *byteCodec) Marshal(v any) ([]byte, error) {
	if b, ok := v.([]byte); ok {
		return b, nil
	}
	return nil, fmt.Errorf("grpctransport: message is %T, but must be a []byte", v)
}

func (c *byteCodec) Unmarshal(data []byte, v any) error {
	if b, ok := v.(*[]byte); ok {
		*b = data
		return nil
	}
	return fmt.Errorf("grpctransport: target is %T, but must be *[]byte", v)
}

func (c *byteCodec) Name() string {
	// Return "" to ensure the Content-Type header is "application/grpc",
	// which is expected by standard gRPC servers for protobuf messages.
	return ""
}
