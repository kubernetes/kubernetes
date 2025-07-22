/*
Copyright 2023 The Kubernetes Authors.

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

package service

import (
	"context"
	"net"
	"time"

	"google.golang.org/grpc"

	kmsapi "k8s.io/kms/apis/v2"
)

// GRPCService is a grpc server that runs the kms v2 alpha1 API.
type GRPCService struct {
	addr    string
	timeout time.Duration
	server  *grpc.Server

	kmsService Service
	kmsapi.UnsafeKeyManagementServiceServer
}

var _ kmsapi.KeyManagementServiceServer = (*GRPCService)(nil)

// NewGRPCService creates an instance of GRPCService.
func NewGRPCService(
	address string,
	timeout time.Duration,

	kmsService Service,
) *GRPCService {
	return &GRPCService{
		addr:       address,
		timeout:    timeout,
		kmsService: kmsService,
	}
}

// ListenAndServe accepts incoming connections on a Unix socket. It is a blocking method.
// Returns non-nil error unless Close or Shutdown is called.
func (s *GRPCService) ListenAndServe() error {
	ln, err := net.Listen("unix", s.addr)
	if err != nil {
		return err
	}
	defer ln.Close()

	gs := grpc.NewServer(
		grpc.ConnectionTimeout(s.timeout),
	)
	s.server = gs

	kmsapi.RegisterKeyManagementServiceServer(gs, s)

	return gs.Serve(ln)
}

// Shutdown performs a graceful shutdown. Doesn't accept new connections and
// blocks until all pending RPCs are finished.
func (s *GRPCService) Shutdown() {
	if s.server != nil {
		s.server.GracefulStop()
	}
}

// Close stops the server by closing all connections immediately and cancels
// all active RPCs.
func (s *GRPCService) Close() {
	if s.server != nil {
		s.server.Stop()
	}
}

// Status sends a status request to specified kms service.
func (s *GRPCService) Status(ctx context.Context, _ *kmsapi.StatusRequest) (*kmsapi.StatusResponse, error) {
	res, err := s.kmsService.Status(ctx)
	if err != nil {
		return nil, err
	}

	return &kmsapi.StatusResponse{
		Version: res.Version,
		Healthz: res.Healthz,
		KeyId:   res.KeyID,
	}, nil
}

// Decrypt sends a decryption request to specified kms service.
func (s *GRPCService) Decrypt(ctx context.Context, req *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	plaintext, err := s.kmsService.Decrypt(ctx, req.Uid, &DecryptRequest{
		Ciphertext:  req.Ciphertext,
		KeyID:       req.KeyId,
		Annotations: req.Annotations,
	})
	if err != nil {
		return nil, err
	}

	return &kmsapi.DecryptResponse{
		Plaintext: plaintext,
	}, nil
}

// Encrypt sends an encryption request to specified kms service.
func (s *GRPCService) Encrypt(ctx context.Context, req *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	encRes, err := s.kmsService.Encrypt(ctx, req.Uid, req.Plaintext)
	if err != nil {
		return nil, err
	}

	return &kmsapi.EncryptResponse{
		Ciphertext:  encRes.Ciphertext,
		KeyId:       encRes.KeyID,
		Annotations: encRes.Annotations,
	}, nil
}
