/*
Copyright 2022 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
	"k8s.io/klog/v2"

	kmsapi "k8s.io/kms/apis/v2alpha1"
)

const (
	version = "v2alpha1"
)

type GRPCService struct {
	addr    string
	timeout time.Duration
	server  *grpc.Server

	kmsUpstream kmsv2.Service
}

var _ kmsapi.KeyManagementServiceServer = (*GRPCService)(nil)

func NewGRPCService(
	address string,
	timeout time.Duration,

	kmsUpstream kmsv2.Service,
) (*GRPCService, error) {
	klog.V(4).Infof("Configure KMS plugin with endpoint: %s", address)

	s := &GRPCService{
		addr:        address,
		timeout:     timeout,
		kmsUpstream: kmsUpstream,
	}

	return s, nil
}

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

func (s *GRPCService) Shutdown() {
	if s.server != nil {
		s.server.GracefulStop()
	}
}

func (s *GRPCService) Close() {
	if s.server != nil {
		s.server.Stop()
	}
}

// Status returns version data to verify the state of the service.
func (s *GRPCService) Status(ctx context.Context, _ *kmsapi.StatusRequest) (*kmsapi.StatusResponse, error) {
	res, err := s.kmsUpstream.Status(ctx)
	if err != nil {
		return nil, err
	}

	return &kmsapi.StatusResponse{
		Version: version,
		Healthz: "ok",
		KeyId:   res.KeyID,
	}, nil
}

// Decrypt send a decrypt request to given upstream kms.
func (s *GRPCService) Decrypt(ctx context.Context, req *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	klog.V(4).Infof("decrypt request (id: %q) received", req.Uid)

	plaintext, err := s.kmsUpstream.Decrypt(ctx, req.Uid, &kmsv2.DecryptRequest{
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

func (s *GRPCService) Encrypt(ctx context.Context, req *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	klog.V(4).Infof("encrypt request received (id: %q)", req.Uid)

	encRes, err := s.kmsUpstream.Encrypt(ctx, req.Uid, req.Plaintext)
	if err != nil {
		return nil, err
	}

	return &kmsapi.EncryptResponse{
		Ciphertext:  encRes.Ciphertext,
		KeyId:       encRes.KeyID,
		Annotations: encRes.Annotations,
	}, nil
}
