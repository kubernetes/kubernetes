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

package internal

import (
	"context"
	"time"

	"k8s.io/kms/pkg/service"
)

type mockLatencyRemoteService struct {
	delegate service.Service
	latency  time.Duration
}

var _ service.Service = &mockLatencyRemoteService{}

func (s *mockLatencyRemoteService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	time.Sleep(s.latency)
	return s.delegate.Decrypt(ctx, uid, req)
}

func (s *mockLatencyRemoteService) Encrypt(ctx context.Context, uid string, data []byte) (*service.EncryptResponse, error) {
	time.Sleep(s.latency)
	return s.delegate.Encrypt(ctx, uid, data)
}

func (s *mockLatencyRemoteService) Status(ctx context.Context) (*service.StatusResponse, error) {
	// Passthrough here, not adding any delays for status as delays are usually negligible compare to encrypt and decrypt requests.
	return s.delegate.Status(ctx)
}

// NewMockLatencyService creates an instance of mockLatencyRemoteService.
func NewMockLatencyService(delegate service.Service, latency time.Duration) service.Service {
	return &mockLatencyRemoteService{
		delegate: delegate,
		latency:  latency,
	}
}
