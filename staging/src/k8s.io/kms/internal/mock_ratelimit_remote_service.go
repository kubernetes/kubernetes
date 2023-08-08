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

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kms/pkg/service"
)

type mockRateLimitRemoteService struct {
	delegate service.Service
	limiter  flowcontrol.RateLimiter
}

var _ service.Service = &mockRateLimitRemoteService{}

func (s *mockRateLimitRemoteService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	if !s.limiter.TryAccept() {
		return nil, status.New(codes.ResourceExhausted, "remote decrypt rate limit exceeded").Err()
	}
	return s.delegate.Decrypt(ctx, uid, req)
}

func (s *mockRateLimitRemoteService) Encrypt(ctx context.Context, uid string, data []byte) (*service.EncryptResponse, error) {
	if !s.limiter.TryAccept() {
		return nil, status.New(codes.ResourceExhausted, "remote encrypt rate limit exceeded").Err()
	}
	return s.delegate.Encrypt(ctx, uid, data)
}

func (s *mockRateLimitRemoteService) Status(ctx context.Context) (*service.StatusResponse, error) {
	// Passthrough here, not adding any rate limiting for status as rate limits are usually for encrypt and decrypt requests.
	return s.delegate.Status(ctx)
}

// NewMockRateLimitService creates an instance of mockRateLimitRemoteService.
func NewMockRateLimitService(delegate service.Service, qps float32, burst int) service.Service {
	return &mockRateLimitRemoteService{
		delegate: delegate,
		limiter:  flowcontrol.NewTokenBucketRateLimiter(qps, burst),
	}
}
