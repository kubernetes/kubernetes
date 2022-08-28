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

package factory

import (
	"context"
	"errors"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

type mockKV struct {
	get func(ctx context.Context) (*clientv3.GetResponse, error)
}

func (mkv mockKV) Put(ctx context.Context, key, val string, opts ...clientv3.OpOption) (*clientv3.PutResponse, error) {
	return nil, nil
}
func (mkv mockKV) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	return mkv.get(ctx)
}
func (mockKV) Delete(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.DeleteResponse, error) {
	return nil, nil
}
func (mockKV) Compact(ctx context.Context, rev int64, opts ...clientv3.CompactOption) (*clientv3.CompactResponse, error) {
	return nil, nil
}
func (mockKV) Do(ctx context.Context, op clientv3.Op) (clientv3.OpResponse, error) {
	return clientv3.OpResponse{}, nil
}
func (mockKV) Txn(ctx context.Context) clientv3.Txn {
	return nil
}

func TestCreateHealthcheck(t *testing.T) {
	etcdConfig := testserver.NewTestConfig(t)
	client := testserver.RunEtcd(t, etcdConfig)
	newETCD3ClientFn := newETCD3Client
	defer func() {
		newETCD3Client = newETCD3ClientFn
	}()
	tests := []struct {
		name         string
		cfg          storagebackend.Config
		want         error
		responseTime time.Duration
	}{
		{
			name: "ok if response time lower than default timeout",
			cfg: storagebackend.Config{
				Type:      storagebackend.StorageTypeETCD3,
				Transport: storagebackend.TransportConfig{},
			},
			responseTime: 1 * time.Second,
			want:         nil,
		},
		{
			name: "ok if response time lower than custom timeout",
			cfg: storagebackend.Config{
				Type:               storagebackend.StorageTypeETCD3,
				Transport:          storagebackend.TransportConfig{},
				HealthcheckTimeout: 5 * time.Second,
			},
			responseTime: 3 * time.Second,
			want:         nil,
		},
		{
			name: "timeouts if response time higher than default timeout",
			cfg: storagebackend.Config{
				Type:      storagebackend.StorageTypeETCD3,
				Transport: storagebackend.TransportConfig{},
			},
			responseTime: 3 * time.Second,
			want:         context.DeadlineExceeded,
		},
		{
			name: "timeouts if response time higher than custom timeout",
			cfg: storagebackend.Config{
				Type:               storagebackend.StorageTypeETCD3,
				Transport:          storagebackend.TransportConfig{},
				HealthcheckTimeout: 3 * time.Second,
			},
			responseTime: 5 * time.Second,
			want:         context.DeadlineExceeded,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ready := make(chan struct{})
			tc.cfg.Transport.ServerList = client.Endpoints()
			newETCD3Client = func(c storagebackend.TransportConfig) (*clientv3.Client, error) {
				defer close(ready)
				dummyKV := mockKV{
					get: func(ctx context.Context) (*clientv3.GetResponse, error) {
						select {
						case <-ctx.Done():
							return nil, ctx.Err()
						case <-time.After(tc.responseTime):
							return nil, nil
						}
					},
				}
				client.KV = dummyKV
				return client, nil
			}
			stop := make(chan struct{})
			defer close(stop)

			healthcheck, err := CreateHealthCheck(tc.cfg, stop)
			if err != nil {
				t.Fatal(err)
			}
			// Wait for healthcheck to establish connection
			<-ready
			got := healthcheck()

			if !errors.Is(got, tc.want) {
				t.Errorf("healthcheck() missmatch want %v got %v", tc.want, got)
			}
		})
	}
}

func TestCreateReadycheck(t *testing.T) {
	etcdConfig := testserver.NewTestConfig(t)
	client := testserver.RunEtcd(t, etcdConfig)
	newETCD3ClientFn := newETCD3Client
	defer func() {
		newETCD3Client = newETCD3ClientFn
	}()
	tests := []struct {
		name         string
		cfg          storagebackend.Config
		want         error
		responseTime time.Duration
	}{
		{
			name: "ok if response time lower than default timeout",
			cfg: storagebackend.Config{
				Type:      storagebackend.StorageTypeETCD3,
				Transport: storagebackend.TransportConfig{},
			},
			responseTime: 1 * time.Second,
			want:         nil,
		},
		{
			name: "ok if response time lower than custom timeout",
			cfg: storagebackend.Config{
				Type:              storagebackend.StorageTypeETCD3,
				Transport:         storagebackend.TransportConfig{},
				ReadycheckTimeout: 5 * time.Second,
			},
			responseTime: 3 * time.Second,
			want:         nil,
		},
		{
			name: "timeouts if response time higher than default timeout",
			cfg: storagebackend.Config{
				Type:      storagebackend.StorageTypeETCD3,
				Transport: storagebackend.TransportConfig{},
			},
			responseTime: 3 * time.Second,
			want:         context.DeadlineExceeded,
		},
		{
			name: "timeouts if response time higher than custom timeout",
			cfg: storagebackend.Config{
				Type:              storagebackend.StorageTypeETCD3,
				Transport:         storagebackend.TransportConfig{},
				ReadycheckTimeout: 3 * time.Second,
			},
			responseTime: 5 * time.Second,
			want:         context.DeadlineExceeded,
		},
		{
			name: "timeouts if response time higher than default timeout with custom healthcheck timeout",
			cfg: storagebackend.Config{
				Type:               storagebackend.StorageTypeETCD3,
				Transport:          storagebackend.TransportConfig{},
				HealthcheckTimeout: 10 * time.Second,
			},
			responseTime: 3 * time.Second,
			want:         context.DeadlineExceeded,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ready := make(chan struct{})
			tc.cfg.Transport.ServerList = client.Endpoints()
			newETCD3Client = func(c storagebackend.TransportConfig) (*clientv3.Client, error) {
				defer close(ready)
				dummyKV := mockKV{
					get: func(ctx context.Context) (*clientv3.GetResponse, error) {
						select {
						case <-ctx.Done():
							return nil, ctx.Err()
						case <-time.After(tc.responseTime):
							return nil, nil
						}
					},
				}
				client.KV = dummyKV
				return client, nil
			}
			stop := make(chan struct{})
			defer close(stop)

			healthcheck, err := CreateReadyCheck(tc.cfg, stop)
			if err != nil {
				t.Fatal(err)
			}
			// Wait for healthcheck to establish connection
			<-ready

			got := healthcheck()

			if !errors.Is(got, tc.want) {
				t.Errorf("healthcheck() missmatch want %v got %v", tc.want, got)
			}
		})
	}
}
