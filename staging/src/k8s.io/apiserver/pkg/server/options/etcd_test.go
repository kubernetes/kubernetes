/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func TestEtcdOptionsValidate(t *testing.T) {
	testCases := []struct {
		name        string
		testOptions *EtcdOptions
		expectErr   string
	}{
		{
			name: "test when ServerList is not specified",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd3",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList: nil,
						KeyFile:    "/var/run/kubernetes/etcd.key",
						CAFile:     "/var/run/kubernetes/etcdca.crt",
						CertFile:   "/var/run/kubernetes/etcdce.crt",
					},
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
				DeleteCollectionWorkers: 1,
				EnableGarbageCollection: true,
				EnableWatchCache:        true,
				DefaultWatchCacheSize:   100,
				EtcdServersOverrides:    []string{"/events#http://127.0.0.1:4002"},
			},
			expectErr: "--etcd-servers must be specified",
		},
		{
			name: "test when storage-backend is invalid",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd4",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList: []string{"http://127.0.0.1"},
						KeyFile:    "/var/run/kubernetes/etcd.key",
						CAFile:     "/var/run/kubernetes/etcdca.crt",
						CertFile:   "/var/run/kubernetes/etcdce.crt",
					},
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
				DeleteCollectionWorkers: 1,
				EnableGarbageCollection: true,
				EnableWatchCache:        true,
				DefaultWatchCacheSize:   100,
				EtcdServersOverrides:    []string{"/events#http://127.0.0.1:4002"},
			},
			expectErr: "--storage-backend invalid, allowed values: etcd3. If not specified, it will default to 'etcd3'",
		},
		{
			name: "test when etcd-servers-overrides is invalid",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type: "etcd3",
					Transport: storagebackend.TransportConfig{
						ServerList: []string{"http://127.0.0.1"},
						KeyFile:    "/var/run/kubernetes/etcd.key",
						CAFile:     "/var/run/kubernetes/etcdca.crt",
						CertFile:   "/var/run/kubernetes/etcdce.crt",
					},
					Prefix:                "/registry",
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
				DeleteCollectionWorkers: 1,
				EnableGarbageCollection: true,
				EnableWatchCache:        true,
				DefaultWatchCacheSize:   100,
				EtcdServersOverrides:    []string{"/events/http://127.0.0.1:4002"},
			},
			expectErr: "--etcd-servers-overrides invalid, must be of format: group/resource#servers, where servers are URLs, semicolon separated",
		},
		{
			name: "test when EtcdOptions is valid",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd3",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList: []string{"http://127.0.0.1"},
						KeyFile:    "/var/run/kubernetes/etcd.key",
						CAFile:     "/var/run/kubernetes/etcdca.crt",
						CertFile:   "/var/run/kubernetes/etcdce.crt",
					},
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
				DeleteCollectionWorkers: 1,
				EnableGarbageCollection: true,
				EnableWatchCache:        true,
				DefaultWatchCacheSize:   100,
				EtcdServersOverrides:    []string{"/events#http://127.0.0.1:4002"},
			},
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			errs := testcase.testOptions.Validate()
			if len(testcase.expectErr) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) {
				t.Errorf("got err: %v, expected err: %s", errs, testcase.expectErr)
			}
			if len(testcase.expectErr) == 0 && len(errs) != 0 {
				t.Errorf("got err: %s, expected err nil", errs)
			}
		})
	}
}

func TestParseWatchCacheSizes(t *testing.T) {
	testCases := []struct {
		name                  string
		cacheSizes            []string
		expectWatchCacheSizes map[schema.GroupResource]int
		expectErr             string
	}{
		{
			name:       "test when invalid value of watch cache size",
			cacheSizes: []string{"deployments.apps#65536", "replicasets.extensions"},
			expectErr:  "invalid value of watch cache size",
		},
		{
			name:       "test when invalid size of watch cache size",
			cacheSizes: []string{"deployments.apps#65536", "replicasets.extensions#655d1"},
			expectErr:  "invalid size of watch cache size",
		},
		{
			name:       "test when watch cache size is negative",
			cacheSizes: []string{"deployments.apps#65536", "replicasets.extensions#-65536"},
			expectErr:  "watch cache size cannot be negative",
		},
		{
			name:       "test when parse watch cache size success",
			cacheSizes: []string{"deployments.apps#65536", "replicasets.extensions#65536"},
			expectWatchCacheSizes: map[schema.GroupResource]int{
				{Group: "apps", Resource: "deployments"}:       65536,
				{Group: "extensions", Resource: "replicasets"}: 65536,
			},
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			result, err := ParseWatchCacheSizes(testcase.cacheSizes)
			if len(testcase.expectErr) != 0 && !strings.Contains(err.Error(), testcase.expectErr) {
				t.Errorf("got err: %v, expected err: %s", err, testcase.expectErr)
			}
			if len(testcase.expectErr) == 0 {
				if err != nil {
					t.Errorf("got err: %v, expected err nil", err)
				} else {
					for key, expectValue := range testcase.expectWatchCacheSizes {
						if resultValue, exist := result[key]; !exist || resultValue != expectValue {
							t.Errorf("got watch cache size: %v, expected watch cache size %v", result, testcase.expectWatchCacheSizes)
						}
					}
				}
			}
		})
	}
}

func TestKMSHealthzEndpoint(t *testing.T) {
	testCases := []struct {
		name                 string
		encryptionConfigPath string
		wantChecks           []string
	}{
		{
			name:                 "single kms-provider, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/single-kms-provider.yaml",
			wantChecks:           []string{"etcd", "kms-provider-0"},
		},
		{
			name:                 "two kms-providers, expect two kms healthz checks",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-providers.yaml",
			wantChecks:           []string{"etcd", "kms-provider-0", "kms-provider-1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			serverConfig := &server.Config{}
			etcdOptions := &EtcdOptions{
				EncryptionProviderConfigFilepath: tc.encryptionConfigPath,
			}
			if err := etcdOptions.addEtcdHealthEndpoint(serverConfig); err != nil {
				t.Fatalf("Failed to add healthz error: %v", err)
			}

			for _, n := range tc.wantChecks {
				found := false
				for _, h := range serverConfig.HealthzChecks {
					if n == h.Name() {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Missing HealthzChecker %s", n)
				}
				found = false
			}
		})
	}
}
