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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
						ServerList:    nil,
						KeyFile:       "/var/run/kubernetes/etcd.key",
						TrustedCAFile: "/var/run/kubernetes/etcdca.crt",
						CertFile:      "/var/run/kubernetes/etcdce.crt",
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
						ServerList:    []string{"http://127.0.0.1"},
						KeyFile:       "/var/run/kubernetes/etcd.key",
						TrustedCAFile: "/var/run/kubernetes/etcdca.crt",
						CertFile:      "/var/run/kubernetes/etcdce.crt",
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
						ServerList:    []string{"http://127.0.0.1"},
						KeyFile:       "/var/run/kubernetes/etcd.key",
						TrustedCAFile: "/var/run/kubernetes/etcdca.crt",
						CertFile:      "/var/run/kubernetes/etcdce.crt",
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
			name: "test when encryption-provider-config-automatic-reload is invalid",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd3",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList:    []string{"http://127.0.0.1"},
						KeyFile:       "/var/run/kubernetes/etcd.key",
						TrustedCAFile: "/var/run/kubernetes/etcdca.crt",
						CertFile:      "/var/run/kubernetes/etcdce.crt",
					},
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				EncryptionProviderConfigAutomaticReload: true,
				DefaultStorageMediaType:                 "application/vnd.kubernetes.protobuf",
				DeleteCollectionWorkers:                 1,
				EnableGarbageCollection:                 true,
				EnableWatchCache:                        true,
				DefaultWatchCacheSize:                   100,
				EtcdServersOverrides:                    []string{"/events#http://127.0.0.1:4002"},
			},
			expectErr: "--encryption-provider-config-automatic-reload must be set with --encryption-provider-config",
		},
		{
			name: "test when EtcdOptions is valid",
			testOptions: &EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd3",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList:    []string{"http://127.0.0.1"},
						KeyFile:       "/var/run/kubernetes/etcd.key",
						TrustedCAFile: "/var/run/kubernetes/etcdca.crt",
						CertFile:      "/var/run/kubernetes/etcdce.crt",
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	testCases := []struct {
		name                 string
		encryptionConfigPath string
		wantChecks           []string
		skipHealth           bool
		reload               bool
	}{
		{
			name:                 "no kms-provider, expect no kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/no-kms-provider.yaml",
			wantChecks:           []string{"etcd"},
		},
		{
			name:                 "no kms-provider+reload, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/no-kms-provider.yaml",
			reload:               true,
			wantChecks:           []string{"etcd", "kms-providers"},
		},
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
		{
			name:                 "two kms-providers+reload, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-providers.yaml",
			reload:               true,
			wantChecks:           []string{"etcd", "kms-providers"},
		},
		{
			name:                 "kms v1+v2, expect three kms healthz checks",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-providers-with-v2.yaml",
			wantChecks:           []string{"etcd", "kms-provider-0", "kms-provider-1", "kms-provider-2"},
		},
		{
			name:                 "kms v1+v2+reload, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-providers-with-v2.yaml",
			reload:               true,
			wantChecks:           []string{"etcd", "kms-providers"},
		},
		{
			name:                 "multiple kms v2, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-v2-providers.yaml",
			wantChecks:           []string{"etcd", "kms-providers"},
		},
		{
			name:                 "multiple kms v2+reload, expect single kms healthz check",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-v2-providers.yaml",
			reload:               true,
			wantChecks:           []string{"etcd", "kms-providers"},
		},
		{
			name:                 "two kms-providers with skip, expect zero kms healthz checks",
			encryptionConfigPath: "testdata/encryption-configs/multiple-kms-providers.yaml",
			wantChecks:           nil,
			skipHealth:           true,
		},
	}

	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			serverConfig := server.NewConfig(codecs)
			etcdOptions := &EtcdOptions{
				EncryptionProviderConfigFilepath:        tc.encryptionConfigPath,
				EncryptionProviderConfigAutomaticReload: tc.reload,
				SkipHealthEndpoints:                     tc.skipHealth,
			}
			if err := etcdOptions.Complete(serverConfig.StorageObjectCountTracker, serverConfig.DrainedNotify()); err != nil {
				t.Fatal(err)
			}
			if err := etcdOptions.ApplyTo(serverConfig); err != nil {
				t.Fatalf("Failed to add healthz error: %v", err)
			}

			healthChecksAreEqual(t, tc.wantChecks, serverConfig.HealthzChecks)
		})
	}
}

func TestReadinessCheck(t *testing.T) {
	testCases := []struct {
		name              string
		wantReadyzChecks  []string
		wantHealthzChecks []string
		skipHealth        bool
	}{
		{
			name:              "Readyz should have etcd-readiness check",
			wantReadyzChecks:  []string{"etcd", "etcd-readiness"},
			wantHealthzChecks: []string{"etcd"},
		},
		{
			name:              "skip health, Readyz should not have etcd-readiness check",
			wantReadyzChecks:  nil,
			wantHealthzChecks: nil,
			skipHealth:        true,
		},
	}

	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			serverConfig := server.NewConfig(codecs)
			etcdOptions := &EtcdOptions{SkipHealthEndpoints: tc.skipHealth}
			if err := etcdOptions.Complete(serverConfig.StorageObjectCountTracker, serverConfig.DrainedNotify()); err != nil {
				t.Fatal(err)
			}
			if err := etcdOptions.ApplyTo(serverConfig); err != nil {
				t.Fatalf("Failed to add healthz error: %v", err)
			}

			healthChecksAreEqual(t, tc.wantReadyzChecks, serverConfig.ReadyzChecks)
			healthChecksAreEqual(t, tc.wantHealthzChecks, serverConfig.HealthzChecks)
		})
	}
}

func healthChecksAreEqual(t *testing.T, want []string, healthChecks []healthz.HealthChecker) {
	t.Helper()

	wantSet := sets.NewString(want...)
	gotSet := sets.NewString()

	for _, h := range healthChecks {
		gotSet.Insert(h.Name())
	}

	gotSet.Delete("log", "ping") // not relevant for our tests

	if !wantSet.Equal(gotSet) {
		t.Errorf("healthz checks are not equal, missing=%q, extra=%q", wantSet.Difference(gotSet).List(), gotSet.Difference(wantSet).List())
	}
}
