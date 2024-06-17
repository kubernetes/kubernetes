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

package controller

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestController(t *testing.T) {
	origMinKMSPluginCloseGracePeriod := minKMSPluginCloseGracePeriod
	t.Cleanup(func() { minKMSPluginCloseGracePeriod = origMinKMSPluginCloseGracePeriod })
	minKMSPluginCloseGracePeriod = 300 * time.Millisecond

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

	const expectedSuccessMetricValue = `
# HELP apiserver_encryption_config_controller_automatic_reload_success_total [ALPHA] Total number of successful automatic reloads of encryption configuration split by apiserver identity.
# TYPE apiserver_encryption_config_controller_automatic_reload_success_total counter
apiserver_encryption_config_controller_automatic_reload_success_total{apiserver_id_hash="sha256:cd8a60cec6134082e9f37e7a4146b4bc14a0bf8a863237c36ec8fdb658c3e027"} 1
# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
# TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
apiserver_encryption_config_controller_automatic_reloads_total{apiserver_id_hash="sha256:cd8a60cec6134082e9f37e7a4146b4bc14a0bf8a863237c36ec8fdb658c3e027",status="success"} 1
`
	const expectedFailureMetricValue = `
# HELP apiserver_encryption_config_controller_automatic_reload_failures_total [ALPHA] Total number of failed automatic reloads of encryption configuration split by apiserver identity.
# TYPE apiserver_encryption_config_controller_automatic_reload_failures_total counter
apiserver_encryption_config_controller_automatic_reload_failures_total{apiserver_id_hash="sha256:cd8a60cec6134082e9f37e7a4146b4bc14a0bf8a863237c36ec8fdb658c3e027"} 1
# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
# TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
apiserver_encryption_config_controller_automatic_reloads_total{apiserver_id_hash="sha256:cd8a60cec6134082e9f37e7a4146b4bc14a0bf8a863237c36ec8fdb658c3e027",status="failure"} 1
`

	tests := []struct {
		name                        string
		wantECFileHash              string
		wantTransformerClosed       bool
		wantLoadCalls               int
		wantHashCalls               int
		wantAddRateLimitedCount     uint64
		wantMetrics                 string
		mockLoadEncryptionConfig    func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error)
		mockGetEncryptionConfigHash func(ctx context.Context, filepath string) (string, error)
	}{
		{
			name:                    "when invalid config is provided previous config shouldn't be changed",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             expectedFailureMetricValue,
			wantAddRateLimitedCount: 1,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return nil, fmt.Errorf("empty config file")
			},
		},
		{
			name:                    "when new valid config is provided it should be updated",
			wantECFileHash:          "some new config hash",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantMetrics:             expectedSuccessMetricValue,
			wantAddRateLimitedCount: 0,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "valid-plugin",
							err:        nil,
						},
					},
					EncryptionFileContentHash: "some new config hash",
				}, nil
			},
		},
		{
			name:                    "when same valid config is provided previous config shouldn't be changed",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             "",
			wantAddRateLimitedCount: 0,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "valid-plugin",
							err:        nil,
						},
					},
					// hash of initial "testdata/ec_config.yaml" config file before reloading
					EncryptionFileContentHash: "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
				}, nil
			},
		},
		{
			name:                    "when transformer's health check fails previous config shouldn't be changed",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             expectedFailureMetricValue,
			wantAddRateLimitedCount: 1,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "invalid-plugin",
							err:        fmt.Errorf("mockingly failing"),
						},
					},
					KMSCloseGracePeriod:       0, // use minKMSPluginCloseGracePeriod
					EncryptionFileContentHash: "anything different",
				}, nil
			},
		},
		{
			name:                    "when multiple health checks are present previous config shouldn't be changed",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             expectedFailureMetricValue,
			wantAddRateLimitedCount: 1,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "valid-plugin",
							err:        nil,
						},
						&mockHealthChecker{
							pluginName: "another-valid-plugin",
							err:        nil,
						},
					},
					EncryptionFileContentHash: "anything different",
				}, nil
			},
		},
		{
			name:                    "when invalid health check URL is provided previous config shouldn't be changed",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             expectedFailureMetricValue,
			wantAddRateLimitedCount: 1,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "invalid\nname",
							err:        nil,
						},
					},
					EncryptionFileContentHash: "anything different",
				}, nil
			},
		},
		{
			name:                    "when config is not updated transformers are closed correctly",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           1,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             "",
			wantAddRateLimitedCount: 0,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "always changes and never errors", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return &encryptionconfig.EncryptionConfiguration{
					HealthChecks: []healthz.HealthChecker{
						&mockHealthChecker{
							pluginName: "valid-plugin",
							err:        nil,
						},
					},
					// hash of initial "testdata/ec_config.yaml" config file before reloading
					EncryptionFileContentHash: "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
				}, nil
			},
		},
		{
			name:                    "when config hash is not updated transformers are closed correctly",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           0,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             "",
			wantAddRateLimitedCount: 0,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				// hash of initial "testdata/ec_config.yaml" config file before reloading
				return "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3", nil
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return nil, fmt.Errorf("should not be called")
			},
		},
		{
			name:                    "when config hash errors transformers are closed correctly",
			wantECFileHash:          "k8s:enc:unstable:1:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3",
			wantLoadCalls:           0,
			wantHashCalls:           1,
			wantTransformerClosed:   true,
			wantMetrics:             expectedFailureMetricValue,
			wantAddRateLimitedCount: 1,
			mockGetEncryptionConfigHash: func(ctx context.Context, filepath string) (string, error) {
				return "", fmt.Errorf("some io error")
			},
			mockLoadEncryptionConfig: func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				return nil, fmt.Errorf("should not be called")
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctxServer, closeServer := context.WithCancel(context.Background())
			ctxTransformers, closeTransformers := context.WithCancel(ctxServer)
			t.Cleanup(closeServer)
			t.Cleanup(closeTransformers)

			legacyregistry.Reset()

			// load initial encryption config
			encryptionConfiguration, err := encryptionconfig.LoadEncryptionConfig(
				ctxTransformers,
				"testdata/ec_config.yaml",
				true,
				"test-apiserver",
			)
			if err != nil {
				t.Fatalf("failed to load encryption config: %v", err)
			}

			d := NewDynamicEncryptionConfiguration(
				"test-controller",
				"does not matter",
				encryptionconfig.NewDynamicTransformers(
					encryptionConfiguration.Transformers,
					encryptionConfiguration.HealthChecks[0],
					closeTransformers,
					0, // set grace period to 0 so that the time.Sleep in DynamicTransformers.Set finishes quickly
				),
				encryptionConfiguration.EncryptionFileContentHash,
				"test-apiserver",
			)
			d.queue.ShutDown() // we do not use the real queue during tests

			queue := &mockWorkQueue{
				addCalled: make(chan struct{}),
				cancel:    closeServer,
			}
			d.queue = queue

			var hashCalls, loadCalls int
			d.loadEncryptionConfig = func(ctx context.Context, filepath string, reload bool, apiServerID string) (*encryptionconfig.EncryptionConfiguration, error) {
				loadCalls++
				queue.ctx = ctx
				return test.mockLoadEncryptionConfig(ctx, filepath, reload, apiServerID)
			}
			d.getEncryptionConfigHash = func(ctx context.Context, filepath string) (string, error) {
				hashCalls++
				queue.ctx = ctx
				return test.mockGetEncryptionConfigHash(ctx, filepath)
			}

			d.Run(ctxServer) // this should block and run exactly one iteration of the worker loop

			if test.wantECFileHash != d.lastLoadedEncryptionConfigHash {
				t.Errorf("expected encryption config hash %q but got %q", test.wantECFileHash, d.lastLoadedEncryptionConfigHash)
			}

			if test.wantLoadCalls != loadCalls {
				t.Errorf("load calls does not match: want=%v, got=%v", test.wantLoadCalls, loadCalls)
			}

			if test.wantHashCalls != hashCalls {
				t.Errorf("hash calls does not match: want=%v, got=%v", test.wantHashCalls, hashCalls)
			}

			if test.wantTransformerClosed != queue.wasCanceled {
				t.Errorf("transformer closed does not match: want=%v, got=%v", test.wantTransformerClosed, queue.wasCanceled)
			}

			if test.wantAddRateLimitedCount != queue.addRateLimitedCount.Load() {
				t.Errorf("queue addRateLimitedCount does not match: want=%v, got=%v", test.wantAddRateLimitedCount, queue.addRateLimitedCount.Load())
			}

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.wantMetrics),
				"apiserver_encryption_config_controller_automatic_reload_success_total",
				"apiserver_encryption_config_controller_automatic_reload_failures_total",
				"apiserver_encryption_config_controller_automatic_reloads_total",
			); err != nil {
				t.Errorf("failed to validate metrics: %v", err)
			}
		})
	}
}

type mockWorkQueue struct {
	workqueue.TypedRateLimitingInterface[string] // will panic if any unexpected method is called

	closeOnce sync.Once
	addCalled chan struct{}

	count       atomic.Uint64
	ctx         context.Context
	wasCanceled bool
	cancel      func()

	addRateLimitedCount atomic.Uint64
}

func (m *mockWorkQueue) Done(item string) {
	m.count.Add(1)
	m.wasCanceled = m.ctx.Err() != nil
	m.cancel()
}

func (m *mockWorkQueue) Get() (item string, shutdown bool) {
	<-m.addCalled

	switch m.count.Load() {
	case 0:
		return "", false
	case 1:
		return "", true
	default:
		panic("too many calls to Get")
	}
}

func (m *mockWorkQueue) Add(item string) {
	m.closeOnce.Do(func() {
		close(m.addCalled)
	})
}

func (m *mockWorkQueue) ShutDown()                  {}
func (m *mockWorkQueue) AddRateLimited(item string) { m.addRateLimitedCount.Add(1) }

type mockHealthChecker struct {
	pluginName string
	err        error
}

func (m *mockHealthChecker) Check(req *http.Request) error {
	return m.err
}

func (m *mockHealthChecker) Name() string {
	return m.pluginName
}
