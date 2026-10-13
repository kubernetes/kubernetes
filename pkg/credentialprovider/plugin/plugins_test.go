/*
Copyright 2025 The Kubernetes Authors.

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

package plugin

import (
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	credentialproviderapi "k8s.io/kubelet/pkg/apis/credentialprovider"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	testingclock "k8s.io/utils/clock/testing"
)

// fakedockerConfigProviderWithCoordinates implements dockerConfigProviderWithCoordinates for testing
type fakedockerConfigProviderWithCoordinates struct {
	dockerConfig         credentialprovider.DockerConfig
	serviceAccountCoords *credentialprovider.ServiceAccountCoordinates
	callCount            int
	lastImageRequested   string
	mu                   sync.Mutex
}

func (f *fakedockerConfigProviderWithCoordinates) provideWithCoordinates(image string) (credentialprovider.DockerConfig, *credentialprovider.ServiceAccountCoordinates) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.callCount++
	f.lastImageRequested = image

	return f.dockerConfig, f.serviceAccountCoords
}

func (f *fakedockerConfigProviderWithCoordinates) getCallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.callCount
}

func (f *fakedockerConfigProviderWithCoordinates) getLastImageRequested() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lastImageRequested
}

// createTestPluginProvider creates a pluginProvider for testing
func createTestPluginProvider(dockerConfig credentialprovider.DockerConfig) *pluginProvider {
	testClock := testingclock.NewFakeClock(time.Now())

	authConfigMap := make(map[string]credentialproviderapi.AuthConfig)
	for registry, config := range dockerConfig {
		authConfigMap[registry] = credentialproviderapi.AuthConfig{
			Username: config.Username,
			Password: config.Password,
		}
	}

	mockPlugin := &fakeExecPlugin{
		cacheKeyType:  credentialproviderapi.ImagePluginCacheKeyType,
		cacheDuration: time.Hour,
		auth:          authConfigMap,
	}

	provider := &pluginProvider{
		clock:                testClock,
		matchImages:          []string{"*"},
		cache:                cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{clock: testClock}),
		defaultCacheDuration: time.Hour,
		lastCachePurge:       testClock.Now(),
		plugin:               mockPlugin,
	}

	return provider
}

// setupTestProviders sets up test providers and cleans up after the test
func setupTestProviders(t *testing.T) {
	t.Helper()

	// Save original state
	providersMutex.Lock()
	originalProviders := providers
	originalSeenProviderNames := seenProviderNames

	// Reset to clean state
	providers = make([]provider, 0)
	seenProviderNames = sets.NewString()
	providersMutex.Unlock()

	t.Cleanup(func() {
		// Restore original state
		providersMutex.Lock()
		providers = originalProviders
		seenProviderNames = originalSeenProviderNames
		providersMutex.Unlock()
	})
}

func TestRegisterCredentialProviderPlugin(t *testing.T) {
	testCases := []struct {
		name           string
		firstProvider  string
		secondProvider string // empty means no second provider
		shouldPanic    bool
	}{
		{
			name:          "successful registration",
			firstProvider: "test-provider",
			shouldPanic:   false,
		},
		{
			name:           "duplicate registration should panic",
			firstProvider:  "duplicate-provider",
			secondProvider: "duplicate-provider",
			shouldPanic:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			setupTestProviders(t)

			mockProvider1 := createTestPluginProvider(credentialprovider.DockerConfig{
				"test.registry.io": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "pass",
				},
			})

			// Register first provider
			registerCredentialProviderPlugin(tc.firstProvider, mockProvider1)

			// Verify first registration
			providersMutex.RLock()
			if len(providers) != 1 {
				t.Errorf("Expected 1 provider after first registration, got %d", len(providers))
			}
			if providers[0].name != tc.firstProvider {
				t.Errorf("Expected provider name '%s', got %s", tc.firstProvider, providers[0].name)
			}
			if providers[0].impl != mockProvider1 {
				t.Errorf("Expected provider implementation to match")
			}
			if !seenProviderNames.Has(tc.firstProvider) {
				t.Errorf("Expected '%s' to be in seenProviderNames", tc.firstProvider)
			}
			providersMutex.RUnlock()

			// If we have a second provider to test, register it
			if tc.secondProvider != "" {
				mockProvider2 := createTestPluginProvider(credentialprovider.DockerConfig{})

				if tc.shouldPanic {
					// Test that registering duplicate provider name panics
					defer func() {
						if r := recover(); r == nil {
							t.Errorf("Expected panic for duplicate provider registration")
						}
					}()
				}

				registerCredentialProviderPlugin(tc.secondProvider, mockProvider2)

				if !tc.shouldPanic {
					// If we didn't expect a panic, verify the second provider was registered
					providersMutex.RLock()
					if len(providers) != 2 {
						t.Errorf("Expected 2 providers after second registration, got %d", len(providers))
					}
					providersMutex.RUnlock()
				}
			}
		})
	}
}

func TestNewExternalCredentialProviderDockerKeyring(t *testing.T) {
	testCases := []struct {
		name                         string
		setupProviders               func()
		featureGateEnabled           bool
		expectedProviderCount        int
		expectedPodNamespace         string
		expectedPodName              string
		expectedPodUID               string
		expectedServiceAccountName   string
		validatePerPodProviderFields bool
	}{
		{
			name:                  "no providers registered",
			setupProviders:        func() {},
			expectedProviderCount: 0,
		},
		{
			name: "multiple providers registered",
			setupProviders: func() {
				mockProvider1 := createTestPluginProvider(credentialprovider.DockerConfig{})
				mockProvider2 := createTestPluginProvider(credentialprovider.DockerConfig{})
				registerCredentialProviderPlugin("enabled-provider-1", mockProvider1)
				registerCredentialProviderPlugin("enabled-provider-2", mockProvider2)
			},
			expectedProviderCount: 2,
		},
		{
			name: "feature gate enabled - pod information should be set",
			setupProviders: func() {
				mockProvider := createTestPluginProvider(credentialprovider.DockerConfig{})
				registerCredentialProviderPlugin("test-provider", mockProvider)
			},
			featureGateEnabled:           true,
			expectedProviderCount:        1,
			expectedPodNamespace:         "test-namespace",
			expectedPodName:              "test-pod",
			expectedPodUID:               "test-uid",
			expectedServiceAccountName:   "test-sa",
			validatePerPodProviderFields: true,
		},
		{
			name: "feature gate disabled - pod information should be empty",
			setupProviders: func() {
				mockProvider := createTestPluginProvider(credentialprovider.DockerConfig{})
				registerCredentialProviderPlugin("test-provider", mockProvider)
			},
			featureGateEnabled:           false,
			expectedProviderCount:        1,
			expectedPodNamespace:         "",
			expectedPodName:              "",
			expectedPodUID:               "",
			expectedServiceAccountName:   "",
			validatePerPodProviderFields: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			setupTestProviders(t)

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletServiceAccountTokenForCredentialProviders, tc.featureGateEnabled)

			tc.setupProviders()

			keyring := NewExternalCredentialProviderDockerKeyring("test-namespace", "test-pod", "test-uid", "test-sa")

			externalKeyring, ok := keyring.(*externalCredentialProviderKeyring)
			if !ok {
				t.Fatalf("Expected externalCredentialProviderKeyring, got %T", keyring)
			}

			if len(externalKeyring.providers) != tc.expectedProviderCount {
				t.Errorf("Expected %d providers, got %d", tc.expectedProviderCount, len(externalKeyring.providers))
			}

			if tc.validatePerPodProviderFields && len(externalKeyring.providers) > 0 {
				perPodProvider, ok := externalKeyring.providers[0].(*perPodPluginProvider)
				if !ok {
					t.Fatalf("Expected perPodPluginProvider, got %T", externalKeyring.providers[0])
				}

				if perPodProvider.podNamespace != tc.expectedPodNamespace {
					t.Errorf("Expected podNamespace '%s', got %s", tc.expectedPodNamespace, perPodProvider.podNamespace)
				}
				if perPodProvider.podName != tc.expectedPodName {
					t.Errorf("Expected podName '%s', got %s", tc.expectedPodName, perPodProvider.podName)
				}
				if string(perPodProvider.podUID) != tc.expectedPodUID {
					t.Errorf("Expected podUID '%s', got %s", tc.expectedPodUID, string(perPodProvider.podUID))
				}
				if perPodProvider.serviceAccountName != tc.expectedServiceAccountName {
					t.Errorf("Expected serviceAccountName '%s', got %s", tc.expectedServiceAccountName, perPodProvider.serviceAccountName)
				}
			}
		})
	}
}

func TestExternalCredentialProviderKeyringLookupNoProviders(t *testing.T) {
	keyring := &externalCredentialProviderKeyring{
		providers: []dockerConfigProviderWithCoordinates{},
	}

	configs, found := keyring.Lookup("test.registry.io/image:tag")

	if found {
		t.Errorf("Expected not found, got found=true")
	}
	if len(configs) != 0 {
		t.Errorf("Expected 0 configs, got %d", len(configs))
	}
}

func TestExternalCredentialProviderKeyringLookupWithProviders(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletEnsureSecretPulledImages, true)

	testCases := []struct {
		name                     string
		image                    string
		providers                []dockerConfigProviderWithCoordinates
		expectedConfigCount      int
		expectedFound            bool
		expectedServiceAccountOK bool
	}{
		{
			name:  "single provider with config",
			image: "test.registry.io/image:tag",
			providers: []dockerConfigProviderWithCoordinates{
				&fakedockerConfigProviderWithCoordinates{
					dockerConfig: credentialprovider.DockerConfig{
						"test.registry.io": credentialprovider.DockerConfigEntry{
							Username: "user1",
							Password: "pass1",
						},
					},
					serviceAccountCoords: nil,
				},
			},
			expectedConfigCount:      1,
			expectedFound:            true,
			expectedServiceAccountOK: false,
		},
		{
			name:  "single provider with service account coordinates",
			image: "test.registry.io/image:tag",
			providers: []dockerConfigProviderWithCoordinates{
				&fakedockerConfigProviderWithCoordinates{
					dockerConfig: credentialprovider.DockerConfig{
						"test.registry.io": credentialprovider.DockerConfigEntry{
							Username: "user1",
							Password: "pass1",
						},
					},
					serviceAccountCoords: &credentialprovider.ServiceAccountCoordinates{
						Namespace: "test-namespace",
						Name:      "test-sa",
						UID:       "test-uid",
					},
				},
			},
			expectedConfigCount:      1,
			expectedFound:            true,
			expectedServiceAccountOK: true,
		},
		{
			name:  "multiple providers",
			image: "test.registry.io/image:tag",
			providers: []dockerConfigProviderWithCoordinates{
				&fakedockerConfigProviderWithCoordinates{
					dockerConfig: credentialprovider.DockerConfig{
						"test.registry.io": credentialprovider.DockerConfigEntry{
							Username: "user1",
							Password: "pass1",
						},
					},
					serviceAccountCoords: &credentialprovider.ServiceAccountCoordinates{
						Namespace: "test-namespace",
						Name:      "test-sa-1",
						UID:       "test-uid-1",
					},
				},
				&fakedockerConfigProviderWithCoordinates{
					dockerConfig: credentialprovider.DockerConfig{
						"test.registry.io": credentialprovider.DockerConfigEntry{
							Username: "user2",
							Password: "pass2",
						},
					},
					serviceAccountCoords: &credentialprovider.ServiceAccountCoordinates{
						Namespace: "test-namespace",
						Name:      "test-sa-2",
						UID:       "test-uid-2",
					},
				},
			},
			expectedConfigCount:      2,
			expectedFound:            true,
			expectedServiceAccountOK: true,
		},
		{
			name:  "provider with empty config",
			image: "test.registry.io/image:tag",
			providers: []dockerConfigProviderWithCoordinates{
				&fakedockerConfigProviderWithCoordinates{
					dockerConfig:         credentialprovider.DockerConfig{},
					serviceAccountCoords: nil,
				},
			},
			expectedConfigCount:      0,
			expectedFound:            false,
			expectedServiceAccountOK: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			keyring := &externalCredentialProviderKeyring{
				providers: tc.providers,
			}

			configs, found := keyring.Lookup(tc.image)

			if found != tc.expectedFound {
				t.Errorf("Expected found=%v, got found=%v", tc.expectedFound, found)
			}

			if len(configs) != tc.expectedConfigCount {
				t.Errorf("Expected %d configs, got %d", tc.expectedConfigCount, len(configs))
			}

			// Verify that each provider was called with the correct image
			for i, provider := range tc.providers {
				mockProvider := provider.(*fakedockerConfigProviderWithCoordinates)
				if mockProvider.getCallCount() != 1 {
					t.Errorf("Provider %d expected 1 call, got %d", i, mockProvider.getCallCount())
				}
				if mockProvider.getLastImageRequested() != tc.image {
					t.Errorf("Provider %d expected image %s, got %s", i, tc.image, mockProvider.getLastImageRequested())
				}
			}

			// Verify service account coordinates in TrackedAuthConfig
			if tc.expectedServiceAccountOK && len(configs) > 0 {
				foundServiceAccountCoords := false
				for _, config := range configs {
					if config.Source != nil && config.Source.ServiceAccount != nil {
						foundServiceAccountCoords = true
						break
					}
				}
				if !foundServiceAccountCoords {
					t.Errorf("Expected to find service account coordinates in TrackedAuthConfig")
				}
			}
		})
	}
}

func TestExternalCredentialProviderKeyringLookupConcurrency(t *testing.T) {
	// Test concurrent access to the keyring
	mockProvider := &fakedockerConfigProviderWithCoordinates{
		dockerConfig: credentialprovider.DockerConfig{
			"test.registry.io": credentialprovider.DockerConfigEntry{
				Username: "user1",
				Password: "pass1",
			},
		},
		serviceAccountCoords: &credentialprovider.ServiceAccountCoordinates{
			Namespace: "test-namespace",
			Name:      "test-sa",
			UID:       "test-uid",
		},
	}

	keyring := &externalCredentialProviderKeyring{
		providers: []dockerConfigProviderWithCoordinates{mockProvider},
	}

	const numGoroutines = 10
	const numCallsPerGoroutine = 5

	var wg sync.WaitGroup
	var errorOccurred bool
	var mu sync.Mutex

	wg.Add(numGoroutines)

	for i := range numGoroutines {
		go func(goroutineID int) {
			defer wg.Done()
			for j := range numCallsPerGoroutine {
				image := "test.registry.io/image:tag"
				configs, found := keyring.Lookup(image)

				if !found {
					mu.Lock()
					errorOccurred = true
					t.Errorf("Goroutine %d call %d: Expected found=true, got found=false", goroutineID, j)
					mu.Unlock()
					return
				}
				if len(configs) != 1 {
					mu.Lock()
					errorOccurred = true
					t.Errorf("Goroutine %d call %d: Expected 1 config, got %d", goroutineID, j, len(configs))
					mu.Unlock()
					return
				}
			}
		}(i)
	}

	wg.Wait()

	if !errorOccurred {
		expectedTotalCalls := numGoroutines * numCallsPerGoroutine
		actualCalls := mockProvider.getCallCount()
		if actualCalls != expectedTotalCalls {
			t.Errorf("Expected %d total calls, got %d", expectedTotalCalls, actualCalls)
		}
	}
}
