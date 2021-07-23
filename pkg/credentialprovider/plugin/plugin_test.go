/*
Copyright 2020 The Kubernetes Authors.

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
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"
	credentialproviderapi "k8s.io/kubelet/pkg/apis/credentialprovider"
	credentialproviderv1alpha1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

type fakeExecPlugin struct {
	cacheKeyType  credentialproviderapi.PluginCacheKeyType
	cacheDuration time.Duration

	auth map[string]credentialproviderapi.AuthConfig
}

func (f *fakeExecPlugin) ExecPlugin(ctx context.Context, image string) (*credentialproviderapi.CredentialProviderResponse, error) {
	return &credentialproviderapi.CredentialProviderResponse{
		CacheKeyType: f.cacheKeyType,
		CacheDuration: &metav1.Duration{
			Duration: f.cacheDuration,
		},
		Auth: f.auth,
	}, nil
}

func Test_Provide(t *testing.T) {
	testcases := []struct {
		name           string
		pluginProvider *pluginProvider
		image          string
		dockerconfig   credentialprovider.DockerConfig
	}{
		{
			name: "exact image match, with Registry cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"test.registry.io"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.RegistryPluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"test.registry.io": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "test.registry.io/foo/bar",
			dockerconfig: credentialprovider.DockerConfig{
				"test.registry.io": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
		{
			name: "exact image match, with Image cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"test.registry.io/foo/bar"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.ImagePluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"test.registry.io/foo/bar": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "test.registry.io/foo/bar",
			dockerconfig: credentialprovider.DockerConfig{
				"test.registry.io/foo/bar": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
		{
			name: "exact image match, with Global cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"test.registry.io"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.GlobalPluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"test.registry.io": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "test.registry.io",
			dockerconfig: credentialprovider.DockerConfig{
				"test.registry.io": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
		{
			name: "wild card image match, with Registry cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"*.registry.io:8080"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.RegistryPluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"*.registry.io:8080": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "test.registry.io:8080/foo",
			dockerconfig: credentialprovider.DockerConfig{
				"*.registry.io:8080": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
		{
			name: "wild card image match, with Image cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"*.*.registry.io"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.ImagePluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"*.*.registry.io": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "foo.bar.registry.io/foo/bar",
			dockerconfig: credentialprovider.DockerConfig{
				"*.*.registry.io": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
		{
			name: "wild card image match, with Global cache key",
			pluginProvider: &pluginProvider{
				matchImages: []string{"*.registry.io"},
				cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
				plugin: &fakeExecPlugin{
					cacheKeyType: credentialproviderapi.GlobalPluginCacheKeyType,
					auth: map[string]credentialproviderapi.AuthConfig{
						"*.registry.io": {
							Username: "user",
							Password: "password",
						},
					},
				},
			},
			image: "test.registry.io",
			dockerconfig: credentialprovider.DockerConfig{
				"*.registry.io": credentialprovider.DockerConfigEntry{
					Username: "user",
					Password: "password",
				},
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			dockerconfig := testcase.pluginProvider.Provide(testcase.image)
			if !reflect.DeepEqual(dockerconfig, testcase.dockerconfig) {
				t.Logf("actual docker config: %v", dockerconfig)
				t.Logf("expected docker config: %v", testcase.dockerconfig)
				t.Error("unexpected docker config")
			}
		})
	}
}

func Test_encodeRequest(t *testing.T) {
	testcases := []struct {
		name         string
		apiVersion   string
		request      *credentialproviderapi.CredentialProviderRequest
		expectedData []byte
		expectedErr  bool
	}{
		{
			name: "successful",
			request: &credentialproviderapi.CredentialProviderRequest{
				Image: "test.registry.io/foobar",
			},
			expectedData: []byte(`{"kind":"CredentialProviderRequest","apiVersion":"credentialprovider.kubelet.k8s.io/v1alpha1","image":"test.registry.io/foobar"}
`),
			expectedErr: false,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			mediaType := "application/json"
			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
			if !ok {
				t.Fatalf("unsupported media type: %s", mediaType)
			}

			e := &execPlugin{
				encoder: codecs.EncoderForVersion(info.Serializer, credentialproviderv1alpha1.SchemeGroupVersion),
			}

			data, err := e.encodeRequest(testcase.request)
			if err != nil && !testcase.expectedErr {
				t.Fatalf("unexpected error: %v", err)
			}

			if err == nil && testcase.expectedErr {
				t.Fatalf("expected error %v but got nil", testcase.expectedErr)
			}

			if !reflect.DeepEqual(data, testcase.expectedData) {
				t.Errorf("actual encoded data: %v", string(data))
				t.Errorf("expected encoded data: %v", string(testcase.expectedData))
				t.Errorf("unexpected encoded response")
			}
		})
	}
}

func Test_decodeResponse(t *testing.T) {
	testcases := []struct {
		name             string
		data             []byte
		expectedResponse *credentialproviderapi.CredentialProviderResponse
		expectedErr      bool
	}{
		{
			name: "success",
			data: []byte(`{"kind":"CredentialProviderResponse","apiVersion":"credentialprovider.kubelet.k8s.io/v1alpha1","cacheKeyType":"Registry","cacheDuration":"1m","auth":{"*.registry.io":{"username":"user","password":"password"}}}`),
			expectedResponse: &credentialproviderapi.CredentialProviderResponse{
				CacheKeyType: credentialproviderapi.RegistryPluginCacheKeyType,
				CacheDuration: &metav1.Duration{
					Duration: time.Minute,
				},
				Auth: map[string]credentialproviderapi.AuthConfig{
					"*.registry.io": {
						Username: "user",
						Password: "password",
					},
				},
			},
			expectedErr: false,
		},
		{
			name:             "wrong Kind",
			data:             []byte(`{"kind":"WrongKind","apiVersion":"credentialprovider.kubelet.k8s.io/v1alpha1","cacheKeyType":"Registry","cacheDuration":"1m","auth":{"*.registry.io":{"username":"user","password":"password"}}}`),
			expectedResponse: nil,
			expectedErr:      true,
		},
		{
			name:             "wrong Group",
			data:             []byte(`{"kind":"CredentialProviderResponse","apiVersion":"foobar.kubelet.k8s.io/v1alpha1","cacheKeyType":"Registry","cacheDuration":"1m","auth":{"*.registry.io":{"username":"user","password":"password"}}}`),
			expectedResponse: nil,
			expectedErr:      true,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			e := &execPlugin{}

			decodedResponse, err := e.decodeResponse(testcase.data)
			if err != nil && !testcase.expectedErr {
				t.Fatalf("unexpected error: %v", err)
			}

			if err == nil && testcase.expectedErr {
				t.Fatalf("expected error %v but not nil", testcase.expectedErr)
			}

			if !reflect.DeepEqual(decodedResponse, testcase.expectedResponse) {
				t.Logf("actual decoded response: %#v", decodedResponse)
				t.Logf("expected decoded response: %#v", testcase.expectedResponse)
				t.Errorf("unexpected decoded response")
			}
		})
	}
}

func Test_RegistryCacheKeyType(t *testing.T) {
	pluginProvider := &pluginProvider{
		matchImages: []string{"*.registry.io"},
		cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
		plugin: &fakeExecPlugin{
			cacheKeyType:  credentialproviderapi.RegistryPluginCacheKeyType,
			cacheDuration: time.Hour,
			auth: map[string]credentialproviderapi.AuthConfig{
				"*.registry.io": {
					Username: "user",
					Password: "password",
				},
			},
		},
	}

	expectedDockerConfig := credentialprovider.DockerConfig{
		"*.registry.io": credentialprovider.DockerConfigEntry{
			Username: "user",
			Password: "password",
		},
	}

	dockerConfig := pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}

	expectedCacheKeys := []string{"test.registry.io"}
	cacheKeys := pluginProvider.cache.ListKeys()

	if !reflect.DeepEqual(cacheKeys, expectedCacheKeys) {
		t.Logf("actual cache keys: %v", cacheKeys)
		t.Logf("expected cache keys: %v", expectedCacheKeys)
		t.Error("unexpected cache keys")
	}

	// nil out the exec plugin, this will test whether credentialproviderapi are fetched
	// from cache, otherwise Provider should panic
	pluginProvider.plugin = nil
	dockerConfig = pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}
}

func Test_ImageCacheKeyType(t *testing.T) {
	pluginProvider := &pluginProvider{
		matchImages: []string{"*.registry.io"},
		cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
		plugin: &fakeExecPlugin{
			cacheKeyType:  credentialproviderapi.ImagePluginCacheKeyType,
			cacheDuration: time.Hour,
			auth: map[string]credentialproviderapi.AuthConfig{
				"*.registry.io": {
					Username: "user",
					Password: "password",
				},
			},
		},
	}

	expectedDockerConfig := credentialprovider.DockerConfig{
		"*.registry.io": credentialprovider.DockerConfigEntry{
			Username: "user",
			Password: "password",
		},
	}

	dockerConfig := pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}

	expectedCacheKeys := []string{"test.registry.io/foo/bar"}
	cacheKeys := pluginProvider.cache.ListKeys()

	if !reflect.DeepEqual(cacheKeys, expectedCacheKeys) {
		t.Logf("actual cache keys: %v", cacheKeys)
		t.Logf("expected cache keys: %v", expectedCacheKeys)
		t.Error("unexpected cache keys")
	}

	// nil out the exec plugin, this will test whether credentialproviderapi are fetched
	// from cache, otherwise Provider should panic
	pluginProvider.plugin = nil
	dockerConfig = pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}
}

func Test_GlobalCacheKeyType(t *testing.T) {
	pluginProvider := &pluginProvider{
		matchImages: []string{"*.registry.io"},
		cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
		plugin: &fakeExecPlugin{
			cacheKeyType:  credentialproviderapi.GlobalPluginCacheKeyType,
			cacheDuration: time.Hour,
			auth: map[string]credentialproviderapi.AuthConfig{
				"*.registry.io": {
					Username: "user",
					Password: "password",
				},
			},
		},
	}

	expectedDockerConfig := credentialprovider.DockerConfig{
		"*.registry.io": credentialprovider.DockerConfigEntry{
			Username: "user",
			Password: "password",
		},
	}

	dockerConfig := pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}

	expectedCacheKeys := []string{"global"}
	cacheKeys := pluginProvider.cache.ListKeys()

	if !reflect.DeepEqual(cacheKeys, expectedCacheKeys) {
		t.Logf("actual cache keys: %v", cacheKeys)
		t.Logf("expected cache keys: %v", expectedCacheKeys)
		t.Error("unexpected cache keys")
	}

	// nil out the exec plugin, this will test whether credentialproviderapi are fetched
	// from cache, otherwise Provider should panic
	pluginProvider.plugin = nil
	dockerConfig = pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}
}

func Test_NoCacheResponse(t *testing.T) {
	pluginProvider := &pluginProvider{
		matchImages: []string{"*.registry.io"},
		cache:       cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{}),
		plugin: &fakeExecPlugin{
			cacheKeyType:  credentialproviderapi.GlobalPluginCacheKeyType,
			cacheDuration: 0, // no cache
			auth: map[string]credentialproviderapi.AuthConfig{
				"*.registry.io": {
					Username: "user",
					Password: "password",
				},
			},
		},
	}

	expectedDockerConfig := credentialprovider.DockerConfig{
		"*.registry.io": credentialprovider.DockerConfigEntry{
			Username: "user",
			Password: "password",
		},
	}

	dockerConfig := pluginProvider.Provide("test.registry.io/foo/bar")
	if !reflect.DeepEqual(dockerConfig, expectedDockerConfig) {
		t.Logf("actual docker config: %v", dockerConfig)
		t.Logf("expected docker config: %v", expectedDockerConfig)
		t.Fatal("unexpected docker config")
	}

	expectedCacheKeys := []string{}
	cacheKeys := pluginProvider.cache.ListKeys()
	if !reflect.DeepEqual(cacheKeys, expectedCacheKeys) {
		t.Logf("actual cache keys: %v", cacheKeys)
		t.Logf("expected cache keys: %v", expectedCacheKeys)
		t.Error("unexpected cache keys")
	}
}

func Test_ExecPluginEnvVars(t *testing.T) {
	testcases := []struct {
		name            string
		systemEnvVars   []string
		execPlugin      *execPlugin
		expectedEnvVars []string
	}{
		{
			name:          "positive append system env vars",
			systemEnvVars: []string{"HOME=/home/foo", "PATH=/usr/bin"},
			execPlugin: &execPlugin{
				envVars: []kubeletconfig.ExecEnvVar{
					{
						Name:  "SUPER_SECRET_STRONG_ACCESS_KEY",
						Value: "123456789",
					},
				},
			},
			expectedEnvVars: []string{
				"HOME=/home/foo",
				"PATH=/usr/bin",
				"SUPER_SECRET_STRONG_ACCESS_KEY=123456789",
			},
		},
		{
			name:          "positive no env vars provided in plugin",
			systemEnvVars: []string{"HOME=/home/foo", "PATH=/usr/bin"},
			execPlugin:    &execPlugin{},
			expectedEnvVars: []string{
				"HOME=/home/foo",
				"PATH=/usr/bin",
			},
		},
		{
			name: "positive no system env vars but env vars are provided in plugin",
			execPlugin: &execPlugin{
				envVars: []kubeletconfig.ExecEnvVar{
					{
						Name:  "SUPER_SECRET_STRONG_ACCESS_KEY",
						Value: "123456789",
					},
				},
			},
			expectedEnvVars: []string{
				"SUPER_SECRET_STRONG_ACCESS_KEY=123456789",
			},
		},
		{
			name:            "positive no system or plugin provided env vars",
			execPlugin:      &execPlugin{},
			expectedEnvVars: nil,
		},
		{
			name:          "positive plugin provided vars takes priority",
			systemEnvVars: []string{"HOME=/home/foo", "PATH=/usr/bin", "SUPER_SECRET_STRONG_ACCESS_KEY=1111"},
			execPlugin: &execPlugin{
				envVars: []kubeletconfig.ExecEnvVar{
					{
						Name:  "SUPER_SECRET_STRONG_ACCESS_KEY",
						Value: "123456789",
					},
				},
			},
			expectedEnvVars: []string{
				"HOME=/home/foo",
				"PATH=/usr/bin",
				"SUPER_SECRET_STRONG_ACCESS_KEY=1111",
				"SUPER_SECRET_STRONG_ACCESS_KEY=123456789",
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			testcase.execPlugin.environ = func() []string {
				return testcase.systemEnvVars
			}

			var configVars []string
			for _, envVar := range testcase.execPlugin.envVars {
				configVars = append(configVars, fmt.Sprintf("%s=%s", envVar.Name, envVar.Value))
			}
			merged := mergeEnvVars(testcase.systemEnvVars, configVars)

			err := validate(testcase.expectedEnvVars, merged)
			if err != nil {
				t.Logf("unexpecged error %v", err)
			}
		})
	}
}

func validate(expected, actual []string) error {
	if len(actual) != len(expected) {
		return errors.New(fmt.Sprintf("actual env var length [%d] and expected env var length [%d] don't match",
			len(actual), len(expected)))
	}

	for i := range actual {
		if actual[i] != expected[i] {
			return fmt.Errorf("mismatch in expected env var %s and actual env var %s", actual[i], expected[i])
		}
	}

	return nil
}
