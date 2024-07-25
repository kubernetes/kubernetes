/*
Copyright 2016 The Kubernetes Authors.

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

package rest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/flowcontrol"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
)

func TestIsConfigTransportTLS(t *testing.T) {
	testCases := []struct {
		Config       *Config
		TransportTLS bool
	}{
		{
			Config:       &Config{},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host: "https://localhost",
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "///:://localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host: "1.2.3.4:567",
				TLSClientConfig: TLSClientConfig{
					Insecure: true,
				},
			},
			TransportTLS: true,
		},
	}
	for _, testCase := range testCases {
		if err := SetKubernetesDefaults(testCase.Config); err != nil {
			t.Errorf("setting defaults failed for %#v: %v", testCase.Config, err)
			continue
		}
		useTLS := IsConfigTransportTLS(*testCase.Config)
		if testCase.TransportTLS != useTLS {
			t.Errorf("expected %v for %#v", testCase.TransportTLS, testCase.Config)
		}
	}
}

func TestSetKubernetesDefaultsUserAgent(t *testing.T) {
	config := &Config{}
	if err := SetKubernetesDefaults(config); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(config.UserAgent, "kubernetes/") {
		t.Errorf("no user agent set: %#v", config)
	}
}

func TestAdjustVersion(t *testing.T) {
	assert := assert.New(t)
	assert.Equal("1.2.3", adjustVersion("1.2.3-alpha4"))
	assert.Equal("1.2.3", adjustVersion("1.2.3-alpha"))
	assert.Equal("1.2.3", adjustVersion("1.2.3"))
	assert.Equal("unknown", adjustVersion(""))
}

func TestAdjustCommit(t *testing.T) {
	assert := assert.New(t)
	assert.Equal("1234567", adjustCommit("1234567890"))
	assert.Equal("123456", adjustCommit("123456"))
	assert.Equal("unknown", adjustCommit(""))
}

func TestAdjustCommand(t *testing.T) {
	assert := assert.New(t)
	assert.Equal("beans", adjustCommand(filepath.Join("home", "bob", "Downloads", "beans")))
	assert.Equal("beans", adjustCommand(filepath.Join(".", "beans")))
	assert.Equal("beans", adjustCommand("beans"))
	assert.Equal("unknown", adjustCommand(""))
}

func TestBuildUserAgent(t *testing.T) {
	assert.New(t).Equal(
		"lynx/nicest (beos/itanium) kubernetes/baaaaaaaaad",
		buildUserAgent(
			"lynx", "nicest",
			"beos", "itanium", "baaaaaaaaad"))
}

// This function untestable since it doesn't accept arguments.
func TestDefaultKubernetesUserAgent(t *testing.T) {
	assert.New(t).Contains(DefaultKubernetesUserAgent(), "kubernetes")
}

func TestRESTClientRequires(t *testing.T) {
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{NegotiatedSerializer: scheme.Codecs}}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{GroupVersion: &v1.SchemeGroupVersion}}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{GroupVersion: &v1.SchemeGroupVersion, NegotiatedSerializer: scheme.Codecs}}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRESTClientLimiter(t *testing.T) {
	testCases := []struct {
		Name    string
		Config  Config
		Limiter flowcontrol.RateLimiter
	}{
		{
			Name:    "with no QPS",
			Config:  Config{},
			Limiter: flowcontrol.NewTokenBucketRateLimiter(5, 10),
		},
		{
			Name:    "with QPS:10",
			Config:  Config{QPS: 10},
			Limiter: flowcontrol.NewTokenBucketRateLimiter(10, 10),
		},
		{
			Name:    "with QPS:-1",
			Config:  Config{QPS: -1},
			Limiter: nil,
		},
		{
			Name: "with RateLimiter",
			Config: Config{
				RateLimiter: flowcontrol.NewTokenBucketRateLimiter(11, 12),
			},
			Limiter: flowcontrol.NewTokenBucketRateLimiter(11, 12),
		},
	}
	for _, testCase := range testCases {
		t.Run("Versioned_"+testCase.Name, func(t *testing.T) {
			config := testCase.Config
			config.Host = "127.0.0.1"
			config.ContentConfig = ContentConfig{GroupVersion: &v1.SchemeGroupVersion, NegotiatedSerializer: scheme.Codecs}
			client, err := RESTClientFor(&config)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(testCase.Limiter, client.rateLimiter) {
				t.Fatalf("unexpected rate limiter: %#v, expected %#v at %s", client.rateLimiter, testCase.Limiter, testCase.Name)
			}
		})
		t.Run("Unversioned_"+testCase.Name, func(t *testing.T) {
			config := testCase.Config
			config.Host = "127.0.0.1"
			config.ContentConfig = ContentConfig{GroupVersion: &v1.SchemeGroupVersion, NegotiatedSerializer: scheme.Codecs}
			client, err := UnversionedRESTClientFor(&config)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(testCase.Limiter, client.rateLimiter) {
				t.Fatalf("unexpected rate limiter: %#v, expected %#v at %s", client.rateLimiter, testCase.Limiter, testCase.Name)
			}
		})
	}
}

type fakeLimiter struct {
	FakeSaturation float64
	FakeQPS        float32
}

func (t *fakeLimiter) TryAccept() bool {
	return true
}

func (t *fakeLimiter) Saturation() float64 {
	return t.FakeSaturation
}

func (t *fakeLimiter) QPS() float32 {
	return t.FakeQPS
}

func (t *fakeLimiter) Wait(ctx context.Context) error {
	return nil
}

func (t *fakeLimiter) Stop() {}

func (t *fakeLimiter) Accept() {}

type fakeCodec struct{}

func (c *fakeCodec) Decode([]byte, *schema.GroupVersionKind, runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	return nil, nil, nil
}

func (c *fakeCodec) Encode(obj runtime.Object, stream io.Writer) error {
	return nil
}

func (c *fakeCodec) Identifier() runtime.Identifier {
	return runtime.Identifier("fake")
}

type fakeRoundTripper struct{}

func (r *fakeRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}

var fakeWrapperFunc = func(http.RoundTripper) http.RoundTripper {
	return &fakeRoundTripper{}
}

type fakeWarningHandler struct{}

func (f fakeWarningHandler) HandleWarningHeader(code int, agent string, message string) {}

type fakeNegotiatedSerializer struct{}

func (n *fakeNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return nil
}

func (n *fakeNegotiatedSerializer) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return &fakeCodec{}
}

func (n *fakeNegotiatedSerializer) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return &fakeCodec{}
}

var fakeDialFunc = func(ctx context.Context, network, addr string) (net.Conn, error) {
	return nil, fakeDialerError
}

var fakeDialerError = errors.New("fakedialer")

func fakeProxyFunc(*http.Request) (*url.URL, error) {
	return nil, errors.New("fakeproxy")
}

type fakeAuthProviderConfigPersister struct{}

func (fakeAuthProviderConfigPersister) Persist(map[string]string) error {
	return fakeAuthProviderConfigPersisterError
}

var fakeAuthProviderConfigPersisterError = errors.New("fakeAuthProviderConfigPersisterError")

func TestAnonymousAuthConfig(t *testing.T) {
	f := fuzz.New().NilChance(0.0).NumElements(1, 1)
	f.Funcs(
		func(r *runtime.Codec, f fuzz.Continue) {
			codec := &fakeCodec{}
			f.Fuzz(codec)
			*r = codec
		},
		func(r *http.RoundTripper, f fuzz.Continue) {
			roundTripper := &fakeRoundTripper{}
			f.Fuzz(roundTripper)
			*r = roundTripper
		},
		func(fn *func(http.RoundTripper) http.RoundTripper, f fuzz.Continue) {
			*fn = fakeWrapperFunc
		},
		func(fn *transport.WrapperFunc, f fuzz.Continue) {
			*fn = fakeWrapperFunc
		},
		func(r *runtime.NegotiatedSerializer, f fuzz.Continue) {
			serializer := &fakeNegotiatedSerializer{}
			f.Fuzz(serializer)
			*r = serializer
		},
		func(r *flowcontrol.RateLimiter, f fuzz.Continue) {
			limiter := &fakeLimiter{}
			f.Fuzz(limiter)
			*r = limiter
		},
		func(h *WarningHandler, f fuzz.Continue) {
			*h = &fakeWarningHandler{}
		},
		// Authentication does not require fuzzer
		func(r *AuthProviderConfigPersister, f fuzz.Continue) {},
		func(r *clientcmdapi.AuthProviderConfig, f fuzz.Continue) {
			r.Config = map[string]string{}
		},
		func(r *func(ctx context.Context, network, addr string) (net.Conn, error), f fuzz.Continue) {
			*r = fakeDialFunc
		},
		func(r *func(*http.Request) (*url.URL, error), f fuzz.Continue) {
			*r = fakeProxyFunc
		},
		func(r *runtime.Object, f fuzz.Continue) {
			unknown := &runtime.Unknown{}
			f.Fuzz(unknown)
			*r = unknown
		},
	)
	for i := 0; i < 20; i++ {
		original := &Config{}
		f.Fuzz(original)
		actual := AnonymousClientConfig(original)
		expected := *original

		// this is the list of known security related fields, add to this list if a new field
		// is added to Config, update AnonymousClientConfig to preserve the field otherwise.
		expected.Impersonate = ImpersonationConfig{}
		expected.BearerToken = ""
		expected.BearerTokenFile = ""
		expected.Username = ""
		expected.Password = ""
		expected.AuthProvider = nil
		expected.AuthConfigPersister = nil
		expected.ExecProvider = nil
		expected.TLSClientConfig.CertData = nil
		expected.TLSClientConfig.CertFile = ""
		expected.TLSClientConfig.KeyData = nil
		expected.TLSClientConfig.KeyFile = ""
		expected.Transport = nil
		expected.WrapTransport = nil

		if actual.Dial != nil {
			_, actualError := actual.Dial(context.Background(), "", "")
			_, expectedError := expected.Dial(context.Background(), "", "")
			if !reflect.DeepEqual(expectedError, actualError) {
				t.Fatalf("AnonymousClientConfig dropped the Dial field")
			}
		}
		actual.Dial = nil
		expected.Dial = nil

		if actual.Proxy != nil {
			_, actualError := actual.Proxy(nil)
			_, expectedError := expected.Proxy(nil)
			if !reflect.DeepEqual(expectedError, actualError) {
				t.Fatalf("AnonymousClientConfig dropped the Proxy field")
			}
		}
		actual.Proxy = nil
		expected.Proxy = nil

		if diff := cmp.Diff(*actual, expected); diff != "" {
			t.Fatalf("AnonymousClientConfig dropped unexpected fields, identify whether they are security related or not (-got, +want): %s", diff)
		}
	}
}

func TestCopyConfig(t *testing.T) {
	f := fuzz.New().NilChance(0.0).NumElements(1, 1)
	f.Funcs(
		func(r *runtime.Codec, f fuzz.Continue) {
			codec := &fakeCodec{}
			f.Fuzz(codec)
			*r = codec
		},
		func(r *http.RoundTripper, f fuzz.Continue) {
			roundTripper := &fakeRoundTripper{}
			f.Fuzz(roundTripper)
			*r = roundTripper
		},
		func(fn *func(http.RoundTripper) http.RoundTripper, f fuzz.Continue) {
			*fn = fakeWrapperFunc
		},
		func(fn *transport.WrapperFunc, f fuzz.Continue) {
			*fn = fakeWrapperFunc
		},
		func(r *runtime.NegotiatedSerializer, f fuzz.Continue) {
			serializer := &fakeNegotiatedSerializer{}
			f.Fuzz(serializer)
			*r = serializer
		},
		func(r *flowcontrol.RateLimiter, f fuzz.Continue) {
			limiter := &fakeLimiter{}
			f.Fuzz(limiter)
			*r = limiter
		},
		func(h *WarningHandler, f fuzz.Continue) {
			*h = &fakeWarningHandler{}
		},
		func(r *AuthProviderConfigPersister, f fuzz.Continue) {
			*r = fakeAuthProviderConfigPersister{}
		},
		func(r *func(ctx context.Context, network, addr string) (net.Conn, error), f fuzz.Continue) {
			*r = fakeDialFunc
		},
		func(r *func(*http.Request) (*url.URL, error), f fuzz.Continue) {
			*r = fakeProxyFunc
		},
		func(r *runtime.Object, f fuzz.Continue) {
			unknown := &runtime.Unknown{}
			f.Fuzz(unknown)
			*r = unknown
		},
	)
	for i := 0; i < 20; i++ {
		original := &Config{}
		f.Fuzz(original)
		actual := CopyConfig(original)
		expected := *original

		// this is the list of known risky fields, add to this list if a new field
		// is added to Config, update CopyConfig to preserve the field otherwise.

		// The DeepEqual cannot handle the func comparison, so we just verify if the
		// function return the expected object.
		if actual.WrapTransport == nil || !reflect.DeepEqual(expected.WrapTransport(nil), &fakeRoundTripper{}) {
			t.Fatalf("CopyConfig dropped the WrapTransport field")
		}
		actual.WrapTransport = nil
		expected.WrapTransport = nil

		if actual.Dial != nil {
			_, actualError := actual.Dial(context.Background(), "", "")
			_, expectedError := expected.Dial(context.Background(), "", "")
			if !reflect.DeepEqual(expectedError, actualError) {
				t.Fatalf("CopyConfig  dropped the Dial field")
			}
		}
		actual.Dial = nil
		expected.Dial = nil

		if actual.AuthConfigPersister != nil {
			actualError := actual.AuthConfigPersister.Persist(nil)
			expectedError := expected.AuthConfigPersister.Persist(nil)
			if !reflect.DeepEqual(expectedError, actualError) {
				t.Fatalf("CopyConfig  dropped the Dial field")
			}
		}
		actual.AuthConfigPersister = nil
		expected.AuthConfigPersister = nil

		if actual.Proxy != nil {
			_, actualError := actual.Proxy(nil)
			_, expectedError := expected.Proxy(nil)
			if !reflect.DeepEqual(expectedError, actualError) {
				t.Fatalf("CopyConfig  dropped the Proxy field")
			}
		}
		actual.Proxy = nil
		expected.Proxy = nil

		if diff := cmp.Diff(*actual, expected); diff != "" {
			t.Fatalf("CopyConfig  dropped unexpected fields, identify whether they are security related or not (-got, +want): %s", diff)
		}
	}
}

func TestConfigStringer(t *testing.T) {
	formatBytes := func(b []byte) string {
		// %#v for []byte always pre-pends "[]byte{".
		// %#v for struct with []byte field always pre-pends "[]uint8{".
		return strings.Replace(fmt.Sprintf("%#v", b), "byte", "uint8", 1)
	}
	tests := []struct {
		desc            string
		c               *Config
		expectContent   []string
		prohibitContent []string
	}{
		{
			desc:          "nil config",
			c:             nil,
			expectContent: []string{"<nil>"},
		},
		{
			desc: "non-sensitive config",
			c: &Config{
				Host:      "localhost:8080",
				APIPath:   "v1",
				UserAgent: "gobot",
			},
			expectContent: []string{"localhost:8080", "v1", "gobot"},
		},
		{
			desc: "sensitive config",
			c: &Config{
				Host:        "localhost:8080",
				Username:    "gopher",
				Password:    "g0ph3r",
				BearerToken: "1234567890",
				TLSClientConfig: TLSClientConfig{
					CertFile: "a.crt",
					KeyFile:  "a.key",
					CertData: []byte("fake cert"),
					KeyData:  []byte("fake key"),
				},
				AuthProvider: &clientcmdapi.AuthProviderConfig{
					Config: map[string]string{"secret": "s3cr3t"},
				},
				ExecProvider: &clientcmdapi.ExecConfig{
					Args:   []string{"secret"},
					Env:    []clientcmdapi.ExecEnvVar{{Name: "secret", Value: "s3cr3t"}},
					Config: &runtime.Unknown{Raw: []byte("here is some config data")},
				},
			},
			expectContent: []string{
				"localhost:8080",
				"gopher",
				"a.crt",
				"a.key",
				"--- REDACTED ---",
				formatBytes([]byte("--- REDACTED ---")),
				formatBytes([]byte("--- TRUNCATED ---")),
			},
			prohibitContent: []string{
				"g0ph3r",
				"1234567890",
				formatBytes([]byte("fake cert")),
				formatBytes([]byte("fake key")),
				"secret",
				"s3cr3t",
				"here is some config data",
				formatBytes([]byte("super secret password")),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			got := tt.c.String()
			t.Logf("formatted config: %q", got)

			for _, expect := range tt.expectContent {
				if !strings.Contains(got, expect) {
					t.Errorf("missing expected string %q", expect)
				}
			}
			for _, prohibit := range tt.prohibitContent {
				if strings.Contains(got, prohibit) {
					t.Errorf("found prohibited string %q", prohibit)
				}
			}
		})
	}
}

func TestConfigSprint(t *testing.T) {
	c := &Config{
		Host:    "localhost:8080",
		APIPath: "v1",
		ContentConfig: ContentConfig{
			AcceptContentTypes: "application/json",
			ContentType:        "application/json",
		},
		Username:    "gopher",
		Password:    "g0ph3r",
		BearerToken: "1234567890",
		Impersonate: ImpersonationConfig{
			UserName: "gopher2",
			UID:      "uid123",
		},
		AuthProvider: &clientcmdapi.AuthProviderConfig{
			Name:   "gopher",
			Config: map[string]string{"secret": "s3cr3t"},
		},
		AuthConfigPersister: fakeAuthProviderConfigPersister{},
		ExecProvider: &clientcmdapi.ExecConfig{
			Command:            "sudo",
			Args:               []string{"secret"},
			Env:                []clientcmdapi.ExecEnvVar{{Name: "secret", Value: "s3cr3t"}},
			ProvideClusterInfo: true,
			Config:             &runtime.Unknown{Raw: []byte("super secret password")},
		},
		TLSClientConfig: TLSClientConfig{
			CertFile:                "a.crt",
			KeyFile:                 "a.key",
			CertData:                []byte("fake cert"),
			KeyData:                 []byte("fake key"),
			NextProtos:              []string{"h2", "http/1.1"},
			DisableTransportCaching: false,
		},
		UserAgent:      "gobot",
		Transport:      &fakeRoundTripper{},
		WrapTransport:  fakeWrapperFunc,
		QPS:            1,
		Burst:          2,
		RateLimiter:    &fakeLimiter{},
		WarningHandler: fakeWarningHandler{},
		Timeout:        3 * time.Second,
		Dial:           fakeDialFunc,
		Proxy:          fakeProxyFunc,
	}
	want := fmt.Sprintf(
		`&rest.Config{Host:"localhost:8080", APIPath:"v1", ContentConfig:rest.ContentConfig{AcceptContentTypes:"application/json", ContentType:"application/json", GroupVersion:(*schema.GroupVersion)(nil), NegotiatedSerializer:runtime.NegotiatedSerializer(nil)}, Username:"gopher", Password:"--- REDACTED ---", BearerToken:"--- REDACTED ---", BearerTokenFile:"", Impersonate:rest.ImpersonationConfig{UserName:"gopher2", UID:"uid123", Groups:[]string(nil), Extra:map[string][]string(nil)}, AuthProvider:api.AuthProviderConfig{Name: "gopher", Config: map[string]string{--- REDACTED ---}}, AuthConfigPersister:rest.AuthProviderConfigPersister(--- REDACTED ---), ExecProvider:api.ExecConfig{Command: "sudo", Args: []string{"--- REDACTED ---"}, Env: []ExecEnvVar{--- REDACTED ---}, APIVersion: "", ProvideClusterInfo: true, Config: runtime.Object(--- REDACTED ---), StdinUnavailable: false}, TLSClientConfig:rest.sanitizedTLSClientConfig{Insecure:false, ServerName:"", CertFile:"a.crt", KeyFile:"a.key", CAFile:"", CertData:[]uint8{0x2d, 0x2d, 0x2d, 0x20, 0x54, 0x52, 0x55, 0x4e, 0x43, 0x41, 0x54, 0x45, 0x44, 0x20, 0x2d, 0x2d, 0x2d}, KeyData:[]uint8{0x2d, 0x2d, 0x2d, 0x20, 0x52, 0x45, 0x44, 0x41, 0x43, 0x54, 0x45, 0x44, 0x20, 0x2d, 0x2d, 0x2d}, CAData:[]uint8(nil), NextProtos:[]string{"h2", "http/1.1"}, DisableTransportCaching:false}, UserAgent:"gobot", DisableCompression:false, Transport:(*rest.fakeRoundTripper)(%p), WrapTransport:(transport.WrapperFunc)(%p), QPS:1, Burst:2, RateLimiter:(*rest.fakeLimiter)(%p), WarningHandler:rest.fakeWarningHandler{}, Timeout:3000000000, Dial:(func(context.Context, string, string) (net.Conn, error))(%p), Proxy:(func(*http.Request) (*url.URL, error))(%p)}`,
		c.Transport, fakeWrapperFunc, c.RateLimiter, fakeDialFunc, fakeProxyFunc,
	)

	for _, f := range []string{"%s", "%v", "%+v", "%#v"} {
		if got := fmt.Sprintf(f, c); want != got {
			t.Errorf("fmt.Sprintf(%q, c)\ngot:  %q\nwant: %q", f, got, want)
		}
	}
}
