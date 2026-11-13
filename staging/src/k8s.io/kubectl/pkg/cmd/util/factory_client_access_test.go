/*
Copyright The Kubernetes Authors.

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

package util

import (
	"errors"
	"sync/atomic"
	"testing"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/rest"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type panicV2Discovery struct {
	*fakediscovery.FakeDiscovery
	t        *testing.T
	v3Client openapiclient.Client
}

func (d *panicV2Discovery) Fresh() bool {
	return true
}

func (d *panicV2Discovery) Invalidate() {}

func (d *panicV2Discovery) OpenAPISchema() (*openapi_v2.Document, error) {
	d.t.Fatal("unexpected fallback to OpenAPI v2")
	return nil, nil
}

func (d *panicV2Discovery) OpenAPIV3() openapiclient.Client {
	return d.v3Client
}

type fakeClientConfig struct{}

func (fakeClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return clientcmdapi.Config{}, nil
}

func (fakeClientConfig) ClientConfig() (*rest.Config, error) {
	return &rest.Config{Host: "https://localhost"}, nil
}

func (fakeClientConfig) Namespace() (string, bool, error) {
	return "default", false, nil
}

func (fakeClientConfig) ConfigAccess() clientcmd.ConfigAccess {
	return nil
}

func TestValidatorDoesNotFallbackToOpenAPIV2(t *testing.T) {
	v3Client := openapitest.NewFakeClient()
	v3Client.ForcedErr = errors.New("openapi v3 error")

	disc := &panicV2Discovery{
		FakeDiscovery: &fakediscovery.FakeDiscovery{Fake: &clienttesting.Fake{}},
		t:             t,
		v3Client:      v3Client,
	}

	flags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(fakeClientConfig{}).
		WithDiscoveryClient(disc)

	f := NewFactory(flags)
	validator, err := f.Validator(metav1.FieldValidationStrict)
	require.NoError(t, err)

	err = validator.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"foo"}}`))
	require.ErrorContains(t, err, "openapi v3 error")
}

// TestValidatorVerifiesListItems checks, against the embedded OpenAPI v3
// schemas, that a list of server-validated kinds is not validated
// client-side, and so never downloads /openapi/v2.
func TestValidatorVerifiesListItems(t *testing.T) {
	disc := &panicV2Discovery{
		FakeDiscovery: &fakediscovery.FakeDiscovery{Fake: &clienttesting.Fake{}},
		t:             t,
		v3Client:      openapitest.NewEmbeddedFileClient(),
	}

	flags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(fakeClientConfig{}).
		WithDiscoveryClient(disc)

	f := NewFactory(flags)
	validator, err := f.Validator(metav1.FieldValidationStrict)
	require.NoError(t, err)

	for _, doc := range []string{
		`{"apiVersion":"v1","kind":"List","items":[{"apiVersion":"v1","kind":"Pod","metadata":{"name":"foo"}}]}`,
		`{"apiVersion":"v1","kind":"ServiceList","items":[{"apiVersion":"v1","kind":"Service","metadata":{"name":"foo"}}]}`,
	} {
		require.NoError(t, validator.ValidateBytes([]byte(doc)), doc)
	}
}

// TestValidatorUnsupportedKindSkipsSchemaValidation checks that a kind the
// server cannot field-validate (Binding has no PATCH operation in the embedded
// OpenAPI v3 schemas) is not validated against the OpenAPI v2 schema; only the
// schema-free checks still run.
func TestValidatorUnsupportedKindSkipsSchemaValidation(t *testing.T) {
	disc := &panicV2Discovery{
		FakeDiscovery: &fakediscovery.FakeDiscovery{Fake: &clienttesting.Fake{}},
		t:             t,
		v3Client:      openapitest.NewEmbeddedFileClient(),
	}

	flags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(fakeClientConfig{}).
		WithDiscoveryClient(disc)

	f := NewFactory(flags)
	validator, err := f.Validator(metav1.FieldValidationStrict)
	require.NoError(t, err)

	err = validator.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Binding","metadata":{"name":"foo"},"unknown":true}`))
	require.NoError(t, err)

	err = validator.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Binding","metadata":{"name":"foo","labels":{"a":"1","a":"2"}}}`))
	require.ErrorContains(t, err, "duplicate key")
}

type countingV3GroupVersion struct {
	delegate    openapiclient.GroupVersion
	relativeURL string
	schemaCalls *atomic.Int32
}

func (c *countingV3GroupVersion) Schema(contentType string) ([]byte, error) {
	c.schemaCalls.Add(1)
	return c.delegate.Schema(contentType)
}

func (c *countingV3GroupVersion) ServerRelativeURL() string {
	return c.relativeURL
}

func TestFactoryMemoizesOpenAPIV3ClientAndValidator(t *testing.T) {
	embeddedPaths, err := openapitest.NewEmbeddedFileClient().Paths()
	require.NoError(t, err)

	var coreV1Calls atomic.Int32
	v3Client := openapitest.NewFakeClient()
	v3Client.PathsMap["api/v1"] = &countingV3GroupVersion{
		delegate:    embeddedPaths["api/v1"],
		relativeURL: "/openapi/v3/api/v1?hash=corev1",
		schemaCalls: &coreV1Calls,
	}

	disc := &panicV2Discovery{
		FakeDiscovery: &fakediscovery.FakeDiscovery{Fake: &clienttesting.Fake{}},
		t:             t,
		v3Client:      v3Client,
	}

	flags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(fakeClientConfig{}).
		WithDiscoveryClient(disc)

	f := NewFactory(flags)

	c1, err := f.OpenAPIV3Client()
	require.NoError(t, err)
	c2, err := f.OpenAPIV3Client()
	require.NoError(t, err)
	require.Same(t, c1, c2, "expected OpenAPIV3Client() to return the same cached client instance")

	// Repeated Validator() calls across strict and warn directives should share
	// the same cached verifier and fetch/unmarshal api/v1 only once across Kinds.
	vStrict1, err := f.Validator(metav1.FieldValidationStrict)
	require.NoError(t, err)
	require.NoError(t, vStrict1.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"p1"}}`)))
	require.NoError(t, vStrict1.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Service","metadata":{"name":"s1"}}`)))

	vStrict2, err := f.Validator(metav1.FieldValidationStrict)
	require.NoError(t, err)
	require.NoError(t, vStrict2.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cm1"}}`)))

	vWarn, err := f.Validator(metav1.FieldValidationWarn)
	require.NoError(t, err)
	require.NoError(t, vWarn.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Secret","metadata":{"name":"sec1"}}`)))

	vIgnore, err := f.Validator(metav1.FieldValidationIgnore)
	require.NoError(t, err)
	require.NoError(t, vIgnore.ValidateBytes([]byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"p2"}}`)))

	require.Equal(t, int32(1), coreV1Calls.Load(), "expected api/v1 schema to be fetched once across repeated Validator() calls and multiple Kinds")

	// Verify discovery and dynamic client errors propagate cleanly from OpenAPIV3Client() and Validator().
	discErrFactory := NewFactory(&discoveryErrGetter{RESTClientGetter: flags})
	_, err = discErrFactory.OpenAPIV3Client()
	require.ErrorContains(t, err, "discovery error")
	_, err = discErrFactory.Validator(metav1.FieldValidationStrict)
	require.ErrorContains(t, err, "discovery error")

	restErrFactory := NewFactory(&restConfigErrGetter{RESTClientGetter: flags})
	_, err = restErrFactory.Validator(metav1.FieldValidationStrict)
	require.ErrorContains(t, err, "rest config error")
}

type discoveryErrGetter struct {
	genericclioptions.RESTClientGetter
}

func (g *discoveryErrGetter) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return nil, errors.New("discovery error")
}

type restConfigErrGetter struct {
	genericclioptions.RESTClientGetter
}

func (g *restConfigErrGetter) ToRESTConfig() (*rest.Config, error) {
	return nil, errors.New("rest config error")
}
