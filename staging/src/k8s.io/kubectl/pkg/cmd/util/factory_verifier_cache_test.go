/*
Copyright 2026 The Kubernetes Authors.

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
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	fakediscovery "k8s.io/client-go/discovery/fake"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	clienttesting "k8s.io/client-go/testing"
)

type hashedCountingOpenAPIClient struct {
	inner       openapiclient.Client
	schemaCalls atomic.Int64
}

func (c *hashedCountingOpenAPIClient) Paths() (map[string]openapiclient.GroupVersion, error) {
	paths, err := c.inner.Paths()
	if err != nil {
		return nil, err
	}

	wrapped := make(map[string]openapiclient.GroupVersion, len(paths))
	for path, groupVersion := range paths {
		parsedURL, err := url.Parse(groupVersion.ServerRelativeURL())
		if err != nil {
			return nil, err
		}

		query := parsedURL.Query()
		query.Set("hash", "factory-shared-cache-test")
		parsedURL.RawQuery = query.Encode()

		wrapped[path] = &hashedCountingGroupVersion{
			GroupVersion:      groupVersion,
			serverRelativeURL: parsedURL.String(),
			schemaCalls:       &c.schemaCalls,
		}
	}

	return wrapped, nil
}

type hashedCountingGroupVersion struct {
	openapiclient.GroupVersion
	serverRelativeURL string
	schemaCalls       *atomic.Int64
}

func (g *hashedCountingGroupVersion) ServerRelativeURL() string {
	return g.serverRelativeURL
}

func (g *hashedCountingGroupVersion) Schema(contentType string) ([]byte, error) {
	g.schemaCalls.Add(1)
	return g.GroupVersion.Schema(contentType)
}

func TestValidatorSharesVerifierCacheAcrossCalls(t *testing.T) {
	v3Client := &hashedCountingOpenAPIClient{
		inner: openapitest.NewEmbeddedFileClient(),
	}

	discoveryClient := &panicV2Discovery{
		FakeDiscovery: &fakediscovery.FakeDiscovery{
			Fake: &clienttesting.Fake{},
		},
		t:        t,
		v3Client: v3Client,
	}

	flags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(fakeClientConfig{}).
		WithDiscoveryClient(discoveryClient)

	factory := NewFactory(flags)

	pod := []byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"cache-test"}}`)

	for range 2 {
		validator, err := factory.Validator(metav1.FieldValidationStrict)
		require.NoError(t, err)
		require.NoError(t, validator.ValidateBytes(pod))
	}

	require.Equal(t, int64(1), v3Client.schemaCalls.Load(),
		"the second Validator call should reuse the first call's cached verification result")
}
