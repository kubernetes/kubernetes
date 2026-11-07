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

package apply

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/openapi3"
	"k8s.io/kube-openapi/pkg/spec3"
)

type countingGroupVersion struct {
	delegate    openapiclient.GroupVersion
	relativeURL string
	schemaCalls *atomic.Int32
}

func (c *countingGroupVersion) Schema(contentType string) ([]byte, error) {
	c.schemaCalls.Add(1)
	return c.delegate.Schema(contentType)
}

func (c *countingGroupVersion) ServerRelativeURL() string {
	return c.relativeURL
}

func TestPatcherOpenAPIV3_CachesAcross20Resources(t *testing.T) {
	var coreV1Calls atomic.Int32
	fakeClient := openapitest.NewFakeClient()
	fakeClient.PathsMap["api/v1"] = &countingGroupVersion{
		delegate:    openapitest.FakeGroupVersion{GVSpec: fakeOpenAPIV3Legacy.SchemaBytesOrDie()},
		relativeURL: "/openapi/v3/api/v1?hash=corev1",
		schemaCalls: &coreV1Calls,
	}

	sharedRoot := openapi3.NewRoot(fakeClient)

	kinds := []string{"ServiceAccount", "Secret", "ConfigMap", "Service", "PersistentVolumeClaim", "Pod", "ReplicationController"}
	for i := range 20 {
		kind := kinds[i%len(kinds)]
		name := fmt.Sprintf("obj-%d", i)
		gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: kind}

		original := fmt.Sprintf(`{"apiVersion":"v1","kind":%q,"metadata":{"name":%q,"namespace":"default","labels":{"a":"1"}}}`, kind, name)
		modified := []byte(fmt.Sprintf(`{"apiVersion":"v1","kind":%q,"metadata":{"name":%q,"namespace":"default","labels":{"a":"2"}}}`, kind, name))

		obj := &unstructured.Unstructured{}
		require.NoError(t, json.Unmarshal([]byte(original), &obj.Object))
		obj.SetAnnotations(map[string]string{
			"kubectl.kubernetes.io/last-applied-configuration": original,
		})

		p := &Patcher{
			Mapping: &meta.RESTMapping{
				GroupVersionKind: gvk,
			},
			Overwrite:     true,
			OpenAPIV3Root: sharedRoot,
		}

		var errOut bytes.Buffer
		patchBytes, _, err := p.patchSimple(obj, modified, "default", name, &errOut, true)
		require.NoError(t, err, "unexpected patchSimple error for %s/%s", kind, name)
		assert.Contains(t, string(patchBytes), `"a":"2"`)
		assert.Empty(t, errOut.String())
	}

	assert.Equal(t, int32(1), coreV1Calls.Load(), "expected api/v1 OpenAPI V3 spec to be fetched and unmarshaled once across 20 resources (was 40 without Resources cache)")
}

type rootOnlyMock struct {
	spec *spec3.OpenAPI
	err  error
}

func (m *rootOnlyMock) GroupVersions() ([]schema.GroupVersion, error) {
	return nil, m.err
}

func (m *rootOnlyMock) GVSpec(gv schema.GroupVersion) (*spec3.OpenAPI, error) {
	return m.spec, m.err
}

func (m *rootOnlyMock) GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error) {
	return nil, m.err
}

func TestPatcherOpenAPIV3_EdgeCasesAndRootOnlyFallback(t *testing.T) {
	cmGVK := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}
	original := []byte(`{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cm","labels":{"a":"1"}}}`)
	modified := []byte(`{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cm","labels":{"a":"2"}}}`)
	current := []byte(`{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cm","labels":{"a":"1"}}}`)

	t.Run("Resources error paths and unknown Kind return expected values", func(t *testing.T) {
		errClient := openapitest.NewFakeClient()
		errClient.ForcedErr = fmt.Errorf("v3 client failure")

		pErr := &Patcher{
			Mapping:       &meta.RESTMapping{GroupVersionKind: cmGVK},
			Overwrite:     true,
			OpenAPIV3Root: openapi3.NewRoot(errClient),
		}
		_, err := pErr.gvkSupportsPatchOpenAPIV3(cmGVK)
		require.ErrorContains(t, err, "v3 client failure")
		_, err = pErr.buildStrategicMergePatchFromOpenAPIV3(original, modified, current)
		require.ErrorContains(t, err, "v3 client failure")

		validClient := openapitest.NewFakeClient()
		validClient.PathsMap["api/v1"] = openapitest.FakeGroupVersion{GVSpec: fakeOpenAPIV3Legacy.SchemaBytesOrDie()}
		pUnknownKind := &Patcher{
			Mapping:       &meta.RESTMapping{GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "DoesNotExist"}},
			Overwrite:     true,
			OpenAPIV3Root: openapi3.NewRoot(validClient),
		}
		supported, err := pUnknownKind.gvkSupportsPatchOpenAPIV3(pUnknownKind.Mapping.GroupVersionKind)
		require.NoError(t, err)
		assert.False(t, supported)

		patchBytes, err := pUnknownKind.buildStrategicMergePatchFromOpenAPIV3(original, modified, current)
		require.NoError(t, err)
		assert.Nil(t, patchBytes)
	})

	t.Run("custom Root mock without Resources falls back to GVSpec", func(t *testing.T) {
		var parsedSpec spec3.OpenAPI
		require.NoError(t, json.Unmarshal(fakeOpenAPIV3Legacy.SchemaBytesOrDie(), &parsedSpec))

		pFallback := &Patcher{
			Mapping:       &meta.RESTMapping{GroupVersionKind: cmGVK},
			Overwrite:     true,
			OpenAPIV3Root: &rootOnlyMock{spec: &parsedSpec},
		}
		supported, err := pFallback.gvkSupportsPatchOpenAPIV3(cmGVK)
		require.NoError(t, err)
		assert.True(t, supported)

		patchBytes, err := pFallback.buildStrategicMergePatchFromOpenAPIV3(original, modified, current)
		require.NoError(t, err)
		assert.Contains(t, string(patchBytes), `"a":"2"`)
	})
}
