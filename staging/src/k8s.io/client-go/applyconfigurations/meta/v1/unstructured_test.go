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

package v1

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

type fakeGroupVersion struct {
	data        []byte
	url         string
	schemaCalls int
}

func (f *fakeGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != "application/json" {
		return nil, fmt.Errorf("unexpected content type %q", contentType)
	}
	f.schemaCalls++
	return f.data, nil
}

func (f *fakeGroupVersion) ServerRelativeURL() string { return f.url }

type fakeOpenAPIClient struct {
	paths      map[string]openapi.GroupVersion
	pathsErr   error
	pathsCalls int
}

func (f *fakeOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	f.pathsCalls++
	if f.pathsErr != nil {
		return nil, f.pathsErr
	}
	return f.paths, nil
}

// jsonSchemaForGV returns the embedded JSON OpenAPI v3 test schema for the
// given discovery path key (e.g. "apis/apps/v1").
func jsonSchemaForGV(t *testing.T, gvKey string) []byte {
	t.Helper()
	paths, err := openapitest.NewEmbeddedFileClient().Paths()
	if err != nil {
		t.Fatal(err)
	}
	gv, ok := paths[gvKey]
	if !ok {
		t.Fatalf("no embedded test schema for %q", gvKey)
	}
	data, err := gv.Schema("application/json")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func newTestExtractor(client *fakeOpenAPIClient) *extractor {
	paths, _ := client.Paths()
	return &extractor{
		cache: &gvkParserCache{
			client:      client,
			paths:       paths,
			lastChecked: time.Now(),
			parsers:     map[schema.GroupVersion]gvkParserCacheEntry{},
		},
	}
}

func testDeployment(replicas int64) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata": map[string]interface{}{
			"name":      "test-deployment",
			"namespace": "default",
			"managedFields": []interface{}{
				map[string]interface{}{
					"manager":    "test-manager",
					"operation":  "Apply",
					"apiVersion": "apps/v1",
					"fieldsType": "FieldsV1",
					"fieldsV1": map[string]interface{}{
						"f:spec": map[string]interface{}{
							"f:replicas": map[string]interface{}{},
						},
					},
				},
			},
		},
		"spec": map[string]interface{}{
			"replicas": replicas,
		},
	}}
}

func testConfigMap() *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":      "test-configmap",
			"namespace": "default",
			"managedFields": []interface{}{
				map[string]interface{}{
					"manager":    "test-manager",
					"operation":  "Apply",
					"apiVersion": "v1",
					"fieldsType": "FieldsV1",
					"fieldsV1": map[string]interface{}{
						"f:data": map[string]interface{}{
							"f:key": map[string]interface{}{},
						},
					},
				},
			},
		},
		"data": map[string]interface{}{
			"key": "value",
		},
	}}
}

// cachedParserURLs returns the discovery URL each cached parser was built
// from, keyed by group-version, for asserting on the cache contents.
func cachedParserURLs(e *extractor) map[schema.GroupVersion]string {
	result := map[schema.GroupVersion]string{}
	for gv, entry := range e.cache.parsers {
		result[gv] = entry.serverRelativeURL
	}
	return result
}

func TestUnstructuredExtract(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsGV}}
	e := newTestExtractor(client)

	result, err := e.Extract(testDeployment(3), "test-manager")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if result.GetName() != "test-deployment" || result.GetNamespace() != "default" ||
		result.GetKind() != "Deployment" || result.GetAPIVersion() != "apps/v1" {
		t.Errorf("unexpected identity fields in extracted object: %v", result.Object)
	}
	replicas, found, err := unstructured.NestedInt64(result.Object, "spec", "replicas")
	if err != nil || !found || replicas != 3 {
		t.Errorf("expected extracted spec.replicas=3, got %v (found=%v, err=%v)", replicas, found, err)
	}

	// a manager owning nothing extracts an empty configuration
	result, err = e.Extract(testDeployment(3), "other-manager")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if _, found, _ := unstructured.NestedInt64(result.Object, "spec", "replicas"); found {
		t.Errorf("expected no spec.replicas for a non-owning manager, got %v", result.Object)
	}
}

func TestUnstructuredExtractLazyPerGroupVersion(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	coreGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "api/v1"), url: "/openapi/v3/api/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{
		"apis/apps/v1": appsGV,
		"api/v1":       coreGV,
	}}
	e := newTestExtractor(client)

	for range 2 {
		if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
			t.Fatalf("Extract failed: %v", err)
		}
	}
	if appsGV.schemaCalls != 1 {
		t.Errorf("expected apps/v1 schema downloaded exactly once, got %d", appsGV.schemaCalls)
	}
	if coreGV.schemaCalls != 0 {
		t.Errorf("expected core/v1 schema never downloaded, got %d downloads", coreGV.schemaCalls)
	}
}

func TestUnstructuredExtractSchemaUpdated(t *testing.T) {
	appsData := jsonSchemaForGV(t, "apis/apps/v1")
	appsGV := &fakeGroupVersion{data: appsData, url: "/openapi/v3/apis/apps/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsGV}}
	e := newTestExtractor(client)

	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	// same hash after the TTL expires: the parser is reused, only the listing refreshes
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if client.pathsCalls != 2 { // once in newTestExtractor, once on expiry
		t.Errorf("expected 2 Paths() calls, got %d", client.pathsCalls)
	}
	if appsGV.schemaCalls != 1 {
		t.Errorf("expected no schema re-download for an unchanged hash, got %d downloads", appsGV.schemaCalls)
	}

	// changed hash after the TTL expires: the schema is re-downloaded
	updatedGV := &fakeGroupVersion{data: appsData, url: "/openapi/v3/apis/apps/v1?hash=2"}
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": updatedGV}
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if updatedGV.schemaCalls != 1 {
		t.Errorf("expected schema re-download for a changed hash, got %d downloads", updatedGV.schemaCalls)
	}
	want := map[schema.GroupVersion]string{{Group: "apps", Version: "v1"}: "/openapi/v3/apis/apps/v1?hash=2"}
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Errorf("expected cached parsers %v, got %v", want, got)
	}
}

func TestUnstructuredExtractPrunesRemovedGroupVersions(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	coreGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "api/v1"), url: "/openapi/v3/api/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{
		"apis/apps/v1": appsGV,
		"api/v1":       coreGV,
	}}
	e := newTestExtractor(client)

	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if _, err := e.Extract(testConfigMap(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	want := map[schema.GroupVersion]string{
		{Group: "apps", Version: "v1"}: "/openapi/v3/apis/apps/v1?hash=1",
		{Group: "", Version: "v1"}:     "/openapi/v3/api/v1?hash=1",
	}
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Fatalf("expected cached parsers %v, got %v", want, got)
	}

	// core/v1 disappears from the listing: its parser is dropped on the next
	// refresh, even though only apps/v1 is extracted from
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": appsGV}
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	want = map[schema.GroupVersion]string{{Group: "apps", Version: "v1"}: "/openapi/v3/apis/apps/v1?hash=1"}
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Errorf("expected cached parsers %v after core/v1 was removed, got %v", want, got)
	}
	if appsGV.schemaCalls != 1 {
		t.Errorf("expected apps/v1 parser to be retained, got %d downloads", appsGV.schemaCalls)
	}
	if _, err := e.Extract(testConfigMap(), "test-manager"); err == nil || !strings.Contains(err.Error(), "no openapi v3 schema found") {
		t.Errorf("expected 'no openapi v3 schema found' error for removed core/v1, got %v", err)
	}

	// core/v1 comes back with an unchanged hash: it has to be downloaded again
	// because the pruned parser is gone
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": appsGV, "api/v1": coreGV}
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testConfigMap(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if coreGV.schemaCalls != 2 {
		t.Errorf("expected core/v1 schema downloaded again after it was pruned, got %d downloads", coreGV.schemaCalls)
	}
	if _, ok := e.cache.parsers[schema.GroupVersion{Version: "v1"}]; !ok {
		t.Errorf("expected core/v1 parser to be cached again, got %v", cachedParserURLs(e))
	}
}

func TestUnstructuredExtractPrunesChangedGroupVersions(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	coreData := jsonSchemaForGV(t, "api/v1")
	coreGV := &fakeGroupVersion{data: coreData, url: "/openapi/v3/api/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{
		"apis/apps/v1": appsGV,
		"api/v1":       coreGV,
	}}
	e := newTestExtractor(client)

	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if _, err := e.Extract(testConfigMap(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	// core/v1's hash changes but only apps/v1 is extracted from: the stale
	// core/v1 parser is dropped on refresh without eagerly re-downloading it
	updatedCoreGV := &fakeGroupVersion{data: coreData, url: "/openapi/v3/api/v1?hash=2"}
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": appsGV, "api/v1": updatedCoreGV}
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	want := map[schema.GroupVersion]string{{Group: "apps", Version: "v1"}: "/openapi/v3/apis/apps/v1?hash=1"}
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Errorf("expected cached parsers %v after core/v1 hash changed, got %v", want, got)
	}
	if updatedCoreGV.schemaCalls != 0 {
		t.Errorf("expected no eager re-download of core/v1, got %d downloads", updatedCoreGV.schemaCalls)
	}
	if appsGV.schemaCalls != 1 {
		t.Errorf("expected apps/v1 parser to be retained, got %d downloads", appsGV.schemaCalls)
	}

	// extracting from core/v1 again rebuilds its parser from the new schema
	if _, err := e.Extract(testConfigMap(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if updatedCoreGV.schemaCalls != 1 {
		t.Errorf("expected core/v1 schema downloaded once on next use, got %d downloads", updatedCoreGV.schemaCalls)
	}
	want[schema.GroupVersion{Version: "v1"}] = "/openapi/v3/api/v1?hash=2"
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Errorf("expected cached parsers %v, got %v", want, got)
	}
}

func TestUnstructuredExtractRefreshFailureKeepsParsers(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsGV}}
	e := newTestExtractor(client)

	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	// a failed listing refresh surfaces the error and leaves the cache intact
	client.pathsErr = fmt.Errorf("server unavailable")
	e.cache.lastChecked = time.Time{}
	if _, err := e.Extract(testDeployment(3), "test-manager"); err == nil || !strings.Contains(err.Error(), "server unavailable") {
		t.Errorf("expected listing error to be returned, got %v", err)
	}
	want := map[schema.GroupVersion]string{{Group: "apps", Version: "v1"}: "/openapi/v3/apis/apps/v1?hash=1"}
	if got := cachedParserURLs(e); !reflect.DeepEqual(got, want) {
		t.Errorf("expected cached parsers %v to survive a failed refresh, got %v", want, got)
	}

	// once the listing is available again the retained parser is reused
	client.pathsErr = nil
	if _, err := e.Extract(testDeployment(3), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if appsGV.schemaCalls != 1 {
		t.Errorf("expected retained parser to be reused without re-download, got %d downloads", appsGV.schemaCalls)
	}
}

func TestUnstructuredExtractErrors(t *testing.T) {
	appsGV := &fakeGroupVersion{data: jsonSchemaForGV(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsGV}}
	e := newTestExtractor(client)

	// group-version not served by the server
	obj := testDeployment(3)
	obj.SetAPIVersion("batch/v1")
	obj.SetKind("Job")
	if _, err := e.Extract(obj, "test-manager"); err == nil || !strings.Contains(err.Error(), "no openapi v3 schema found") {
		t.Errorf("expected 'no openapi v3 schema found' error, got %v", err)
	}

	// kind unknown to the group-version's schema
	obj = testDeployment(3)
	obj.SetKind("Bogus")
	if _, err := e.Extract(obj, "test-manager"); err == nil || !strings.Contains(err.Error(), "no type found") {
		t.Errorf("expected 'no type found' error, got %v", err)
	}
}
