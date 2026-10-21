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
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

type fakeGroupVersion struct {
	data        []byte
	err         error
	url         string
	mu          sync.Mutex
	schemaCalls int
}

func (f *fakeGroupVersion) Schema(contentType string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.schemaCalls++
	if contentType != "application/json" {
		return nil, fmt.Errorf("unexpected content type %q", contentType)
	}
	return f.data, f.err
}

func (f *fakeGroupVersion) ServerRelativeURL() string { return f.url }

func (f *fakeGroupVersion) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.schemaCalls
}

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

// embeddedSchema returns the embedded JSON OpenAPI v3 test schema for the
// given discovery path key (e.g. "apis/apps/v1").
func embeddedSchema(t *testing.T, gvKey string) []byte {
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

func appsV1(t *testing.T, hash string) *fakeGroupVersion {
	return &fakeGroupVersion{data: embeddedSchema(t, "apis/apps/v1"), url: "/openapi/v3/apis/apps/v1?hash=" + hash}
}

func coreV1(t *testing.T, hash string) *fakeGroupVersion {
	return &fakeGroupVersion{data: embeddedSchema(t, "api/v1"), url: "/openapi/v3/api/v1?hash=" + hash}
}

func newTestExtractor(t *testing.T, client *fakeOpenAPIClient) *extractor {
	t.Helper()
	paths, err := client.Paths()
	if err != nil {
		t.Fatal(err)
	}
	return &extractor{cache: &gvkParserCache{
		client:      client,
		paths:       paths,
		lastChecked: time.Now(),
		parsers:     map[string]*managedfields.GvkParser{},
	}}
}

// expireTTL makes the next extraction refresh the discovery listing.
func expireTTL(e *extractor) { e.cache.lastChecked = time.Time{} }

// cachedURLs returns the schema URLs the extractor currently holds parsers for.
func cachedURLs(e *extractor) []string {
	urls := make([]string, 0, len(e.cache.parsers))
	for url := range e.cache.parsers {
		urls = append(urls, url)
	}
	sort.Strings(urls)
	return urls
}

func managedFieldsEntry(subresource string, fields map[string]interface{}) map[string]interface{} {
	entry := map[string]interface{}{
		"manager":    "test-manager",
		"operation":  "Apply",
		"apiVersion": "apps/v1",
		"fieldsType": "FieldsV1",
		"fieldsV1":   fields,
	}
	if subresource != "" {
		entry["subresource"] = subresource
	}
	return entry
}

func testDeployment() *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata": map[string]interface{}{
			"name":      "test-deployment",
			"namespace": "default",
			"managedFields": []interface{}{
				managedFieldsEntry("", map[string]interface{}{"f:spec": map[string]interface{}{"f:replicas": map[string]interface{}{}}}),
				managedFieldsEntry("status", map[string]interface{}{"f:status": map[string]interface{}{"f:replicas": map[string]interface{}{}}}),
			},
		},
		"spec":   map[string]interface{}{"replicas": int64(3), "paused": true},
		"status": map[string]interface{}{"replicas": int64(2), "readyReplicas": int64(1)},
	}}
}

func testConfigMap() *unstructured.Unstructured {
	entry := managedFieldsEntry("", map[string]interface{}{"f:data": map[string]interface{}{"f:key": map[string]interface{}{}}})
	entry["apiVersion"] = "v1"
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name":          "test-configmap",
			"namespace":     "default",
			"managedFields": []interface{}{entry},
		},
		"data": map[string]interface{}{"key": "value", "other": "ignored"},
	}}
}

func expectInt64(t *testing.T, obj *unstructured.Unstructured, want int64, fields ...string) {
	t.Helper()
	got, found, err := unstructured.NestedInt64(obj.Object, fields...)
	if err != nil || !found || got != want {
		t.Errorf("expected %s=%d, got %v (found=%v, err=%v)", strings.Join(fields, "."), want, got, found, err)
	}
}

func expectAbsent(t *testing.T, obj *unstructured.Unstructured, fields ...string) {
	t.Helper()
	if _, found, _ := unstructured.NestedFieldNoCopy(obj.Object, fields...); found {
		t.Errorf("expected %s to be absent from the extracted object, got %v", strings.Join(fields, "."), obj.Object)
	}
}

func TestUnstructuredExtract(t *testing.T) {
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsV1(t, "1")}}
	e := newTestExtractor(t, client)

	result, err := e.Extract(testDeployment(), "test-manager")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if result.GetName() != "test-deployment" || result.GetNamespace() != "default" ||
		result.GetKind() != "Deployment" || result.GetAPIVersion() != "apps/v1" {
		t.Errorf("unexpected identity fields in extracted object: %v", result.Object)
	}
	// only the fields the manager owns on the main resource
	expectInt64(t, result, 3, "spec", "replicas")
	expectAbsent(t, result, "spec", "paused")
	expectAbsent(t, result, "status")

	// a manager owning nothing extracts an empty configuration
	result, err = e.Extract(testDeployment(), "other-manager")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	expectAbsent(t, result, "spec")
}

func TestUnstructuredExtractStatus(t *testing.T) {
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": appsV1(t, "1")}}
	e := newTestExtractor(t, client)

	result, err := e.ExtractStatus(testDeployment(), "test-manager")
	if err != nil {
		t.Fatalf("ExtractStatus failed: %v", err)
	}
	// only the fields the manager owns through the status subresource
	expectInt64(t, result, 2, "status", "replicas")
	expectAbsent(t, result, "status", "readyReplicas")
	expectAbsent(t, result, "spec")
}

func TestUnstructuredExtractLazyPerGroupVersion(t *testing.T) {
	apps, core := appsV1(t, "1"), coreV1(t, "1")
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": apps, "api/v1": core}}
	e := newTestExtractor(t, client)

	for range 3 {
		if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
			t.Fatalf("Extract failed: %v", err)
		}
	}
	if apps.calls() != 1 {
		t.Errorf("expected apps/v1 schema downloaded exactly once, got %d", apps.calls())
	}
	if core.calls() != 0 {
		t.Errorf("expected core/v1 schema never downloaded, got %d downloads", core.calls())
	}

	result, err := e.Extract(testConfigMap(), "test-manager")
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if got, _, _ := unstructured.NestedString(result.Object, "data", "key"); got != "value" {
		t.Errorf("expected data.key=value, got %v", result.Object)
	}
	expectAbsent(t, result, "data", "other")
	if core.calls() != 1 {
		t.Errorf("expected core/v1 schema downloaded once on first use, got %d", core.calls())
	}
}

func TestUnstructuredExtractSchemaUpdated(t *testing.T) {
	apps := appsV1(t, "1")
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": apps}}
	e := newTestExtractor(t, client)

	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	// same hash after the TTL expires: the listing refreshes, the parser is reused
	expireTTL(e)
	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if client.pathsCalls != 2 { // once in newTestExtractor, once on expiry
		t.Errorf("expected 2 Paths() calls, got %d", client.pathsCalls)
	}
	if apps.calls() != 1 {
		t.Errorf("expected no schema re-download for an unchanged hash, got %d downloads", apps.calls())
	}

	// changed hash after the TTL expires: the schema is re-downloaded and the
	// old parser is gone
	updated := appsV1(t, "2")
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": updated}
	expireTTL(e)
	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if updated.calls() != 1 {
		t.Errorf("expected schema re-download for a changed hash, got %d downloads", updated.calls())
	}
	if got := cachedURLs(e); len(got) != 1 || got[0] != updated.url {
		t.Errorf("expected only the new schema to be cached, got %v", got)
	}

	// before the TTL expires a changed listing is not seen yet
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": appsV1(t, "3")}
	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if client.pathsCalls != 3 {
		t.Errorf("expected no Paths() call within the TTL, got %d calls", client.pathsCalls)
	}
}

func TestUnstructuredExtractPrunesRemovedGroupVersions(t *testing.T) {
	apps, core := appsV1(t, "1"), coreV1(t, "1")
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": apps, "api/v1": core}}
	e := newTestExtractor(t, client)

	for _, obj := range []*unstructured.Unstructured{testDeployment(), testConfigMap()} {
		if _, err := e.Extract(obj, "test-manager"); err != nil {
			t.Fatalf("Extract failed: %v", err)
		}
	}
	if got := cachedURLs(e); len(got) != 2 {
		t.Fatalf("expected two cached parsers, got %v", got)
	}

	// core/v1 disappears from the listing: its parser is dropped on the next
	// refresh even though only apps/v1 is extracted from, and extracting from
	// it fails
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": apps}
	expireTTL(e)
	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if got := cachedURLs(e); len(got) != 1 || got[0] != apps.url {
		t.Errorf("expected only apps/v1 to stay cached after core/v1 was removed, got %v", got)
	}
	if _, err := e.Extract(testConfigMap(), "test-manager"); err == nil || !strings.Contains(err.Error(), "no openapi v3 schema found") {
		t.Errorf("expected 'no openapi v3 schema found' for removed core/v1, got %v", err)
	}

	// core/v1 comes back: it is downloaded again on next use
	client.paths = map[string]openapi.GroupVersion{"apis/apps/v1": apps, "api/v1": core}
	expireTTL(e)
	if _, err := e.Extract(testConfigMap(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if core.calls() != 2 {
		t.Errorf("expected core/v1 schema downloaded again after it was pruned, got %d downloads", core.calls())
	}
}

func TestUnstructuredExtractRefreshFailureKeepsParsers(t *testing.T) {
	apps := appsV1(t, "1")
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": apps}}
	e := newTestExtractor(t, client)

	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	// a failed listing refresh surfaces the error and leaves the cache intact
	client.pathsErr = errors.New("server unavailable")
	expireTTL(e)
	if _, err := e.Extract(testDeployment(), "test-manager"); err == nil || !strings.Contains(err.Error(), "server unavailable") {
		t.Errorf("expected the listing error to be returned, got %v", err)
	}
	if got := cachedURLs(e); len(got) != 1 || got[0] != apps.url {
		t.Errorf("expected the cached parser to survive a failed refresh, got %v", got)
	}

	// once the listing is available again the retained parser is reused
	client.pathsErr = nil
	if _, err := e.Extract(testDeployment(), "test-manager"); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if apps.calls() != 1 {
		t.Errorf("expected the retained parser to be reused without re-download, got %d downloads", apps.calls())
	}
}

func TestUnstructuredExtractErrors(t *testing.T) {
	broken := &fakeGroupVersion{data: []byte("not json"), url: "/openapi/v3/apis/batch/v1?hash=1"}
	failing := &fakeGroupVersion{err: errors.New("download failed"), url: "/openapi/v3/apis/policy/v1?hash=1"}
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{
		"apis/apps/v1":   appsV1(t, "1"),
		"apis/batch/v1":  broken,
		"apis/policy/v1": failing,
	}}
	e := newTestExtractor(t, client)

	for _, tc := range []struct {
		name       string
		apiVersion string
		kind       string
		wantErr    string
	}{
		{name: "group-version not served", apiVersion: "example.com/v1", kind: "Widget", wantErr: "no openapi v3 schema found"},
		{name: "kind unknown to the group-version", apiVersion: "apps/v1", kind: "Bogus", wantErr: "no type found"},
		{name: "schema download fails", apiVersion: "policy/v1", kind: "PodDisruptionBudget", wantErr: "download failed"},
		{name: "schema is not valid JSON", apiVersion: "batch/v1", kind: "Job", wantErr: "failed to parse openapi v3 schema"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			obj := testDeployment()
			obj.SetAPIVersion(tc.apiVersion)
			obj.SetKind(tc.kind)
			if _, err := e.Extract(obj, "test-manager"); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
	// only the schema that built successfully is cached
	if got := cachedURLs(e); len(got) != 1 || !strings.Contains(got[0], "apis/apps/v1") {
		t.Errorf("expected only apps/v1 to be cached after the failed builds, got %v", got)
	}
}

func TestUnstructuredExtractConcurrent(t *testing.T) {
	apps, core := appsV1(t, "1"), coreV1(t, "1")
	client := &fakeOpenAPIClient{paths: map[string]openapi.GroupVersion{"apis/apps/v1": apps, "api/v1": core}}
	e := newTestExtractor(t, client)

	var wg sync.WaitGroup
	for i := range 20 {
		wg.Go(func() {
			obj := testDeployment()
			if i%2 == 1 {
				obj = testConfigMap()
			}
			if _, err := e.Extract(obj, "test-manager"); err != nil {
				t.Errorf("Extract failed: %v", err)
			}
		})
	}
	wg.Wait()
	if apps.calls() != 1 || core.calls() != 1 {
		t.Errorf("expected each schema downloaded once under concurrency, got apps=%d core=%d", apps.calls(), core.calls())
	}
}
