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

package csimock

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	testNamespace  = "e2e-csi-mock-volumes-snapshot-1234"
	otherNamespace = "some-other-namespace"
)

// fakeSnapshotReader serves canned objects and records how it was called, so
// the tests can assert the request scope and bounds as well as the output.
type fakeSnapshotReader struct {
	lists    map[string]*unstructured.UnstructuredList
	listErrs map[string]error
	gets     map[string]*unstructured.Unstructured
	getErrs  map[string]error

	// respectContext makes the reader behave like a real client against a
	// context that is already over.
	respectContext bool

	listOptions map[string]metav1.ListOptions
	listCalls   []string
	getNames    []string
}

func (f *fakeSnapshotReader) List(ctx context.Context, gvr schema.GroupVersionResource, namespace string, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	if f.listOptions == nil {
		f.listOptions = map[string]metav1.ListOptions{}
	}
	f.listOptions[gvr.Resource] = opts
	f.listCalls = append(f.listCalls, gvr.Resource)
	if f.respectContext {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := f.listErrs[gvr.Resource]; err != nil {
		return nil, err
	}
	if list := f.lists[gvr.Resource]; list != nil {
		return list, nil
	}
	return &unstructured.UnstructuredList{}, nil
}

func (f *fakeSnapshotReader) Get(ctx context.Context, gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error) {
	f.getNames = append(f.getNames, name)
	if f.respectContext {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := f.getErrs[name]; err != nil {
		return nil, err
	}
	if obj := f.gets[name]; obj != nil {
		return obj, nil
	}
	return nil, apierrors.NewNotFound(gvr.GroupResource(), name)
}

func newSnapshot(name string, object map[string]interface{}) *unstructured.Unstructured {
	base := map[string]interface{}{
		"apiVersion": utils.SnapshotAPIVersion,
		"kind":       "VolumeSnapshot",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": testNamespace,
			"uid":       "snapshot-uid-" + name,
		},
	}
	for k, v := range object {
		base[k] = v
	}
	return &unstructured.Unstructured{Object: base}
}

// newContent builds a VolumeSnapshotContent bound to a VolumeSnapshot of the
// given namespace.
func newContent(name, snapshotNamespace string, object map[string]interface{}) *unstructured.Unstructured {
	base := map[string]interface{}{
		"apiVersion": utils.SnapshotAPIVersion,
		"kind":       "VolumeSnapshotContent",
		"metadata": map[string]interface{}{
			"name": name,
			"uid":  "content-uid-" + name,
		},
		"spec": map[string]interface{}{
			"volumeSnapshotRef": map[string]interface{}{
				"name":      "snapshot-for-" + name,
				"namespace": snapshotNamespace,
			},
		},
	}
	for k, v := range object {
		base[k] = v
	}
	return &unstructured.Unstructured{Object: base}
}

func listOf(items ...*unstructured.Unstructured) *unstructured.UnstructuredList {
	list := &unstructured.UnstructuredList{}
	for _, item := range items {
		list.Items = append(list.Items, *item)
	}
	return list
}

func requireContains(t *testing.T, report string, substrings ...string) {
	t.Helper()
	for _, s := range substrings {
		if !strings.Contains(report, s) {
			t.Errorf("report does not contain %q:\n%s", s, report)
		}
	}
}

func requireNotContains(t *testing.T, report string, substrings ...string) {
	t.Helper()
	for _, s := range substrings {
		if strings.Contains(report, s) {
			t.Errorf("report unexpectedly contains %q:\n%s", s, report)
		}
	}
}

func TestCollectSnapshotDiagnosticsReportsStuckDeletion(t *testing.T) {
	snapshot := newSnapshot("snapshot-stuck", map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":              "snapshot-stuck",
			"namespace":         testNamespace,
			"uid":               "snapshot-uid",
			"deletionTimestamp": "2026-01-02T03:04:05Z",
			"finalizers":        []interface{}{"snapshot.storage.kubernetes.io/volumesnapshot-bound-protection"},
		},
		"spec": map[string]interface{}{
			"volumeSnapshotClassName": "csi-mock-snapshot-class",
			"source": map[string]interface{}{
				"persistentVolumeClaimName": "snapshot-test-pvc",
			},
		},
		"status": map[string]interface{}{
			"boundVolumeSnapshotContentName": "snapcontent-1",
			"readyToUse":                     true,
			"creationTime":                   "2026-01-02T03:00:00Z",
		},
	})
	content := newContent("snapcontent-1", testNamespace, map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":              "snapcontent-1",
			"uid":               "content-uid",
			"deletionTimestamp": "2026-01-02T03:04:06Z",
			"finalizers":        []interface{}{"snapshot.storage.kubernetes.io/volumesnapshotcontent-bound-protection"},
		},
		"spec": map[string]interface{}{
			"deletionPolicy": "Delete",
			"driver":         "csi-mock-driver",
			"volumeSnapshotRef": map[string]interface{}{
				"name":      "snapshot-stuck",
				"namespace": testNamespace,
				"uid":       "snapshot-uid",
			},
		},
		"status": map[string]interface{}{
			"readyToUse":   false,
			"creationTime": int64(1767322800000000000),
			"error": map[string]interface{}{
				"message": "failed to delete snapshot",
				"time":    "2026-01-02T03:05:00Z",
			},
		},
	})

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: listOf(snapshot)},
		gets:  map[string]*unstructured.Unstructured{"snapcontent-1": content},
	}

	report := collectSnapshotDiagnostics(context.Background(), reader, testNamespace).report()

	requireContains(t, report,
		"VolumeSnapshot "+testNamespace+"/snapshot-stuck",
		"deletionTimestamp=2026-01-02T03:04:05Z",
		"finalizers=[snapshot.storage.kubernetes.io/volumesnapshot-bound-protection]",
		"boundContent=snapcontent-1",
		"sourcePVC=snapshot-test-pvc",
		"VolumeSnapshotContent snapcontent-1",
		"finalizers=[snapshot.storage.kubernetes.io/volumesnapshotcontent-bound-protection]",
		"deletionPolicy=Delete",
		"driver=csi-mock-driver",
		"readyToUse=false",
		"creationTime=1767322800000000000",
		"error=failed to delete snapshot",
	)

	if got := reader.listOptions[utils.SnapshotGVR.Resource].Limit; got != maxSnapshotsReported {
		t.Errorf("VolumeSnapshot list limit = %d, want %d", got, maxSnapshotsReported)
	}
	if got := reader.listOptions[utils.SnapshotContentGVR.Resource].Limit; got != maxContentScanPageSize {
		t.Errorf("VolumeSnapshotContent scan limit = %d, want %d", got, maxContentScanPageSize)
	}
	if len(reader.getNames) != 1 || reader.getNames[0] != "snapcontent-1" {
		t.Errorf("content Get calls = %v, want exactly [snapcontent-1]", reader.getNames)
	}
}

func TestCollectSnapshotDiagnosticsScopesToTestNamespace(t *testing.T) {
	// The orphan has no VolumeSnapshot left to reference it, which is exactly
	// the leak the bounded scan has to find.
	orphan := newContent("snapcontent-orphan", testNamespace, nil)
	foreign := newContent("snapcontent-other-namespace", otherNamespace, nil)
	referenced := newContent("snapcontent-1", testNamespace, nil)
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"status": map[string]interface{}{"boundVolumeSnapshotContentName": "snapcontent-1"},
	})

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{
			utils.SnapshotGVR.Resource:        listOf(snapshot),
			utils.SnapshotContentGVR.Resource: listOf(referenced, foreign, orphan),
		},
		gets: map[string]*unstructured.Unstructured{"snapcontent-1": referenced},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)
	report := diagnostics.report()

	requireContains(t, report, "snapcontent-1", "snapcontent-orphan")
	requireNotContains(t, report, "snapcontent-other-namespace", otherNamespace)
	if len(diagnostics.contents) != 2 {
		t.Errorf("collected %d contents, want 2 (the referenced one and the orphan)", len(diagnostics.contents))
	}
}

func TestCollectSnapshotDiagnosticsSkipsContentBoundToAnotherNamespace(t *testing.T) {
	// VolumeSnapshotContents are cluster scoped, so a spec can name one that
	// belongs to a different namespace. Reporting it would leak another test's
	// object into this report.
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"spec": map[string]interface{}{
			"source": map[string]interface{}{"volumeSnapshotContentName": "snapcontent-elsewhere"},
		},
	})
	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: listOf(snapshot)},
		gets: map[string]*unstructured.Unstructured{
			"snapcontent-elsewhere": newContent("snapcontent-elsewhere", otherNamespace, nil),
		},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)
	report := diagnostics.report()

	if len(diagnostics.contents) != 0 {
		t.Errorf("collected %d contents, want none: the content is bound to another namespace", len(diagnostics.contents))
	}
	requireContains(t, report, "skipped VolumeSnapshotContent snapcontent-elsewhere")
	requireNotContains(t, report, otherNamespace+"/", "content-uid-snapcontent-elsewhere")
}

func TestCollectSnapshotDiagnosticsReadsPreProvisionedSource(t *testing.T) {
	// A snapshot that names the same content in spec.source and in status must
	// be read once, and a snapshot that names no content at all must not
	// trigger a Get for the placeholder value.
	bound := newSnapshot("snapshot-bound", map[string]interface{}{
		"spec": map[string]interface{}{
			"source": map[string]interface{}{"volumeSnapshotContentName": "snapcontent-1"},
		},
		"status": map[string]interface{}{"boundVolumeSnapshotContentName": "snapcontent-1"},
	})
	unbound := newSnapshot("snapshot-unbound", nil)

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: listOf(bound, unbound)},
		gets:  map[string]*unstructured.Unstructured{"snapcontent-1": newContent("snapcontent-1", testNamespace, nil)},
	}

	report := collectSnapshotDiagnostics(context.Background(), reader, testNamespace).report()

	if len(reader.getNames) != 1 || reader.getNames[0] != "snapcontent-1" {
		t.Errorf("content Get calls = %v, want exactly [snapcontent-1]", reader.getNames)
	}
	requireContains(t, report, "sourceContent=snapcontent-1", "VolumeSnapshotContent snapcontent-1")
}

func TestCollectSnapshotDiagnosticsFallsBackToScanAfterAFailedGet(t *testing.T) {
	// A content that could not be read by name is not "already reported", so
	// the scan has to pick it up: it carries the finalizer evidence.
	content := newContent("snapcontent-1", testNamespace, map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":       "snapcontent-1",
			"finalizers": []interface{}{"snapshot.storage.kubernetes.io/volumesnapshotcontent-bound-protection"},
		},
	})
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"status": map[string]interface{}{"boundVolumeSnapshotContentName": "snapcontent-1"},
	})

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{
			utils.SnapshotGVR.Resource:        listOf(snapshot),
			utils.SnapshotContentGVR.Resource: listOf(content),
		},
		getErrs: map[string]error{"snapcontent-1": apierrors.NewInternalError(fmt.Errorf("apiserver is restarting"))},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)
	report := diagnostics.report()

	if len(diagnostics.contents) != 1 {
		t.Fatalf("collected %d contents, want the one the scan recovered", len(diagnostics.contents))
	}
	requireContains(t, report,
		"could not get VolumeSnapshotContent snapcontent-1",
		"VolumeSnapshotContent snapcontent-1",
		"finalizers=[snapshot.storage.kubernetes.io/volumesnapshotcontent-bound-protection]",
	)
}

func TestCollectSnapshotDiagnosticsOmitsFieldsOutsideTheAllowlist(t *testing.T) {
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":        "snapshot-a",
			"namespace":   testNamespace,
			"annotations": map[string]interface{}{"private.example.com/note": "annotation-value"},
			"labels":      map[string]interface{}{"private.example.com/label": "label-value"},
		},
		"status": map[string]interface{}{"boundVolumeSnapshotContentName": "snapcontent-1"},
	})
	content := newContent("snapcontent-1", testNamespace, map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":        "snapcontent-1",
			"annotations": map[string]interface{}{"snapshot.storage.kubernetes.io/deletion-secret-name": "secret-name-value"},
		},
		"spec": map[string]interface{}{
			"driver": "csi-mock-driver",
			"source": map[string]interface{}{
				"volumeHandle":   "volume-handle-value",
				"snapshotHandle": "snapshot-handle-value",
			},
			"volumeSnapshotRef": map[string]interface{}{"name": "snapshot-a", "namespace": testNamespace},
		},
		"status": map[string]interface{}{"snapshotHandle": "status-snapshot-handle-value"},
	})

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: listOf(snapshot)},
		gets:  map[string]*unstructured.Unstructured{"snapcontent-1": content},
	}

	report := collectSnapshotDiagnostics(context.Background(), reader, testNamespace).report()

	requireContains(t, report, "snapcontent-1", "driver=csi-mock-driver")
	requireNotContains(t, report,
		"annotation-value",
		"label-value",
		"secret-name-value",
		"volume-handle-value",
		"snapshot-handle-value",
		"status-snapshot-handle-value",
	)
}

func TestCollectSnapshotDiagnosticsReportsPartialFailures(t *testing.T) {
	testCases := []struct {
		name     string
		reader   *fakeSnapshotReader
		expected []string
		rejected []string
	}{
		{
			name: "snapshot CRDs not installed",
			reader: &fakeSnapshotReader{
				listErrs: map[string]error{
					utils.SnapshotGVR.Resource:        apierrors.NewNotFound(utils.SnapshotGVR.GroupResource(), ""),
					utils.SnapshotContentGVR.Resource: apierrors.NewNotFound(utils.SnapshotContentGVR.GroupResource(), ""),
				},
			},
			expected: []string{
				"could not list VolumeSnapshots",
				"the snapshot CRDs are not installed",
				"VolumeSnapshots could not be read",
				"VolumeSnapshotContents could not be read",
			},
			// An unreadable namespace must never be rendered as an empty one.
			rejected: []string{"no VolumeSnapshots found", "no VolumeSnapshotContents found"},
		},
		{
			name: "content read denied",
			reader: &fakeSnapshotReader{
				lists: map[string]*unstructured.UnstructuredList{
					utils.SnapshotGVR.Resource: listOf(newSnapshot("snapshot-a", map[string]interface{}{
						"status": map[string]interface{}{"boundVolumeSnapshotContentName": "snapcontent-1"},
					})),
				},
				getErrs:  map[string]error{"snapcontent-1": apierrors.NewForbidden(utils.SnapshotContentGVR.GroupResource(), "snapcontent-1", fmt.Errorf("denied"))},
				listErrs: map[string]error{utils.SnapshotContentGVR.Resource: apierrors.NewUnauthorized("no credentials")},
			},
			expected: []string{
				"VolumeSnapshot " + testNamespace + "/snapshot-a",
				"could not get VolumeSnapshotContent snapcontent-1: not permitted",
				"could not scan VolumeSnapshotContents",
				"not permitted",
			},
		},
		{
			name: "request rejected with a deadline error",
			reader: &fakeSnapshotReader{
				listErrs: map[string]error{
					utils.SnapshotGVR.Resource:        context.DeadlineExceeded,
					utils.SnapshotContentGVR.Resource: context.DeadlineExceeded,
				},
			},
			expected: []string{
				"could not list VolumeSnapshots",
				context.DeadlineExceeded.Error(),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			report := collectSnapshotDiagnostics(context.Background(), tc.reader, testNamespace).report()
			requireContains(t, report, tc.expected...)
			requireNotContains(t, report, tc.rejected...)
		})
	}
}

func TestCollectSnapshotDiagnosticsWithAnExpiredContext(t *testing.T) {
	// The production entry point builds its own context, but the collector
	// still has to survive one that is already over.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	reader := &fakeSnapshotReader{respectContext: true}
	diagnostics := collectSnapshotDiagnostics(ctx, reader, testNamespace)
	report := diagnostics.report()

	if len(diagnostics.snapshots) != 0 || len(diagnostics.contents) != 0 {
		t.Errorf("collected objects from an expired context: %+v", diagnostics)
	}
	requireContains(t, report, "could not list VolumeSnapshots", context.Canceled.Error())
}

func TestCollectSnapshotDiagnosticsRequiresANamespace(t *testing.T) {
	// An empty namespace means "all namespaces" to the dynamic client, which
	// would turn a scoped diagnostic into a cluster-wide dump.
	reader := &fakeSnapshotReader{}
	report := collectSnapshotDiagnostics(context.Background(), reader, "").report()

	if len(reader.listCalls) != 0 {
		t.Errorf("reader was called %v for an empty namespace, want no requests", reader.listCalls)
	}
	requireContains(t, report, "no namespace to inspect")
}

func TestCollectSnapshotDiagnosticsReportsTruncation(t *testing.T) {
	// The fake returns more items than the requested limit on purpose: the
	// in-loop caps are the defence against a server that ignores Limit.
	snapshots := listOf()
	for i := 0; i < maxSnapshotsReported+5; i++ {
		snapshots.Items = append(snapshots.Items, *newSnapshot(fmt.Sprintf("snapshot-%d", i), nil))
	}
	snapshots.SetContinue("next-page-token")

	contents := listOf()
	for i := 0; i < maxContentsReported+5; i++ {
		contents.Items = append(contents.Items, *newContent(fmt.Sprintf("snapcontent-%d", i), testNamespace, nil))
	}
	contents.SetContinue("next-page-token")

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{
			utils.SnapshotGVR.Resource:        snapshots,
			utils.SnapshotContentGVR.Resource: contents,
		},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)

	if len(diagnostics.snapshots) != maxSnapshotsReported {
		t.Errorf("collected %d snapshots, want the cap of %d", len(diagnostics.snapshots), maxSnapshotsReported)
	}
	if len(diagnostics.contents) != maxContentsReported {
		t.Errorf("collected %d contents, want the cap of %d", len(diagnostics.contents), maxContentsReported)
	}
	requireContains(t, diagnostics.report(),
		fmt.Sprintf("stopped after %d VolumeSnapshots", maxSnapshotsReported),
		fmt.Sprintf("holds more VolumeSnapshots than the %d that were requested", maxSnapshotsReported),
		"the scanned page held more for namespace "+testNamespace,
		fmt.Sprintf("scan read only the first %d items", maxContentScanPageSize),
	)
}

// referencingSnapshots builds maxSnapshotsReported snapshots that each name
// two distinct contents, so the referenced list is twice the content cap.
func referencingSnapshots() (*unstructured.UnstructuredList, []string) {
	snapshots := listOf()
	var names []string
	for i := 0; i < maxSnapshotsReported; i++ {
		bound := fmt.Sprintf("snapcontent-bound-%02d", i)
		source := fmt.Sprintf("snapcontent-source-%02d", i)
		snapshots.Items = append(snapshots.Items, *newSnapshot(fmt.Sprintf("snapshot-%02d", i), map[string]interface{}{
			"spec":   map[string]interface{}{"source": map[string]interface{}{"volumeSnapshotContentName": source}},
			"status": map[string]interface{}{"boundVolumeSnapshotContentName": bound},
		}))
		names = append(names, bound, source)
	}
	return snapshots, names
}

func TestCollectSnapshotDiagnosticsBoundsTheContentReads(t *testing.T) {
	// The number of requests has to be bounded on its own: a read that fails,
	// or that names an object of another namespace, reports nothing, so the
	// reported objects cannot be the only brake.
	snapshots, names := referencingSnapshots()
	referencedTotal := len(names)

	okGets := map[string]*unstructured.Unstructured{}
	foreignGets := map[string]*unstructured.Unstructured{}
	failedGets := map[string]error{}
	for _, name := range names {
		okGets[name] = newContent(name, testNamespace, nil)
		foreignGets[name] = newContent(name, otherNamespace, nil)
		failedGets[name] = apierrors.NewInternalError(fmt.Errorf("apiserver is restarting"))
	}

	testCases := []struct {
		name         string
		reader       *fakeSnapshotReader
		wantContents int
		wantNote     string
	}{
		{
			name: "every read fails",
			reader: &fakeSnapshotReader{
				lists:   map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: snapshots},
				getErrs: failedGets,
			},
			wantNote: fmt.Sprintf("stopped after %d VolumeSnapshotContent reads, %d further referenced contents were not read",
				maxContentGets, referencedTotal-maxContentGets),
		},
		{
			name: "every read names another namespace",
			reader: &fakeSnapshotReader{
				lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: snapshots},
				gets:  foreignGets,
			},
			wantNote: fmt.Sprintf("stopped after %d VolumeSnapshotContent reads, %d further referenced contents were not read",
				maxContentGets, referencedTotal-maxContentGets),
		},
		{
			name: "every read succeeds",
			reader: &fakeSnapshotReader{
				lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: snapshots},
				gets:  okGets,
			},
			wantContents: maxContentsReported,
			wantNote: fmt.Sprintf("stopped after %d VolumeSnapshotContents, %d further referenced contents were not read",
				maxContentsReported, referencedTotal-maxContentsReported),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			diagnostics := collectSnapshotDiagnostics(context.Background(), tc.reader, testNamespace)

			if got := len(tc.reader.getNames); got > maxContentGets {
				t.Errorf("made %d content reads, want at most %d", got, maxContentGets)
			}
			if got := len(diagnostics.contents); got != tc.wantContents {
				t.Errorf("reported %d contents, want %d", got, tc.wantContents)
			}
			requireContains(t, diagnostics.report(), tc.wantNote)
		})
	}
}

func TestCollectSnapshotDiagnosticsBoundsTheScannedPage(t *testing.T) {
	// A server that ignores Limit must not make the scan unbounded. The items
	// belong to another namespace so that the reported-content cap cannot be
	// what stops the loop.
	contents := listOf()
	for i := 0; i < maxContentScanPageSize*2; i++ {
		contents.Items = append(contents.Items, *newContent(fmt.Sprintf("snapcontent-%03d", i), otherNamespace, nil))
	}

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotContentGVR.Resource: contents},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)

	if len(diagnostics.contents) != 0 {
		t.Errorf("reported %d contents of another namespace, want none", len(diagnostics.contents))
	}
	requireContains(t, diagnostics.report(),
		fmt.Sprintf("examined only the first %d items of the VolumeSnapshotContent page, the server returned %d",
			maxContentScanPageSize, len(contents.Items)),
	)
}

func TestCollectSnapshotDiagnosticsWithoutObjects(t *testing.T) {
	report := collectSnapshotDiagnostics(context.Background(), &fakeSnapshotReader{}, testNamespace).report()
	requireContains(t, report,
		fmt.Sprintf("VolumeSnapshot diagnostics for namespace %q", testNamespace),
		"no VolumeSnapshots found",
		"no VolumeSnapshotContents found",
	)
}

func TestSummarizeHandlesUnsetAndNonScalarFields(t *testing.T) {
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"spec":   map[string]interface{}{"source": map[string]interface{}{"persistentVolumeClaimName": ""}},
		"status": map[string]interface{}{"error": map[string]interface{}{"message": map[string]interface{}{"unexpected": "object"}}},
	})

	summary := summarizeSnapshot(snapshot)
	if summary.sourcePVC != unsetFieldValue {
		t.Errorf("sourcePVC = %q, want %q for an empty string", summary.sourcePVC, unsetFieldValue)
	}
	if summary.deletionTimestamp != unsetFieldValue {
		t.Errorf("deletionTimestamp = %q, want %q when the object is not being deleted", summary.deletionTimestamp, unsetFieldValue)
	}
	if summary.boundContent != unsetFieldValue {
		t.Errorf("boundContent = %q, want %q when status is missing the field", summary.boundContent, unsetFieldValue)
	}
	if !strings.HasPrefix(summary.errorMessage, "<non-scalar") {
		t.Errorf("errorMessage = %q, want a non-scalar marker rather than a dumped object", summary.errorMessage)
	}
	if strings.Contains(summary.String(), "unexpected") {
		t.Errorf("summary leaked the contents of a non-scalar field: %s", summary.String())
	}
}

func TestTruncateMessage(t *testing.T) {
	short := strings.Repeat("a", maxMessageLength)
	if got := truncateMessage(short); got != short {
		t.Errorf("a message of exactly the maximum length was modified")
	}

	long := strings.Repeat("b", maxMessageLength+100)
	got := truncateMessage(long)
	if len(got) <= maxMessageLength || !strings.Contains(got, "truncated") {
		t.Errorf("long message was not truncated with a marker: %q", got)
	}
	if !strings.HasPrefix(got, strings.Repeat("b", maxMessageLength)) {
		t.Errorf("truncated message does not start with the first %d bytes", maxMessageLength)
	}
}

func TestRenderingIsBoundedButLookupsAreNot(t *testing.T) {
	longName := strings.Repeat("n", maxFieldLength*3)
	snapshot := newSnapshot("snapshot-a", map[string]interface{}{
		"metadata": map[string]interface{}{
			"name":       "snapshot-a",
			"namespace":  testNamespace,
			"finalizers": manyFinalizers(maxFinalizersRendered + 4),
		},
		"spec":   map[string]interface{}{"volumeSnapshotClassName": strings.Repeat("c", maxFieldLength*2)},
		"status": map[string]interface{}{"boundVolumeSnapshotContentName": longName},
	})

	reader := &fakeSnapshotReader{
		lists: map[string]*unstructured.UnstructuredList{utils.SnapshotGVR.Resource: listOf(snapshot)},
		gets:  map[string]*unstructured.Unstructured{longName: newContent(longName, testNamespace, nil)},
	}

	diagnostics := collectSnapshotDiagnostics(context.Background(), reader, testNamespace)
	report := diagnostics.report()

	// The stored value is the identity used to find the object, so it must
	// reach the API server whole even though it is rendered short.
	if len(reader.getNames) != 1 || reader.getNames[0] != longName {
		t.Errorf("content was looked up with a shortened name: %v", reader.getNames)
	}
	if len(diagnostics.contents) != 1 {
		t.Errorf("reported %d contents, want the one that was found by its full name", len(diagnostics.contents))
	}
	requireNotContains(t, report, longName)
	requireContains(t, report,
		fmt.Sprintf("truncated, %d bytes total", len(longName)),
		fmt.Sprintf("(+%d more)", 4),
	)
	for _, line := range strings.Split(report, "\n") {
		if strings.HasPrefix(line, "  VolumeSnapshot") && len(line) > maxReportLength {
			t.Errorf("rendered object line is %d bytes", len(line))
		}
	}
}

func manyFinalizers(count int) []interface{} {
	finalizers := make([]interface{}, 0, count)
	for i := 0; i < count; i++ {
		finalizers = append(finalizers, fmt.Sprintf("example.com/finalizer-%02d", i))
	}
	return finalizers
}

func TestReportIsBounded(t *testing.T) {
	// Build a report that cannot fit: long object lines and long notes. The
	// notes explain the gaps, so they have to survive the cut.
	d := &snapshotDiagnostics{namespace: testNamespace, snapshotsListed: true, contentsScanned: true}
	for i := 0; i < maxSnapshotsReported; i++ {
		d.snapshots = append(d.snapshots, summarizeSnapshot(newSnapshot(fmt.Sprintf("snapshot-%02d", i), map[string]interface{}{
			"status": map[string]interface{}{
				"error": map[string]interface{}{"message": strings.Repeat("e", maxMessageLength*2)},
			},
		})))
	}
	for i := 0; i < maxNotes*2; i++ {
		d.addNote("note %02d: %s", i, strings.Repeat("x", maxMessageLength))
	}

	report := d.report()

	if len(report) > maxReportLength {
		t.Errorf("report is %d bytes, want at most %d", len(report), maxReportLength)
	}
	// The notes keep their share of the budget, so the first ones and the
	// marker that accounts for the rest are still there.
	requireContains(t, report,
		"VolumeSnapshot diagnostics for namespace",
		"note: note 00:",
		"further notes omitted",
		"object lines omitted",
	)
}

func TestReportCapsTheNumberOfNotes(t *testing.T) {
	d := &snapshotDiagnostics{namespace: testNamespace, snapshotsListed: true, contentsScanned: true}
	for i := 0; i < maxNotes*2; i++ {
		d.addNote("short note %02d", i)
	}

	report := d.report()

	if got := strings.Count(report, "  note: short note"); got != maxNotes {
		t.Errorf("rendered %d notes, want the cap of %d", got, maxNotes)
	}
	requireContains(t, report, fmt.Sprintf("truncated, %d further notes omitted", len(d.notes)-maxNotes))
}

func TestDescribeAPIErrorIsBounded(t *testing.T) {
	// A server behind a failing proxy can answer with a page-long body, and
	// the notes carry that text into the test log.
	huge := strings.Repeat("x", maxMessageLength*4)
	testCases := []struct {
		name  string
		err   error
		label string
	}{
		{
			name:  "not found",
			err:   apierrors.NewNotFound(schema.GroupResource{Resource: huge}, "snapcontent-1"),
			label: "not found",
		},
		{
			name:  "forbidden",
			err:   apierrors.NewForbidden(utils.SnapshotContentGVR.GroupResource(), "snapcontent-1", fmt.Errorf("%s", huge)),
			label: "not permitted",
		},
		{
			name:  "other",
			err:   apierrors.NewInternalError(fmt.Errorf("%s", huge)),
			label: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := describeAPIError(tc.err)
			if len(got) > maxMessageLength+100 {
				t.Errorf("rendered error is %d bytes, want it capped near %d", len(got), maxMessageLength)
			}
			if !strings.Contains(got, "truncated") {
				t.Errorf("rendered error lacks the truncation marker: %q", got)
			}
			if tc.label != "" && !strings.HasPrefix(got, tc.label) {
				t.Errorf("rendered error = %q, want it to start with %q", got, tc.label)
			}
		})
	}
}

func TestSnapshotDiagnosticsContextIsIndependentOfTheSpecContext(t *testing.T) {
	ctx, cancel := snapshotDiagnosticsContext()
	defer cancel()

	if err := ctx.Err(); err != nil {
		t.Fatalf("diagnostics context is already done: %v", err)
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("diagnostics context has no deadline, so the collection would be unbounded")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > snapshotDiagnosticsTimeout {
		t.Errorf("diagnostics context expires in %v, want at most %v", remaining, snapshotDiagnosticsTimeout)
	}
}

func TestShouldDumpSnapshotDiagnostics(t *testing.T) {
	testCases := []struct {
		name  string
		setup *mockDriverSetup
		when  bool
		want  bool
	}{
		{
			name:  "failed snapshot spec",
			setup: &mockDriverSetup{tp: testParameters{enableSnapshot: true}},
			when:  true,
			want:  true,
		},
		{
			name:  "condition not met",
			setup: &mockDriverSetup{tp: testParameters{enableSnapshot: true}},
			when:  false,
		},
		{
			name:  "spec does not use snapshots",
			setup: &mockDriverSetup{tp: testParameters{enableSnapshot: false}},
			when:  true,
		},
		{
			name:  "already collected for this spec",
			setup: &mockDriverSetup{tp: testParameters{enableSnapshot: true}, snapshotDiagnosticsDumped: true},
			when:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.setup.shouldDumpSnapshotDiagnostics(tc.when); got != tc.want {
				t.Errorf("shouldDumpSnapshotDiagnostics(%v) = %v, want %v", tc.when, got, tc.want)
			}
		})
	}
}

func TestDumpSnapshotDiagnosticsKeepsTheLatchForALaterPhase(t *testing.T) {
	// A collection that could not run must not consume the latch: the
	// cleanup-errors phase has to stay able to report.
	m := &mockDriverSetup{tp: testParameters{enableSnapshot: true}, f: &framework.Framework{}}

	m.dumpSnapshotDiagnostics(true, snapshotDiagnosticsPhaseCleanupEntry)

	if m.snapshotDiagnosticsDumped {
		t.Error("an uninitialized framework consumed the once-per-spec latch")
	}
}

func TestWriteSnapshotDiagnostics(t *testing.T) {
	report := collectSnapshotDiagnostics(context.Background(), &fakeSnapshotReader{}, testNamespace).report()

	t.Run("no report directory configured", func(t *testing.T) {
		// Must be a no-op: with an empty directory the path would be relative
		// and the report would land in the working directory.
		t.Chdir(t.TempDir())
		writeSnapshotDiagnostics("", testNamespace, snapshotDiagnosticsPhaseCleanupEntry, report)
		if _, err := os.Stat("volumesnapshots"); !os.IsNotExist(err) {
			t.Errorf("an empty report directory wrote into the working directory (stat error: %v)", err)
		}
	})

	t.Run("report directory configured", func(t *testing.T) {
		dir := t.TempDir()
		writeSnapshotDiagnostics(dir, testNamespace, snapshotDiagnosticsPhaseCleanupEntry, report)

		path := filepath.Join(dir, "volumesnapshots", testNamespace, snapshotDiagnosticsPhaseCleanupEntry+".txt")
		written, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading the written diagnostics: %v", err)
		}
		if string(written) != report {
			t.Errorf("written diagnostics = %q, want the report %q", string(written), report)
		}
	})

	t.Run("directory cannot be created", func(t *testing.T) {
		// A failed write is logged, never propagated: teardown continues.
		dir := filepath.Join(t.TempDir(), "file")
		if err := os.WriteFile(dir, []byte("not a directory"), 0644); err != nil {
			t.Fatalf("test setup error: %v", err)
		}
		writeSnapshotDiagnostics(dir, testNamespace, snapshotDiagnosticsPhaseCleanupErrors, report)
	})

	t.Run("file cannot be written", func(t *testing.T) {
		if os.Geteuid() == 0 {
			t.Skip("root ignores the directory permissions this case relies on")
		}
		dir := t.TempDir()
		target := filepath.Join(dir, "volumesnapshots", testNamespace)
		if err := os.MkdirAll(target, 0755); err != nil {
			t.Fatalf("test setup error: %v", err)
		}
		if err := os.Chmod(target, 0555); err != nil {
			t.Fatalf("test setup error: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(target, 0755) })

		writeSnapshotDiagnostics(dir, testNamespace, snapshotDiagnosticsPhaseCleanupErrors, report)

		if _, err := os.Stat(filepath.Join(target, snapshotDiagnosticsPhaseCleanupErrors+".txt")); !os.IsNotExist(err) {
			t.Errorf("expected no file to be written into a read-only directory (stat error: %v)", err)
		}
	})
}
