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
	"time"

	"github.com/onsi/ginkgo/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// Bounds for the snapshot diagnostics. The diagnostics run while a spec is
// already failing, so every request, every list page and every rendered field
// is capped: a debugging aid must not turn one failed spec into a slow or
// noisy one.
const (
	// snapshotDiagnosticsTimeout bounds the whole collection. It is applied to
	// a context derived from context.Background(), because the spec context is
	// routinely already expired by the time a wait has timed out.
	snapshotDiagnosticsTimeout = 30 * time.Second

	// maxSnapshotsReported caps the VolumeSnapshots reported for the test
	// namespace.
	maxSnapshotsReported = 20

	// maxContentsReported caps the VolumeSnapshotContents reported, counting
	// both the ones referenced by the namespace's VolumeSnapshots and the ones
	// found by the bounded scan.
	maxContentsReported = 20

	// maxContentGets caps the by-name reads. A page of snapshots can reference
	// twice as many contents, and a read that fails or names an object of
	// another namespace reports nothing, so the number of requests has to be
	// bounded on its own rather than through the reported objects.
	maxContentGets = 20

	// maxContentScanPageSize caps the single page read when scanning the
	// cluster-scoped VolumeSnapshotContents for objects that reference the test
	// namespace. Only one page is ever requested, and only this many of its
	// items are examined even if the server returned more.
	maxContentScanPageSize = 100

	// maxFieldLength caps a single rendered field.
	maxFieldLength = 128

	// maxMessageLength caps a single rendered error message or note.
	maxMessageLength = 512

	// maxFinalizersRendered caps the finalizers listed for one object.
	maxFinalizersRendered = 8

	// maxNotes caps the notes rendered.
	maxNotes = 32

	// maxReportLength is the hard cap on the whole rendered report. The notes
	// are rendered against their own share of it, so the explanation of a gap
	// survives even when the object lines have to be cut.
	maxReportLength = 16384

	// Phases identify when the diagnostics ran. They double as file name stems
	// under the report directory, so they must stay path safe.
	snapshotDiagnosticsPhaseCleanupEntry  = "cleanup-entry"
	snapshotDiagnosticsPhaseCleanupErrors = "cleanup-errors"

	unsetFieldValue = "<unset>"

	reportTruncationMarker = "  ...(report truncated at the diagnostics output cap)\n"
)

// snapshotObjectReader is the slice of the dynamic client the collector needs.
// Keeping it this small lets the collection, filtering and bounding logic be
// unit tested without a cluster or a generated fake.
type snapshotObjectReader interface {
	Get(ctx context.Context, gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error)
	List(ctx context.Context, gvr schema.GroupVersionResource, namespace string, opts metav1.ListOptions) (*unstructured.UnstructuredList, error)
}

type dynamicSnapshotObjectReader struct {
	client dynamic.Interface
}

func (r dynamicSnapshotObjectReader) Get(ctx context.Context, gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error) {
	if namespace == "" {
		return r.client.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
	}
	return r.client.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
}

func (r dynamicSnapshotObjectReader) List(ctx context.Context, gvr schema.GroupVersionResource, namespace string, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	if namespace == "" {
		return r.client.Resource(gvr).List(ctx, opts)
	}
	return r.client.Resource(gvr).Namespace(namespace).List(ctx, opts)
}

// snapshotSummary holds the allowlisted fields of a VolumeSnapshot, as the API
// returned them. Everything that is not needed to explain a stuck deletion is
// left out on purpose: arbitrary annotations and labels, secret references and
// storage handles are never collected. Capping happens when the summary is
// rendered, so that a long name still matches the object it came from.
type snapshotSummary struct {
	name              string
	namespace         string
	uid               string
	deletionTimestamp string
	finalizers        []string
	sourcePVC         string
	sourceContent     string
	snapshotClass     string
	boundContent      string
	readyToUse        string
	creationTime      string
	errorMessage      string
	errorTime         string
}

// String renders the summary. The fields are stored as the API returned them,
// because they are also used to look objects up, and are shortened here: only
// the rendering is capped, never the value a later request is built from.
func (s snapshotSummary) String() string {
	return fmt.Sprintf("VolumeSnapshot %s/%s: uid=%s deletionTimestamp=%s finalizers=%s sourcePVC=%s sourceContent=%s class=%s boundContent=%s readyToUse=%s creationTime=%s error=%s errorTime=%s",
		truncateField(s.namespace), truncateField(s.name), truncateField(s.uid),
		truncateField(s.deletionTimestamp), renderList(s.finalizers), truncateField(s.sourcePVC),
		truncateField(s.sourceContent), truncateField(s.snapshotClass), truncateField(s.boundContent),
		truncateField(s.readyToUse), truncateField(s.creationTime),
		truncateMessage(s.errorMessage), truncateField(s.errorTime))
}

// snapshotContentSummary holds the allowlisted fields of a
// VolumeSnapshotContent. Storage handles (spec.source.volumeHandle,
// spec.source.snapshotHandle, status.snapshotHandle) and secret references are
// deliberately not collected.
type snapshotContentSummary struct {
	name              string
	uid               string
	deletionTimestamp string
	finalizers        []string
	snapshotRefName   string
	snapshotRefNS     string
	snapshotRefUID    string
	deletionPolicy    string
	driver            string
	snapshotClass     string
	readyToUse        string
	creationTime      string
	errorMessage      string
	errorTime         string
}

// String renders the summary, with the same rendering-only capping as
// snapshotSummary.String.
func (c snapshotContentSummary) String() string {
	return fmt.Sprintf("VolumeSnapshotContent %s: uid=%s deletionTimestamp=%s finalizers=%s snapshotRef=%s/%s snapshotRefUID=%s deletionPolicy=%s driver=%s class=%s readyToUse=%s creationTime=%s error=%s errorTime=%s",
		truncateField(c.name), truncateField(c.uid), truncateField(c.deletionTimestamp),
		renderList(c.finalizers), truncateField(c.snapshotRefNS), truncateField(c.snapshotRefName),
		truncateField(c.snapshotRefUID), truncateField(c.deletionPolicy), truncateField(c.driver),
		truncateField(c.snapshotClass), truncateField(c.readyToUse), truncateField(c.creationTime),
		truncateMessage(c.errorMessage), truncateField(c.errorTime))
}

// snapshotDiagnostics is the result of one bounded collection. Notes carry
// everything that did not work (missing CRDs, denied requests, truncated
// pages), so a partial result can never be mistaken for a complete one.
type snapshotDiagnostics struct {
	namespace string
	snapshots []snapshotSummary
	contents  []snapshotContentSummary
	notes     []string

	// Whether the corresponding read actually happened, so that an empty
	// result is never rendered as "nothing is there" when it really means
	// "nothing could be read".
	snapshotsListed bool
	contentsScanned bool
}

func (d *snapshotDiagnostics) addNote(format string, args ...interface{}) {
	d.notes = append(d.notes, fmt.Sprintf(format, args...))
}

// report renders the diagnostics as plain text lines, never longer than
// maxReportLength.
func (d *snapshotDiagnostics) report() string {
	header := fmt.Sprintf("VolumeSnapshot diagnostics for namespace %q\n", d.namespace)
	noteLines := d.renderNotes()
	objectLines := d.renderObjects()

	notesLength := 0
	for _, line := range noteLines {
		notesLength += len(line)
	}

	var b strings.Builder
	b.WriteString(header)

	// The notes explain every gap in the object lines, so they keep their
	// share of the budget and the objects fill what is left.
	budget := maxReportLength - len(header) - notesLength - len(reportTruncationMarker)
	omitted := 0
	for i, line := range objectLines {
		if len(line) > budget {
			omitted = len(objectLines) - i
			break
		}
		b.WriteString(line)
		budget -= len(line)
	}
	if omitted > 0 {
		fmt.Fprintf(&b, "  ...(truncated, %d further object lines omitted)\n", omitted)
	}
	for _, line := range noteLines {
		b.WriteString(line)
	}

	report := b.String()
	if len(report) > maxReportLength {
		report = truncateUTF8(report[:maxReportLength-len(reportTruncationMarker)]) + reportTruncationMarker
	}
	return report
}

func (d *snapshotDiagnostics) renderObjects() []string {
	var lines []string
	if len(d.snapshots) == 0 {
		if d.snapshotsListed {
			lines = append(lines, "  no VolumeSnapshots found\n")
		} else {
			lines = append(lines, "  VolumeSnapshots could not be read, see the notes below\n")
		}
	}
	for _, s := range d.snapshots {
		lines = append(lines, fmt.Sprintf("  %s\n", s.String()))
	}
	if len(d.contents) == 0 {
		if d.contentsScanned {
			lines = append(lines, "  no VolumeSnapshotContents found for this namespace\n")
		} else {
			lines = append(lines, "  VolumeSnapshotContents could not be read, see the notes below\n")
		}
	}
	for _, c := range d.contents {
		lines = append(lines, fmt.Sprintf("  %s\n", c.String()))
	}
	return lines
}

func (d *snapshotDiagnostics) renderNotes() []string {
	var lines []string
	budget := maxReportLength / 2
	for i, n := range d.notes {
		line := fmt.Sprintf("  note: %s\n", truncateMessage(n))
		if i >= maxNotes || len(line) > budget {
			lines = append(lines, fmt.Sprintf("  note: ...(truncated, %d further notes omitted)\n", len(d.notes)-i))
			break
		}
		lines = append(lines, line)
		budget -= len(line)
	}
	return lines
}

// collectSnapshotDiagnostics gathers the VolumeSnapshots of one namespace and
// the VolumeSnapshotContents that belong to them. It never returns an error:
// everything it could not read is recorded as a note, because the caller is
// handling a failure that must not be replaced by a diagnostics failure.
func collectSnapshotDiagnostics(ctx context.Context, reader snapshotObjectReader, namespace string) *snapshotDiagnostics {
	d := &snapshotDiagnostics{namespace: namespace}
	if namespace == "" {
		// The dynamic client reads every namespace when it is given none, and
		// these diagnostics are scoped to the namespace of one spec.
		d.addNote("no namespace to inspect, nothing was collected")
		return d
	}

	var referenced []string
	// requested dedupes the reads; reported dedupes the output. A name that
	// could not be read stays out of reported, so the scan below can still
	// pick the object up.
	requested := map[string]bool{}
	reported := map[string]bool{}

	snapshotList, err := reader.List(ctx, utils.SnapshotGVR, namespace, metav1.ListOptions{Limit: maxSnapshotsReported})
	switch {
	case err != nil:
		d.addNote("could not list VolumeSnapshots in namespace %s: %s", namespace, describeAPIError(err))
	default:
		d.snapshotsListed = true
		for i := range snapshotList.Items {
			if len(d.snapshots) >= maxSnapshotsReported {
				d.addNote("stopped after %d VolumeSnapshots, the returned page held more than the requested limit", maxSnapshotsReported)
				break
			}
			s := summarizeSnapshot(&snapshotList.Items[i])
			d.snapshots = append(d.snapshots, s)
			for _, name := range []string{s.boundContent, s.sourceContent} {
				if name != "" && name != unsetFieldValue && !requested[name] {
					requested[name] = true
					referenced = append(referenced, name)
				}
			}
		}
		if snapshotList.GetContinue() != "" {
			d.addNote("namespace %s holds more VolumeSnapshots than the %d that were requested, the rest were not inspected", namespace, maxSnapshotsReported)
		}
	}

	// Referenced contents first: they are the ones the spec actually worked
	// with, and a Get by name is cheaper and more precise than any scan.
	for i, name := range referenced {
		if len(d.contents) >= maxContentsReported {
			d.addNote("stopped after %d VolumeSnapshotContents, %d further referenced contents were not read", maxContentsReported, len(referenced)-i)
			break
		}
		if i >= maxContentGets {
			d.addNote("stopped after %d VolumeSnapshotContent reads, %d further referenced contents were not read", maxContentGets, len(referenced)-i)
			break
		}
		content, err := reader.Get(ctx, utils.SnapshotContentGVR, "", name)
		if err != nil {
			d.addNote("could not get VolumeSnapshotContent %s: %s", name, describeAPIError(err))
			continue
		}
		// Contents are cluster scoped, so a snapshot of this namespace can
		// name one that is bound somewhere else. That object belongs to
		// another test and is not reported here.
		if ns := nestedString(content, "spec", "volumeSnapshotRef", "namespace"); ns != namespace && ns != unsetFieldValue {
			d.addNote("skipped VolumeSnapshotContent %s, it is bound to a VolumeSnapshot outside namespace %s", name, namespace)
			continue
		}
		reported[name] = true
		d.contents = append(d.contents, summarizeSnapshotContent(content))
	}

	// A content whose VolumeSnapshot is already gone is exactly the leak worth
	// reporting, and no VolumeSnapshot references it any more. One bounded page
	// of the cluster-scoped list, filtered on the snapshot reference, finds it
	// without ever walking the whole cluster.
	if len(d.contents) >= maxContentsReported {
		d.addNote("skipped the VolumeSnapshotContent scan, the report already holds %d contents, so a content whose VolumeSnapshot is gone may be missing", maxContentsReported)
		return d
	}

	contentList, err := reader.List(ctx, utils.SnapshotContentGVR, "", metav1.ListOptions{Limit: maxContentScanPageSize})
	switch {
	case err != nil:
		d.addNote("could not scan VolumeSnapshotContents for namespace %s: %s", namespace, describeAPIError(err))
	default:
		d.contentsScanned = true
		truncatedByCap := false
		truncatedByPage := false
		for i := range contentList.Items {
			if i >= maxContentScanPageSize {
				truncatedByPage = true
				break
			}
			content := &contentList.Items[i]
			if reported[content.GetName()] || !contentReferencesNamespace(content, namespace) {
				continue
			}
			if len(d.contents) >= maxContentsReported {
				truncatedByCap = true
				break
			}
			reported[content.GetName()] = true
			d.contents = append(d.contents, summarizeSnapshotContent(content))
		}
		if truncatedByCap {
			d.addNote("stopped after %d VolumeSnapshotContents, the scanned page held more for namespace %s", maxContentsReported, namespace)
		}
		if truncatedByPage {
			d.addNote("examined only the first %d items of the VolumeSnapshotContent page, the server returned %d", maxContentScanPageSize, len(contentList.Items))
		}
		if contentList.GetContinue() != "" {
			d.addNote("the VolumeSnapshotContent scan read only the first %d items, contents of namespace %s beyond that page were not inspected", maxContentScanPageSize, namespace)
		}
	}

	return d
}

// summarizeSnapshot copies the allowlisted VolumeSnapshot fields.
func summarizeSnapshot(u *unstructured.Unstructured) snapshotSummary {
	s := snapshotSummary{
		name:              u.GetName(),
		namespace:         u.GetNamespace(),
		uid:               string(u.GetUID()),
		deletionTimestamp: unsetFieldValue,
		finalizers:        u.GetFinalizers(),
		sourcePVC:         nestedString(u, "spec", "source", "persistentVolumeClaimName"),
		sourceContent:     nestedString(u, "spec", "source", "volumeSnapshotContentName"),
		snapshotClass:     nestedString(u, "spec", "volumeSnapshotClassName"),
		boundContent:      nestedString(u, "status", "boundVolumeSnapshotContentName"),
		readyToUse:        nestedString(u, "status", "readyToUse"),
		creationTime:      nestedString(u, "status", "creationTime"),
		errorMessage:      nestedString(u, "status", "error", "message"),
		errorTime:         nestedString(u, "status", "error", "time"),
	}
	if ts := u.GetDeletionTimestamp(); ts != nil {
		s.deletionTimestamp = ts.UTC().Format(time.RFC3339)
	}
	return s
}

// summarizeSnapshotContent copies the allowlisted VolumeSnapshotContent fields.
func summarizeSnapshotContent(u *unstructured.Unstructured) snapshotContentSummary {
	c := snapshotContentSummary{
		name:              u.GetName(),
		uid:               string(u.GetUID()),
		deletionTimestamp: unsetFieldValue,
		finalizers:        u.GetFinalizers(),
		snapshotRefName:   nestedString(u, "spec", "volumeSnapshotRef", "name"),
		snapshotRefNS:     nestedString(u, "spec", "volumeSnapshotRef", "namespace"),
		snapshotRefUID:    nestedString(u, "spec", "volumeSnapshotRef", "uid"),
		deletionPolicy:    nestedString(u, "spec", "deletionPolicy"),
		driver:            nestedString(u, "spec", "driver"),
		snapshotClass:     nestedString(u, "spec", "volumeSnapshotClassName"),
		readyToUse:        nestedString(u, "status", "readyToUse"),
		creationTime:      nestedString(u, "status", "creationTime"),
		errorMessage:      nestedString(u, "status", "error", "message"),
		errorTime:         nestedString(u, "status", "error", "time"),
	}
	if ts := u.GetDeletionTimestamp(); ts != nil {
		c.deletionTimestamp = ts.UTC().Format(time.RFC3339)
	}
	return c
}

// contentReferencesNamespace reports whether a VolumeSnapshotContent points at
// a VolumeSnapshot in the given namespace.
func contentReferencesNamespace(u *unstructured.Unstructured, namespace string) bool {
	return nestedString(u, "spec", "volumeSnapshotRef", "namespace") == namespace
}

// nestedString renders a scalar field as a string, whatever JSON type it has.
// VolumeSnapshot and VolumeSnapshotContent disagree on the type of equivalent
// fields (creationTime is a timestamp in one and an integer in the other), and
// a diagnostic has no business failing over that.
func nestedString(u *unstructured.Unstructured, fields ...string) string {
	value, found, err := unstructured.NestedFieldNoCopy(u.Object, fields...)
	if err != nil || !found || value == nil {
		return unsetFieldValue
	}
	switch v := value.(type) {
	case string:
		if v == "" {
			return unsetFieldValue
		}
		return v
	case bool, int64, float64:
		return fmt.Sprintf("%v", v)
	default:
		// Not a scalar: report the kind rather than dumping an arbitrary
		// object, which could carry fields outside the allowlist.
		return fmt.Sprintf("<non-scalar %T>", v)
	}
}

// truncateValue shortens a rendered value and says so. Cutting at a byte
// boundary can split a rune, and an invalid UTF-8 tail in a test log helps
// nobody.
func truncateValue(value string, max int) string {
	if len(value) <= max {
		return value
	}
	return truncateUTF8(value[:max]) + fmt.Sprintf("...(truncated, %d bytes total)", len(value))
}

func truncateUTF8(value string) string {
	return strings.ToValidUTF8(value, "")
}

func truncateMessage(msg string) string {
	return truncateValue(msg, maxMessageLength)
}

func truncateField(value string) string {
	return truncateValue(value, maxFieldLength)
}

func renderList(items []string) string {
	if len(items) == 0 {
		return "[]"
	}
	rendered := make([]string, 0, min(len(items), maxFinalizersRendered))
	for _, item := range items[:min(len(items), maxFinalizersRendered)] {
		rendered = append(rendered, truncateField(item))
	}
	list := "[" + strings.Join(rendered, ",") + "]"
	if len(items) > maxFinalizersRendered {
		list += fmt.Sprintf("(+%d more)", len(items)-maxFinalizersRendered)
	}
	return list
}

// describeAPIError labels the cases a reader has to tell apart: a missing CRD,
// a denied request and a vanished object all mean different things for a stuck
// deletion.
// It is only ever called with a non-nil error, and its result is capped like
// any other rendered message: a server behind a failing proxy can return a
// page-long error body.
func describeAPIError(err error) string {
	switch {
	case apierrors.IsNotFound(err):
		return truncateMessage(fmt.Sprintf("not found (object deleted, or the snapshot CRDs are not installed): %v", err))
	case apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err):
		return truncateMessage(fmt.Sprintf("not permitted: %v", err))
	default:
		return truncateMessage(err.Error())
	}
}

// shouldDumpSnapshotDiagnostics gates the collection. Only the specs that
// exercise snapshots have anything to report; the other mock driver specs would
// just pay for a list that always comes back empty. The test parameters are the
// right gate because init resets them per spec, while the collections of the
// mockDriverSetup shared by a container are not cleared.
func (m *mockDriverSetup) shouldDumpSnapshotDiagnostics(when bool) bool {
	return when && !m.snapshotDiagnosticsDumped && m.tp.enableSnapshot
}

// snapshotDiagnosticsContext returns a context independent of the spec context.
// A spec that failed on a wait has usually exhausted its own context already,
// and a diagnostic inherited from it would collect nothing.
func snapshotDiagnosticsContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), snapshotDiagnosticsTimeout)
}

// dumpSnapshotDiagnostics collects and logs the snapshot state of the test
// namespace at most once per spec, when `when` is true. It is best effort: it
// reports what it could not read and returns normally, so the spec's own
// failure stays the reported one and teardown continues either way. It delays
// teardown by at most snapshotDiagnosticsTimeout.
func (m *mockDriverSetup) dumpSnapshotDiagnostics(when bool, phase string) {
	if !m.shouldDumpSnapshotDiagnostics(when) {
		return
	}
	if m.config == nil || m.config.Framework == nil || m.config.Framework.DynamicClient == nil || m.f.Namespace == nil {
		framework.Logf("Skipping VolumeSnapshot diagnostics (%s): the framework is not initialized", phase)
		return
	}
	m.snapshotDiagnosticsDumped = true

	namespace := m.f.Namespace.Name
	ctx, cancel := snapshotDiagnosticsContext()
	defer cancel()

	ginkgo.By(fmt.Sprintf("collecting VolumeSnapshot diagnostics at %s", phase))
	diagnostics := collectSnapshotDiagnostics(ctx, dynamicSnapshotObjectReader{client: m.config.Framework.DynamicClient}, namespace)
	report := diagnostics.report()
	framework.Logf("VolumeSnapshot diagnostics (%s):\n%s", phase, report)
	writeSnapshotDiagnostics(framework.TestContext.ReportDir, namespace, phase, report)
}

// writeSnapshotDiagnostics also stores the report under the report directory
// when one is configured, following the convention of the other e2e dumps. The
// log output above is the primary channel: not every runner sets --report-dir.
func writeSnapshotDiagnostics(reportDir, namespace, phase, report string) {
	if reportDir == "" {
		return
	}
	dir := filepath.Join(reportDir, "volumesnapshots", namespace)
	if err := os.MkdirAll(dir, 0755); err != nil {
		framework.Logf("Could not create the VolumeSnapshot diagnostics directory %s: %v", dir, err)
		return
	}
	path := filepath.Join(dir, phase+".txt")
	if err := os.WriteFile(path, []byte(report), 0644); err != nil {
		framework.Logf("Could not write the VolumeSnapshot diagnostics to %s: %v", path, err)
	}
}
