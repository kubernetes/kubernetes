package testingtproxy

import (
	"fmt"
	"io"
	"os"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/types"
)

type failFunc func(message string, callerSkip ...int)
type skipFunc func(message string, callerSkip ...int)
type cleanupFunc func(args ...any)
type reportFunc func() types.SpecReport
type addReportEntryFunc func(names string, args ...any)
type ginkgoWriterInterface interface {
	io.Writer

	Print(a ...interface{})
	Printf(format string, a ...interface{})
	Println(a ...interface{})
}
type ginkgoRecoverFunc func()
type attachProgressReporterFunc func(func() string) func()

func New(writer ginkgoWriterInterface, fail failFunc, skip skipFunc, cleanup cleanupFunc, report reportFunc, addReportEntry addReportEntryFunc, ginkgoRecover ginkgoRecoverFunc, attachProgressReporter attachProgressReporterFunc, randomSeed int64, parallelProcess int, parallelTotal int, noColor bool, offset int) *ginkgoTestingTProxy {
	return &ginkgoTestingTProxy{
		fail:                   fail,
		offset:                 offset,
		writer:                 writer,
		skip:                   skip,
		cleanup:                cleanup,
		report:                 report,
		addReportEntry:         addReportEntry,
		ginkgoRecover:          ginkgoRecover,
		attachProgressReporter: attachProgressReporter,
		randomSeed:             randomSeed,
		parallelProcess:        parallelProcess,
		parallelTotal:          parallelTotal,
		f:                      formatter.NewWithNoColorBool(noColor),
	}
}

type ginkgoTestingTProxy struct {
	fail                   failFunc
	skip                   skipFunc
	cleanup                cleanupFunc
	report                 reportFunc
	offset                 int
	writer                 ginkgoWriterInterface
	addReportEntry         addReportEntryFunc
	ginkgoRecover          ginkgoRecoverFunc
	attachProgressReporter attachProgressReporterFunc
	randomSeed             int64
	parallelProcess        int
	parallelTotal          int
	f                      formatter.Formatter
}

// basic testing.T support

func (t *ginkgoTestingTProxy) Cleanup(f func()) {
	t.cleanup(f, internal.Offset(1))
}

func (t *ginkgoTestingTProxy) Setenv(key, value string) {
	originalValue, exists := os.LookupEnv(key)
	if exists {
		t.cleanup(os.Setenv, key, originalValue, internal.Offset(1))
	} else {
		t.cleanup(os.Unsetenv, key, internal.Offset(1))
	}

	err := os.Setenv(key, value)
	if err != nil {
		t.fail(fmt.Sprintf("Failed to set environment variable: %v", err), 1)
	}
}

func (t *ginkgoTestingTProxy) Error(args ...interface{}) {
	t.fail(fmt.Sprintln(args...), t.offset)
}

func (t *ginkgoTestingTProxy) Errorf(format string, args ...interface{}) {
	t.fail(fmt.Sprintf(format, args...), t.offset)
}

func (t *ginkgoTestingTProxy) Fail() {
	t.fail("failed", t.offset)
}

func (t *ginkgoTestingTProxy) FailNow() {
	t.fail("failed", t.offset)
}

func (t *ginkgoTestingTProxy) Failed() bool {
	return t.report().Failed()
}

func (t *ginkgoTestingTProxy) Fatal(args ...interface{}) {
	t.fail(fmt.Sprintln(args...), t.offset)
}

func (t *ginkgoTestingTProxy) Fatalf(format string, args ...interface{}) {
	t.fail(fmt.Sprintf(format, args...), t.offset)
}

func (t *ginkgoTestingTProxy) Helper() {
	types.MarkAsHelper(1)
}

func (t *ginkgoTestingTProxy) Log(args ...interface{}) {
	fmt.Fprintln(t.writer, args...)
}

func (t *ginkgoTestingTProxy) Logf(format string, args ...interface{}) {
	t.Log(fmt.Sprintf(format, args...))
}

func (t *ginkgoTestingTProxy) Name() string {
	return t.report().FullText()
}

func (t *ginkgoTestingTProxy) Parallel() {
	// No-op
}

func (t *ginkgoTestingTProxy) Skip(args ...interface{}) {
	t.skip(fmt.Sprintln(args...), t.offset)
}

func (t *ginkgoTestingTProxy) SkipNow() {
	t.skip("skip", t.offset)
}

func (t *ginkgoTestingTProxy) Skipf(format string, args ...interface{}) {
	t.skip(fmt.Sprintf(format, args...), t.offset)
}

func (t *ginkgoTestingTProxy) Skipped() bool {
	return t.report().State.Is(types.SpecStateSkipped)
}

func (t *ginkgoTestingTProxy) TempDir() string {
	tmpDir, err := os.MkdirTemp("", "ginkgo")
	if err != nil {
		t.fail(fmt.Sprintf("Failed to create temporary directory: %v", err), 1)
		return ""
	}
	t.cleanup(os.RemoveAll, tmpDir)

	return tmpDir
}

// FullGinkgoTInterface
func (t *ginkgoTestingTProxy) AddReportEntryVisibilityAlways(name string, args ...any) {
	finalArgs := []any{internal.Offset(1), types.ReportEntryVisibilityAlways}
	t.addReportEntry(name, append(finalArgs, args...)...)
}
func (t *ginkgoTestingTProxy) AddReportEntryVisibilityFailureOrVerbose(name string, args ...any) {
	finalArgs := []any{internal.Offset(1), types.ReportEntryVisibilityFailureOrVerbose}
	t.addReportEntry(name, append(finalArgs, args...)...)
}
func (t *ginkgoTestingTProxy) AddReportEntryVisibilityNever(name string, args ...any) {
	finalArgs := []any{internal.Offset(1), types.ReportEntryVisibilityNever}
	t.addReportEntry(name, append(finalArgs, args...)...)
}
func (t *ginkgoTestingTProxy) Print(a ...any) {
	t.writer.Print(a...)
}
func (t *ginkgoTestingTProxy) Printf(format string, a ...any) {
	t.writer.Printf(format, a...)
}
func (t *ginkgoTestingTProxy) Println(a ...any) {
	t.writer.Println(a...)
}
func (t *ginkgoTestingTProxy) F(format string, args ...any) string {
	return t.f.F(format, args...)
}
func (t *ginkgoTestingTProxy) Fi(indentation uint, format string, args ...any) string {
	return t.f.Fi(indentation, format, args...)
}
func (t *ginkgoTestingTProxy) Fiw(indentation uint, maxWidth uint, format string, args ...any) string {
	return t.f.Fiw(indentation, maxWidth, format, args...)
}
func (t *ginkgoTestingTProxy) GinkgoRecover() {
	t.ginkgoRecover()
}
func (t *ginkgoTestingTProxy) DeferCleanup(args ...any) {
	finalArgs := []any{internal.Offset(1)}
	t.cleanup(append(finalArgs, args...)...)
}
func (t *ginkgoTestingTProxy) RandomSeed() int64 {
	return t.randomSeed
}
func (t *ginkgoTestingTProxy) ParallelProcess() int {
	return t.parallelProcess
}
func (t *ginkgoTestingTProxy) ParallelTotal() int {
	return t.parallelTotal
}
func (t *ginkgoTestingTProxy) AttachProgressReporter(f func() string) func() {
	return t.attachProgressReporter(f)
}
