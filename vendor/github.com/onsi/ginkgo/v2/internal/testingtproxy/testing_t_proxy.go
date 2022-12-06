package testingtproxy

import (
	"fmt"
	"io"
	"os"

	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/types"
)

type failFunc func(message string, callerSkip ...int)
type skipFunc func(message string, callerSkip ...int)
type cleanupFunc func(args ...interface{})
type reportFunc func() types.SpecReport

func New(writer io.Writer, fail failFunc, skip skipFunc, cleanup cleanupFunc, report reportFunc, offset int) *ginkgoTestingTProxy {
	return &ginkgoTestingTProxy{
		fail:    fail,
		offset:  offset,
		writer:  writer,
		skip:    skip,
		cleanup: cleanup,
		report:  report,
	}
}

type ginkgoTestingTProxy struct {
	fail    failFunc
	skip    skipFunc
	cleanup cleanupFunc
	report  reportFunc
	offset  int
	writer  io.Writer
}

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
	// No-op
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
