package ginkgo

import (
	"context"
	"io"
	"testing"

	"github.com/onsi/ginkgo/v2/internal/testingtproxy"
	"github.com/onsi/ginkgo/v2/types"
)

/*
GinkgoT() implements an interface that allows third party libraries to integrate with and build on top of Ginkgo.

GinkgoT() is analogous to *testing.T and implements the majority of *testing.T's methods.  It can be typically be used a a drop-in replacement with third-party libraries that accept *testing.T through an interface.

GinkgoT() takes an optional offset argument that can be used to get the
correct line number associated with the failure - though you do not need to use this if you call GinkgoHelper() or GinkgoT().Helper() appropriately

GinkgoT() attempts to mimic the behavior of `testing.T` with the exception of the following:

- Error/Errorf: failures in Ginkgo always immediately stop execution and there is no mechanism to log a failure without aborting the test.  As such Error/Errorf are equivalent to Fatal/Fatalf.
- Parallel() is a no-op as Ginkgo's multi-process parallelism model is substantially different from go test's in-process model.

You can learn more here: https://onsi.github.io/ginkgo/#using-third-party-libraries
*/
func GinkgoT(optionalOffset ...int) FullGinkgoTInterface {
	offset := 1
	if len(optionalOffset) > 0 {
		offset = optionalOffset[0]
	}
	return testingtproxy.New(
		GinkgoWriter,
		Fail,
		Skip,
		DeferCleanup,
		CurrentSpecReport,
		AddReportEntry,
		GinkgoRecover,
		AttachProgressReporter,
		suiteConfig.RandomSeed,
		suiteConfig.ParallelProcess,
		suiteConfig.ParallelTotal,
		reporterConfig.NoColor,
		offset)
}

/*
The portion of the interface returned by GinkgoT() that maps onto methods in the testing package's T.
*/
type GinkgoTInterface interface {
	Cleanup(func())
	Chdir(dir string)
	Context() context.Context
	Setenv(kev, value string)
	Error(args ...any)
	Errorf(format string, args ...any)
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Log(args ...any)
	Logf(format string, args ...any)
	Name() string
	Parallel()
	Skip(args ...any)
	SkipNow()
	Skipf(format string, args ...any)
	Skipped() bool
	TempDir() string
	Attr(key, value string)
	Output() io.Writer
}

/*
Additional methods returned by GinkgoT() that provide deeper integration points into Ginkgo
*/
type FullGinkgoTInterface interface {
	GinkgoTInterface

	AddReportEntryVisibilityAlways(name string, args ...any)
	AddReportEntryVisibilityFailureOrVerbose(name string, args ...any)
	AddReportEntryVisibilityNever(name string, args ...any)

	//Prints to the GinkgoWriter
	Print(a ...any)
	Printf(format string, a ...any)
	Println(a ...any)

	//Provides access to Ginkgo's color formatting, correctly configured to match the color settings specified in the invocation of ginkgo
	F(format string, args ...any) string
	Fi(indentation uint, format string, args ...any) string
	Fiw(indentation uint, maxWidth uint, format string, args ...any) string

	//Generates a formatted string version of the current spec's timeline
	RenderTimeline() string

	GinkgoRecover()
	DeferCleanup(args ...any)

	RandomSeed() int64
	ParallelProcess() int
	ParallelTotal() int

	AttachProgressReporter(func() string) func()
}

/*
GinkgoTB() implements a wrapper that exactly matches the testing.TB interface.

In go 1.18 a new private() function was added to the testing.TB interface. Any function which accepts testing.TB as input needs to be passed in something that directly implements testing.TB.

This wrapper satisfies the testing.TB interface and intended to be used as a drop-in replacement with third party libraries that accept testing.TB.

Similar to GinkgoT(), GinkgoTB() takes an optional offset argument that can be used to get the
correct line number associated with the failure - though you do not need to use this if you call GinkgoHelper() or GinkgoT().Helper() appropriately
*/
func GinkgoTB(optionalOffset ...int) *GinkgoTBWrapper {
	offset := 2
	if len(optionalOffset) > 0 {
		offset = optionalOffset[0]
	}
	return &GinkgoTBWrapper{GinkgoT: GinkgoT(offset)}
}

type GinkgoTBWrapper struct {
	testing.TB
	GinkgoT FullGinkgoTInterface
}

func (g *GinkgoTBWrapper) Cleanup(f func()) {
	g.GinkgoT.Cleanup(f)
}
func (g *GinkgoTBWrapper) Chdir(dir string) {
	g.GinkgoT.Chdir(dir)
}
func (g *GinkgoTBWrapper) Context() context.Context {
	return g.GinkgoT.Context()
}
func (g *GinkgoTBWrapper) Error(args ...any) {
	g.GinkgoT.Error(args...)
}
func (g *GinkgoTBWrapper) Errorf(format string, args ...any) {
	g.GinkgoT.Errorf(format, args...)
}
func (g *GinkgoTBWrapper) Fail() {
	g.GinkgoT.Fail()
}
func (g *GinkgoTBWrapper) FailNow() {
	g.GinkgoT.FailNow()
}
func (g *GinkgoTBWrapper) Failed() bool {
	return g.GinkgoT.Failed()
}
func (g *GinkgoTBWrapper) Fatal(args ...any) {
	g.GinkgoT.Fatal(args...)
}
func (g *GinkgoTBWrapper) Fatalf(format string, args ...any) {
	g.GinkgoT.Fatalf(format, args...)
}
func (g *GinkgoTBWrapper) Helper() {
	types.MarkAsHelper(1)
}
func (g *GinkgoTBWrapper) Log(args ...any) {
	g.GinkgoT.Log(args...)
}
func (g *GinkgoTBWrapper) Logf(format string, args ...any) {
	g.GinkgoT.Logf(format, args...)
}
func (g *GinkgoTBWrapper) Name() string {
	return g.GinkgoT.Name()
}
func (g *GinkgoTBWrapper) Setenv(key, value string) {
	g.GinkgoT.Setenv(key, value)
}
func (g *GinkgoTBWrapper) Skip(args ...any) {
	g.GinkgoT.Skip(args...)
}
func (g *GinkgoTBWrapper) SkipNow() {
	g.GinkgoT.SkipNow()
}
func (g *GinkgoTBWrapper) Skipf(format string, args ...any) {
	g.GinkgoT.Skipf(format, args...)
}
func (g *GinkgoTBWrapper) Skipped() bool {
	return g.GinkgoT.Skipped()
}
func (g *GinkgoTBWrapper) TempDir() string {
	return g.GinkgoT.TempDir()
}
