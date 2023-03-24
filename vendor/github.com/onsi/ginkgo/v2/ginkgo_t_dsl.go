package ginkgo

import (
	"github.com/onsi/ginkgo/v2/internal/testingtproxy"
)

/*
GinkgoT() implements an interface that allows third party libraries to integrate with and build on top of Ginkgo.

GinkgoT() is analogous to *testing.T and implements the majority of *testing.T's methods.  It can be typically be used a a drop-in replacement with third-party libraries that accept *testing.T through an interface.

GinkgoT() takes an optional offset argument that can be used to get the
correct line number associated with the failure - though you do not need to use this if you call GinkgoHelper() or GinkgoT().Helper() appropriately

You can learn more here: https://onsi.github.io/ginkgo/#using-third-party-libraries
*/
func GinkgoT(optionalOffset ...int) FullGinkgoTInterface {
	offset := 3
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
	Setenv(kev, value string)
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Helper()
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	Parallel()
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool
	TempDir() string
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
	Print(a ...interface{})
	Printf(format string, a ...interface{})
	Println(a ...interface{})

	//Provides access to Ginkgo's color formatting, correctly configured to match the color settings specified in the invocation of ginkgo
	F(format string, args ...any) string
	Fi(indentation uint, format string, args ...any) string
	Fiw(indentation uint, maxWidth uint, format string, args ...any) string

	GinkgoRecover()
	DeferCleanup(args ...any)

	RandomSeed() int64
	ParallelProcess() int
	ParallelTotal() int

	AttachProgressReporter(func() string) func()
}
