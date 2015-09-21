package codelocation_test

import (
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
	"runtime"
)

var _ = Describe("CodeLocation", func() {
	var (
		codeLocation       types.CodeLocation
		expectedFileName   string
		expectedLineNumber int
	)

	caller0 := func() {
		codeLocation = codelocation.New(1)
	}

	caller1 := func() {
		_, expectedFileName, expectedLineNumber, _ = runtime.Caller(0)
		expectedLineNumber += 2
		caller0()
	}

	BeforeEach(func() {
		caller1()
	})

	It("should use the passed in skip parameter to pick out the correct file & line number", func() {
		Ω(codeLocation.FileName).Should(Equal(expectedFileName))
		Ω(codeLocation.LineNumber).Should(Equal(expectedLineNumber))
	})

	Describe("stringer behavior", func() {
		It("should stringify nicely", func() {
			Ω(codeLocation.String()).Should(ContainSubstring("code_location_test.go:%d", expectedLineNumber))
		})
	})

	//There's no better way than to test this private method as it
	//goes out of its way to prune out ginkgo related code in the stack trace
	Describe("PruneStack", func() {
		It("should remove any references to ginkgo and pkg/testing and pkg/runtime", func() {
			input := `/Skip/me
Skip: skip()
/Skip/me
Skip: skip()
/Users/whoever/gospace/src/github.com/onsi/ginkgo/whatever.go:10 (0x12314)
Something: Func()
/Users/whoever/gospace/src/github.com/onsi/ginkgo/whatever_else.go:10 (0x12314)
SomethingInternalToGinkgo: Func()
/usr/goroot/pkg/strings/oops.go:10 (0x12341)
Oops: BlowUp()
/Users/whoever/gospace/src/mycode/code.go:10 (0x12341)
MyCode: Func()
/Users/whoever/gospace/src/mycode/code_test.go:10 (0x12341)
MyCodeTest: Func()
/Users/whoever/gospace/src/mycode/code_suite_test.go:12 (0x37f08)
TestFoo: RunSpecs(t, "Foo Suite")
/usr/goroot/pkg/testing/testing.go:12 (0x37f08)
TestingT: Blah()
/usr/goroot/pkg/runtime/runtime.go:12 (0x37f08)
Something: Func()
`
			prunedStack := codelocation.PruneStack(input, 1)
			Ω(prunedStack).Should(Equal(`/usr/goroot/pkg/strings/oops.go:10 (0x12341)
Oops: BlowUp()
/Users/whoever/gospace/src/mycode/code.go:10 (0x12341)
MyCode: Func()
/Users/whoever/gospace/src/mycode/code_test.go:10 (0x12341)
MyCodeTest: Func()
/Users/whoever/gospace/src/mycode/code_suite_test.go:12 (0x37f08)
TestFoo: RunSpecs(t, "Foo Suite")`))
		})
	})
})
