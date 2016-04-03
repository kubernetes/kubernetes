package fail_fixture_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = It("handles top level skips", func() {
	Skip("a top level skip on line 9")
	println("NEVER SEE THIS")
})

var _ = It("handles async top level skips", func(done Done) {
	Skip("an async top level skip on line 14")
	println("NEVER SEE THIS")
}, 0.1)

var _ = It("SKIP in a goroutine", func(done Done) {
	go func() {
		defer GinkgoRecover()
		Skip("a top level goroutine skip on line 21")
		println("NEVER SEE THIS")
	}()
}, 0.1)

var _ = Describe("Excercising different skip modes", func() {
	It("synchronous skip", func() {
		Skip("a sync SKIP")
		println("NEVER SEE THIS")
	})

	It("async skip", func(done Done) {
		Skip("an async SKIP")
		println("NEVER SEE THIS")
	}, 0.1)

	It("SKIP in a goroutine", func(done Done) {
		go func() {
			defer GinkgoRecover()
			Skip("a goroutine SKIP")
			println("NEVER SEE THIS")
		}()
	}, 0.1)

	Measure("a SKIP measure", func(Benchmarker) {
		Skip("a measure SKIP")
		println("NEVER SEE THIS")
	}, 1)
})


var _ = Describe("SKIP in a BeforeEach", func() {
	BeforeEach(func() {
		Skip("a BeforeEach SKIP")
		println("NEVER SEE THIS")
	})

	It("a SKIP BeforeEach", func() {
		println("NEVER SEE THIS")
	})
})

var _ = Describe("SKIP in an AfterEach", func() {
	AfterEach(func() {
		Skip("an AfterEach SKIP")
		println("NEVER SEE THIS")
	})

	It("a SKIP AfterEach", func() {
		Expect(true).To(BeTrue())
	})
})