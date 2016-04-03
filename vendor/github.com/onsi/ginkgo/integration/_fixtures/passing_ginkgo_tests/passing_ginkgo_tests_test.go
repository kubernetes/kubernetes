package passing_ginkgo_tests_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/integration/_fixtures/passing_ginkgo_tests"
	. "github.com/onsi/gomega"
)

var _ = Describe("PassingGinkgoTests", func() {
	It("should proxy strings", func() {
		Ω(StringIdentity("foo")).Should(Equal("foo"))
	})

	It("should proxy integers", func() {
		Ω(IntegerIdentity(3)).Should(Equal(3))
	})

	It("should do it again", func() {
		Ω(StringIdentity("foo")).Should(Equal("foo"))
		Ω(IntegerIdentity(3)).Should(Equal(3))
	})

	It("should be able to run Bys", func() {
		By("emitting one By")
		Ω(3).Should(Equal(3))

		By("emitting another By")
		Ω(4).Should(Equal(4))
	})
})
