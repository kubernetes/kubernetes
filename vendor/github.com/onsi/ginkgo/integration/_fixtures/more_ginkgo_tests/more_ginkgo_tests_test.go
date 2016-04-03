package more_ginkgo_tests_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/integration/_fixtures/more_ginkgo_tests"
	. "github.com/onsi/gomega"
)

var _ = Describe("MoreGinkgoTests", func() {
	It("should pass", func() {
		Ω(AlwaysTrue()).Should(BeTrue())
	})

	It("should always pass", func() {
		Ω(AlwaysTrue()).Should(BeTrue())
	})
})
