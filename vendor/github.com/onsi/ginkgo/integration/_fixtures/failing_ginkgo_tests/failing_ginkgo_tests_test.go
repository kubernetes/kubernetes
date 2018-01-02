package failing_ginkgo_tests_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/integration/_fixtures/failing_ginkgo_tests"
	. "github.com/onsi/gomega"
)

var _ = Describe("FailingGinkgoTests", func() {
	It("should fail", func() {
		Ω(AlwaysFalse()).Should(BeTrue())
	})

	It("should pass", func() {
		Ω(AlwaysFalse()).Should(BeFalse())
	})
})
