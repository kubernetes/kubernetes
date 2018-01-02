package suite_command_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Testing suite command", func() {
	It("it should succeed", func() {
		Ω(true).Should(Equal(true))
	})

	PIt("a failing test", func() {
		It("should fail", func() {
			Ω(true).Should(Equal(false))
		})
	})
})
