package progress_fixture_test

import (
	"fmt"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("ProgressFixture", func() {
	BeforeEach(func() {
		fmt.Fprintln(GinkgoWriter, ">outer before<")
	})

	JustBeforeEach(func() {
		fmt.Fprintln(GinkgoWriter, ">outer just before<")
	})

	AfterEach(func() {
		fmt.Fprintln(GinkgoWriter, ">outer after<")
	})

	Context("Inner Context", func() {
		BeforeEach(func() {
			fmt.Fprintln(GinkgoWriter, ">inner before<")
		})

		JustBeforeEach(func() {
			fmt.Fprintln(GinkgoWriter, ">inner just before<")
		})

		AfterEach(func() {
			fmt.Fprintln(GinkgoWriter, ">inner after<")
		})

		It("should emit progress as it goes", func() {
			fmt.Fprintln(GinkgoWriter, ">it<")
		})
	})
})
