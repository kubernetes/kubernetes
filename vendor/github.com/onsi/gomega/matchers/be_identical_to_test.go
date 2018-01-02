package matchers_test

import (
	"errors"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeIdenticalTo", func() {
	Context("when asserting that nil equals nil", func() {
		It("should error", func() {
			success, err := (&BeIdenticalToMatcher{Expected: nil}).Match(nil)

			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	It("should treat the same pointer to a struct as identical", func() {
		mySpecialStruct := myCustomType{}
		Ω(&mySpecialStruct).Should(BeIdenticalTo(&mySpecialStruct))
		Ω(&myCustomType{}).ShouldNot(BeIdenticalTo(&mySpecialStruct))
	})

	It("should be strict about types", func() {
		Ω(5).ShouldNot(BeIdenticalTo("5"))
		Ω(5).ShouldNot(BeIdenticalTo(5.0))
		Ω(5).ShouldNot(BeIdenticalTo(3))
	})

	It("should treat primtives as identical", func() {
		Ω("5").Should(BeIdenticalTo("5"))
		Ω("5").ShouldNot(BeIdenticalTo("55"))

		Ω(5.55).Should(BeIdenticalTo(5.55))
		Ω(5.55).ShouldNot(BeIdenticalTo(6.66))

		Ω(5).Should(BeIdenticalTo(5))
		Ω(5).ShouldNot(BeIdenticalTo(55))
	})

	It("should treat the same pointers to a slice as identical", func() {
		mySlice := []int{1, 2}
		Ω(&mySlice).Should(BeIdenticalTo(&mySlice))
		Ω(&mySlice).ShouldNot(BeIdenticalTo(&[]int{1, 2}))
	})

	It("should treat the same pointers to a map as identical", func() {
		myMap := map[string]string{"a": "b", "c": "d"}
		Ω(&myMap).Should(BeIdenticalTo(&myMap))
		Ω(myMap).ShouldNot(BeIdenticalTo(map[string]string{"a": "b", "c": "d"}))
	})

	It("should treat the same pointers to an error as identical", func() {
		myError := errors.New("foo")
		Ω(&myError).Should(BeIdenticalTo(&myError))
		Ω(errors.New("foo")).ShouldNot(BeIdenticalTo(errors.New("bar")))
	})
})
