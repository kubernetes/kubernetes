package spec_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/spec"
	. "github.com/onsi/gomega"
)

var _ = Describe("ParallelizedIndexRange", func() {
	var startIndex, count int

	It("should return the correct index range for 4 tests on 2 nodes", func() {
		startIndex, count = ParallelizedIndexRange(4, 2, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(4, 2, 2)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(2))
	})

	It("should return the correct index range for 5 tests on 2 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 2, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(3))

		startIndex, count = ParallelizedIndexRange(5, 2, 2)
		Ω(startIndex).Should(Equal(3))
		Ω(count).Should(Equal(2))
	})

	It("should return the correct index range for 5 tests on 3 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 3, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(5, 3, 2)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(5, 3, 3)
		Ω(startIndex).Should(Equal(4))
		Ω(count).Should(Equal(1))
	})

	It("should return the correct index range for 5 tests on 4 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 4, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(5, 4, 2)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 4, 3)
		Ω(startIndex).Should(Equal(3))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 4, 4)
		Ω(startIndex).Should(Equal(4))
		Ω(count).Should(Equal(1))
	})

	It("should return the correct index range for 5 tests on 5 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 5, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 5, 2)
		Ω(startIndex).Should(Equal(1))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 5, 3)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 5, 4)
		Ω(startIndex).Should(Equal(3))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 5, 5)
		Ω(startIndex).Should(Equal(4))
		Ω(count).Should(Equal(1))
	})

	It("should return the correct index range for 5 tests on 6 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 6, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 6, 2)
		Ω(startIndex).Should(Equal(1))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 6, 3)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 6, 4)
		Ω(startIndex).Should(Equal(3))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 6, 5)
		Ω(startIndex).Should(Equal(4))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(5, 6, 6)
		Ω(count).Should(Equal(0))
	})

	It("should return the correct index range for 5 tests on 7 nodes", func() {
		startIndex, count = ParallelizedIndexRange(5, 7, 6)
		Ω(count).Should(Equal(0))

		startIndex, count = ParallelizedIndexRange(5, 7, 7)
		Ω(count).Should(Equal(0))
	})

	It("should return the correct index range for 11 tests on 7 nodes", func() {
		startIndex, count = ParallelizedIndexRange(11, 7, 1)
		Ω(startIndex).Should(Equal(0))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(11, 7, 2)
		Ω(startIndex).Should(Equal(2))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(11, 7, 3)
		Ω(startIndex).Should(Equal(4))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(11, 7, 4)
		Ω(startIndex).Should(Equal(6))
		Ω(count).Should(Equal(2))

		startIndex, count = ParallelizedIndexRange(11, 7, 5)
		Ω(startIndex).Should(Equal(8))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(11, 7, 6)
		Ω(startIndex).Should(Equal(9))
		Ω(count).Should(Equal(1))

		startIndex, count = ParallelizedIndexRange(11, 7, 7)
		Ω(startIndex).Should(Equal(10))
		Ω(count).Should(Equal(1))
	})

})
