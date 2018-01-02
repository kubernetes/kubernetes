package gstruct_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/gstruct"
)

var _ = Describe("PointTo", func() {
	It("should fail when passed nil", func() {
		var p *struct{}
		Ω(p).Should(BeNil())
	})

	It("should succeed when passed non-nil pointer", func() {
		var s struct{}
		Ω(&s).Should(PointTo(Ignore()))
	})

	It("should unwrap the pointee value", func() {
		i := 1
		Ω(&i).Should(PointTo(Equal(1)))
		Ω(&i).ShouldNot(PointTo(Equal(2)))
	})

	It("should work with nested pointers", func() {
		i := 1
		ip := &i
		ipp := &ip
		Ω(ipp).Should(PointTo(PointTo(Equal(1))))
		Ω(ipp).ShouldNot(PointTo(PointTo(Equal(2))))
	})
})
