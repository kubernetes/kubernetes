package nested

import (
	. "github.com/onsi/ginkgo"
)

var _ = Describe("Testing with Ginkgo", func() {
	It("something less important", func() {

		whatever := &UselessStruct{}
		GinkgoT().Fail(whatever.ImportantField != "SECRET_PASSWORD")
	})
})
