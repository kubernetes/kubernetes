package tmp

import (
	. "github.com/onsi/ginkgo"
)

var _ = Describe("Testing with Ginkgo", func() {
	It("something important", func() {

		whatever := &UselessStruct{
			T:              GinkgoT(),
			ImportantField: "SECRET_PASSWORD",
		}
		something := &UselessStruct{ImportantField: "string value"}
		assertEqual(GinkgoT(), whatever.ImportantField, "SECRET_PASSWORD")
		assertEqual(GinkgoT(), something.ImportantField, "string value")

		var foo = func(t GinkgoTInterface) {}
		foo(GinkgoT())

		strp := "something"
		testFunc(GinkgoT(), &strp)
		GinkgoT().Fail()
	})
	It("3 things", func() {

		if 3 != 3 {
			GinkgoT().Fail()
		}
	})
})

type UselessStruct struct {
	ImportantField string
	T              GinkgoTInterface
}

var testFunc = func(t GinkgoTInterface, arg *string) {}

func assertEqual(t GinkgoTInterface, arg1, arg2 interface{}) {
	if arg1 != arg2 {
		t.Fail()
	}
}
