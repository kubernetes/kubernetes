package ginkgo

import (
	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/internal/global"
)

func AppendSpecText(test *internal.Spec, text string) {
	test.AppendText(text)
}

func GetSuite() *internal.Suite {
	return global.Suite
}

func GetFailer() *internal.Failer {
	return global.Failer
}

func GetWriter() *internal.Writer {
	return GinkgoWriter.(*internal.Writer)
}
