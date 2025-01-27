package ginkgo

import (
	"io"

	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/internal/global"
	"github.com/onsi/ginkgo/v2/types"
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

func NewWriter(w io.Writer) *internal.Writer {
	return internal.NewWriter(w)
}

func GetWriter() *internal.Writer {
	return GinkgoWriter.(*internal.Writer)
}

func SetReporterConfig(r types.ReporterConfig) {
	reporterConfig = r
}
