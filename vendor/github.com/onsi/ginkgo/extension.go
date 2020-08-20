package ginkgo

import (
	"github.com/onsi/ginkgo/internal/suite"
	"github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/types"
)

func GlobalSuite() *suite.Suite {
	return globalSuite
}

func GinkgoWriterType() *writer.Writer {
	return GinkgoWriter.(*writer.Writer)
}

func WalkTests(fn func(name, parentName string, node types.TestNode)) {
	globalSuite.WalkTests(fn)
}
