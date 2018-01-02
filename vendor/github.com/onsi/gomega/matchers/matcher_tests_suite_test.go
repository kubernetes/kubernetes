package matchers_test

import (
	"testing"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type myStringer struct {
	a string
}

func (s *myStringer) String() string {
	return s.a
}

type StringAlias string

type myCustomType struct {
	s   string
	n   int
	f   float32
	arr []string
}

func Test(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Gomega Matchers")
}
