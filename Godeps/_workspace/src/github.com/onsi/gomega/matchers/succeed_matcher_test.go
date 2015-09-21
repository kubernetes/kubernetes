package matchers_test

import (
	"errors"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

func Erroring() error {
	return errors.New("bam")
}

func NotErroring() error {
	return nil
}

type AnyType struct{}

func Invalid() *AnyType {
	return nil
}

var _ = Describe("Succeed", func() {
	It("should succeed if the function succeeds", func() {
		Ω(NotErroring()).Should(Succeed())
	})

	It("should succeed (in the negated) if the function errored", func() {
		Ω(Erroring()).ShouldNot(Succeed())
	})

	It("should not if passed a non-error", func() {
		success, err := (&SucceedMatcher{}).Match(Invalid())
		Ω(success).Should(BeFalse())
		Ω(err).Should(HaveOccurred())
	})
})
