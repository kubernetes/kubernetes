package remote_test

import (
	"fmt"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/remote"
	. "github.com/onsi/gomega"
	"os"
)

var _ = Describe("OutputInterceptor", func() {
	var interceptor OutputInterceptor

	BeforeEach(func() {
		interceptor = NewOutputInterceptor()
	})

	It("should capture all stdout/stderr output", func() {
		err := interceptor.StartInterceptingOutput()
		Ω(err).ShouldNot(HaveOccurred())

		fmt.Fprint(os.Stdout, "STDOUT")
		fmt.Fprint(os.Stderr, "STDERR")
		print("PRINT")

		output, err := interceptor.StopInterceptingAndReturnOutput()

		Ω(output).Should(Equal("STDOUTSTDERRPRINT"))
		Ω(err).ShouldNot(HaveOccurred())
	})

	It("should error if told to intercept output twice", func() {
		err := interceptor.StartInterceptingOutput()
		Ω(err).ShouldNot(HaveOccurred())

		print("A")

		err = interceptor.StartInterceptingOutput()
		Ω(err).Should(HaveOccurred())

		print("B")

		output, err := interceptor.StopInterceptingAndReturnOutput()

		Ω(output).Should(Equal("AB"))
		Ω(err).ShouldNot(HaveOccurred())
	})

	It("should allow multiple interception sessions", func() {
		err := interceptor.StartInterceptingOutput()
		Ω(err).ShouldNot(HaveOccurred())
		print("A")
		output, err := interceptor.StopInterceptingAndReturnOutput()
		Ω(output).Should(Equal("A"))
		Ω(err).ShouldNot(HaveOccurred())

		err = interceptor.StartInterceptingOutput()
		Ω(err).ShouldNot(HaveOccurred())
		print("B")
		output, err = interceptor.StopInterceptingAndReturnOutput()
		Ω(output).Should(Equal("B"))
		Ω(err).ShouldNot(HaveOccurred())
	})
})
