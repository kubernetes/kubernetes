package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Tags", func() {
	var pathToTest string
	BeforeEach(func() {
		pathToTest = tmpPath("tags")
		copyIn("tags_tests", pathToTest)
	})

	It("should honor the passed in -tags flag", func() {
		session := startGinkgo(pathToTest, "--noColor")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())
		Ω(output).Should(ContainSubstring("Ran 1 of 1 Specs"))

		session = startGinkgo(pathToTest, "--noColor", "-tags=complex_tests")
		Eventually(session).Should(gexec.Exit(0))
		output = string(session.Out.Contents())
		Ω(output).Should(ContainSubstring("Ran 3 of 3 Specs"))
	})
})
