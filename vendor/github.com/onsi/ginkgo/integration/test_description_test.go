package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gbytes"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("TestDescription", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("test_description")
		copyIn("test_description", pathToTest)
	})

	It("should capture and emit information about the current test", func() {
		session := startGinkgo(pathToTest, "--noColor")
		Eventually(session).Should(gexec.Exit(1))

		Ω(session).Should(gbytes.Say("TestDescription should pass:false"))
		Ω(session).Should(gbytes.Say("TestDescription should fail:true"))
	})
})
