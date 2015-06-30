package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Skipping Specs", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("skipping")
		copyIn("skip_fixture", pathToTest)
	})

	It("should skip in all the possible ways", func() {
		session := startGinkgo(pathToTest, "--noColor")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).ShouldNot(ContainSubstring("NEVER SEE THIS"))

		Ω(output).Should(ContainSubstring("a top level skip on line 9"))
		Ω(output).Should(ContainSubstring("skip_fixture_test.go:9"))
		Ω(output).Should(ContainSubstring("an async top level skip on line 14"))
		Ω(output).Should(ContainSubstring("skip_fixture_test.go:14"))
		Ω(output).Should(ContainSubstring("a top level goroutine skip on line 21"))
		Ω(output).Should(ContainSubstring("skip_fixture_test.go:21"))

		Ω(output).Should(ContainSubstring("a sync SKIP"))
		Ω(output).Should(ContainSubstring("an async SKIP"))
		Ω(output).Should(ContainSubstring("a goroutine SKIP"))
		Ω(output).Should(ContainSubstring("a measure SKIP"))

		Ω(output).Should(ContainSubstring("S [SKIPPING] in Spec Setup (BeforeEach) ["))
		Ω(output).Should(ContainSubstring("a BeforeEach SKIP"))
		Ω(output).Should(ContainSubstring("S [SKIPPING] in Spec Teardown (AfterEach) ["))
		Ω(output).Should(ContainSubstring("an AfterEach SKIP"))

		Ω(output).Should(ContainSubstring("0 Passed | 0 Failed | 0 Pending | 9 Skipped"))
	})
})
