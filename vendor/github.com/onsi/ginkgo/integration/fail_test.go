package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Failing Specs", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("failing")
		copyIn("fail_fixture", pathToTest)
	})

	It("should fail in all the possible ways", func() {
		session := startGinkgo(pathToTest, "--noColor")
		Eventually(session).Should(gexec.Exit(1))
		output := string(session.Out.Contents())

		Ω(output).ShouldNot(ContainSubstring("NEVER SEE THIS"))

		Ω(output).Should(ContainSubstring("a top level failure on line 9"))
		Ω(output).Should(ContainSubstring("fail_fixture_test.go:9"))
		Ω(output).Should(ContainSubstring("an async top level failure on line 14"))
		Ω(output).Should(ContainSubstring("fail_fixture_test.go:14"))
		Ω(output).Should(ContainSubstring("a top level goroutine failure on line 21"))
		Ω(output).Should(ContainSubstring("fail_fixture_test.go:21"))

		Ω(output).Should(ContainSubstring("a sync failure"))
		Ω(output).Should(MatchRegexp(`Test Panicked\n\s+a sync panic`))
		Ω(output).Should(ContainSubstring("a sync FAIL failure"))
		Ω(output).Should(ContainSubstring("async timeout [It]"))
		Ω(output).Should(ContainSubstring("Timed out"))
		Ω(output).Should(ContainSubstring("an async failure"))
		Ω(output).Should(MatchRegexp(`Test Panicked\n\s+an async panic`))
		Ω(output).Should(ContainSubstring("an async FAIL failure"))
		Ω(output).Should(ContainSubstring("a goroutine FAIL failure"))
		Ω(output).Should(ContainSubstring("a goroutine failure"))
		Ω(output).Should(MatchRegexp(`Test Panicked\n\s+a goroutine panic`))
		Ω(output).Should(ContainSubstring("a measure failure"))
		Ω(output).Should(ContainSubstring("a measure FAIL failure"))
		Ω(output).Should(MatchRegexp(`Test Panicked\n\s+a measure panic`))

		Ω(output).Should(ContainSubstring("0 Passed | 16 Failed"))
	})
})
