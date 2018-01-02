package integration_test

import (
	"regexp"
	"runtime"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Verbose And Succinct Mode", func() {
	var pathToTest string
	var otherPathToTest string

	isWindows := (runtime.GOOS == "windows")
	denoter := "•"

	if isWindows {
		denoter = "+"
	}

	Context("when running one package", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
		})

		It("should default to non-succinct mode", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(0))
			output := session.Out.Contents()

			Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
		})
	})

	Context("when running more than one package", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
			otherPathToTest = tmpPath("more_ginkgo")
			copyIn("more_ginkgo_tests", otherPathToTest)
		})

		Context("with no flags set", func() {
			It("should default to succinct mode", func() {
				session := startGinkgo(pathToTest, "--noColor", pathToTest, otherPathToTest)
				Eventually(session).Should(gexec.Exit(0))
				output := session.Out.Contents()

				Ω(output).Should(MatchRegexp(`\] Passing_ginkgo_tests Suite - 4/4 specs [%s]{4} SUCCESS!`, regexp.QuoteMeta(denoter)))
				Ω(output).Should(MatchRegexp(`\] More_ginkgo_tests Suite - 2/2 specs [%s]{2} SUCCESS!`, regexp.QuoteMeta(denoter)))
			})
		})

		Context("with --succinct=false", func() {
			It("should not be in succinct mode", func() {
				session := startGinkgo(pathToTest, "--noColor", "--succinct=false", pathToTest, otherPathToTest)
				Eventually(session).Should(gexec.Exit(0))
				output := session.Out.Contents()

				Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
				Ω(output).Should(ContainSubstring("Running Suite: More_ginkgo_tests Suite"))
			})
		})

		Context("with -v", func() {
			It("should not be in succinct mode, but should be verbose", func() {
				session := startGinkgo(pathToTest, "--noColor", "-v", pathToTest, otherPathToTest)
				Eventually(session).Should(gexec.Exit(0))
				output := session.Out.Contents()

				Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
				Ω(output).Should(ContainSubstring("Running Suite: More_ginkgo_tests Suite"))
				Ω(output).Should(ContainSubstring("should proxy strings"))
				Ω(output).Should(ContainSubstring("should always pass"))
			})

			It("should emit output from Bys", func() {
				session := startGinkgo(pathToTest, "--noColor", "-v", pathToTest)
				Eventually(session).Should(gexec.Exit(0))
				output := session.Out.Contents()

				Ω(output).Should(ContainSubstring("emitting one By"))
				Ω(output).Should(ContainSubstring("emitting another By"))
			})
		})
	})
})
