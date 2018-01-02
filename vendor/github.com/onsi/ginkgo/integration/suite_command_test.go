package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Suite Command Specs", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("suite_command")
		copyIn("suite_command_tests", pathToTest)
	})

	It("Runs command after suite echoing out suite data, properly reporting suite name and passing status in successful command output", func() {
		command := "-afterSuiteHook=echo THIS IS A (ginkgo-suite-passed) TEST OF THE (ginkgo-suite-name) SYSTEM, THIS IS ONLY A TEST"
		expected := "THIS IS A [PASS] TEST OF THE suite_command SYSTEM, THIS IS ONLY A TEST"
		session := startGinkgo(pathToTest, command)
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("1 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("1 Pending"))
		Ω(output).Should(ContainSubstring("0 Skipped"))
		Ω(output).Should(ContainSubstring("Test Suite Passed"))
		Ω(output).Should(ContainSubstring("Post-suite command succeeded:"))
		Ω(output).Should(ContainSubstring(expected))
	})

	It("Runs command after suite reporting that command failed", func() {
		command := "-afterSuiteHook=exit 1"
		session := startGinkgo(pathToTest, command)
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("1 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("1 Pending"))
		Ω(output).Should(ContainSubstring("0 Skipped"))
		Ω(output).Should(ContainSubstring("Test Suite Passed"))
		Ω(output).Should(ContainSubstring("Post-suite command failed:"))
	})

	It("Runs command after suite echoing out suite data, properly reporting suite name and failing status in successful command output", func() {
		command := "-afterSuiteHook=echo THIS IS A (ginkgo-suite-passed) TEST OF THE (ginkgo-suite-name) SYSTEM, THIS IS ONLY A TEST"
		expected := "THIS IS A [FAIL] TEST OF THE suite_command SYSTEM, THIS IS ONLY A TEST"
		session := startGinkgo(pathToTest, "-failOnPending=true", command)
		Eventually(session).Should(gexec.Exit(1))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("1 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("1 Pending"))
		Ω(output).Should(ContainSubstring("0 Skipped"))
		Ω(output).Should(ContainSubstring("Test Suite Failed"))
		Ω(output).Should(ContainSubstring("Post-suite command succeeded:"))
		Ω(output).Should(ContainSubstring(expected))
	})

})
