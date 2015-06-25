package integration_test

import (
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("SuiteSetup", func() {
	var pathToTest string

	Context("when the BeforeSuite and AfterSuite pass", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("suite_setup")
			copyIn("passing_suite_setup", pathToTest)
		})

		It("should run the BeforeSuite once, then run all the tests", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(1))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(1))
		})

		It("should run the BeforeSuite once per parallel node, then run all the tests", func() {
			session := startGinkgo(pathToTest, "--noColor", "--nodes=2")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(2))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(2))
		})
	})

	Context("when the BeforeSuite fails", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("suite_setup")
			copyIn("failing_before_suite", pathToTest)
		})

		It("should run the BeforeSuite once, none of the tests, but it should run the AfterSuite", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(1))
			Ω(strings.Count(output, "Test Panicked")).Should(Equal(1))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(1))
			Ω(output).ShouldNot(ContainSubstring("NEVER SEE THIS"))
		})

		It("should run the BeforeSuite once per parallel node, none of the tests, but it should run the AfterSuite for each node", func() {
			session := startGinkgo(pathToTest, "--noColor", "--nodes=2")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(2))
			Ω(strings.Count(output, "Test Panicked")).Should(Equal(2))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(2))
			Ω(output).ShouldNot(ContainSubstring("NEVER SEE THIS"))
		})
	})

	Context("when the AfterSuite fails", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("suite_setup")
			copyIn("failing_after_suite", pathToTest)
		})

		It("should run the BeforeSuite once, none of the tests, but it should run the AfterSuite", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(1))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(1))
			Ω(strings.Count(output, "Test Panicked")).Should(Equal(1))
			Ω(strings.Count(output, "A TEST")).Should(Equal(2))
		})

		It("should run the BeforeSuite once per parallel node, none of the tests, but it should run the AfterSuite for each node", func() {
			session := startGinkgo(pathToTest, "--noColor", "--nodes=2")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(strings.Count(output, "BEFORE SUITE")).Should(Equal(2))
			Ω(strings.Count(output, "AFTER SUITE")).Should(Equal(2))
			Ω(strings.Count(output, "Test Panicked")).Should(Equal(2))
			Ω(strings.Count(output, "A TEST")).Should(Equal(2))
		})
	})

	Context("With passing synchronized before and after suites", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("suite_setup")
			copyIn("synchronized_setup_tests", pathToTest)
		})

		Context("when run with one node", func() {
			It("should do all the work on that one node", func() {
				session := startGinkgo(pathToTest, "--noColor")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				Ω(output).Should(ContainSubstring("BEFORE_A_1\nBEFORE_B_1: DATA"))
				Ω(output).Should(ContainSubstring("AFTER_A_1\nAFTER_B_1"))
			})
		})

		Context("when run across multiple nodes", func() {
			It("should run the first BeforeSuite function (BEFORE_A) on node 1, the second (BEFORE_B) on all the nodes, the first AfterSuite (AFTER_A) on all the nodes, and then the second (AFTER_B) on Node 1 *after* everything else is finished", func() {
				session := startGinkgo(pathToTest, "--noColor", "--nodes=3")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				Ω(output).Should(ContainSubstring("BEFORE_A_1"))
				Ω(output).Should(ContainSubstring("BEFORE_B_1: DATA"))
				Ω(output).Should(ContainSubstring("BEFORE_B_2: DATA"))
				Ω(output).Should(ContainSubstring("BEFORE_B_3: DATA"))

				Ω(output).ShouldNot(ContainSubstring("BEFORE_A_2"))
				Ω(output).ShouldNot(ContainSubstring("BEFORE_A_3"))

				Ω(output).Should(ContainSubstring("AFTER_A_1"))
				Ω(output).Should(ContainSubstring("AFTER_A_2"))
				Ω(output).Should(ContainSubstring("AFTER_A_3"))
				Ω(output).Should(ContainSubstring("AFTER_B_1"))

				Ω(output).ShouldNot(ContainSubstring("AFTER_B_2"))
				Ω(output).ShouldNot(ContainSubstring("AFTER_B_3"))
			})
		})

		Context("when streaming across multiple nodes", func() {
			It("should run the first BeforeSuite function (BEFORE_A) on node 1, the second (BEFORE_B) on all the nodes, the first AfterSuite (AFTER_A) on all the nodes, and then the second (AFTER_B) on Node 1 *after* everything else is finished", func() {
				session := startGinkgo(pathToTest, "--noColor", "--nodes=3", "--stream")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				Ω(output).Should(ContainSubstring("[1] BEFORE_A_1"))
				Ω(output).Should(ContainSubstring("[1] BEFORE_B_1: DATA"))
				Ω(output).Should(ContainSubstring("[2] BEFORE_B_2: DATA"))
				Ω(output).Should(ContainSubstring("[3] BEFORE_B_3: DATA"))

				Ω(output).ShouldNot(ContainSubstring("BEFORE_A_2"))
				Ω(output).ShouldNot(ContainSubstring("BEFORE_A_3"))

				Ω(output).Should(ContainSubstring("[1] AFTER_A_1"))
				Ω(output).Should(ContainSubstring("[2] AFTER_A_2"))
				Ω(output).Should(ContainSubstring("[3] AFTER_A_3"))
				Ω(output).Should(ContainSubstring("[1] AFTER_B_1"))

				Ω(output).ShouldNot(ContainSubstring("AFTER_B_2"))
				Ω(output).ShouldNot(ContainSubstring("AFTER_B_3"))
			})
		})
	})

	Context("With a failing synchronized before suite", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("suite_setup")
			copyIn("exiting_synchronized_setup_tests", pathToTest)
		})

		It("should fail and let the user know that node 1 disappeared prematurely", func() {
			session := startGinkgo(pathToTest, "--noColor", "--nodes=3")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Node 1 disappeared before completing BeforeSuite"))
			Ω(output).Should(ContainSubstring("Ginkgo timed out waiting for all parallel nodes to report back!"))
		})
	})
})
