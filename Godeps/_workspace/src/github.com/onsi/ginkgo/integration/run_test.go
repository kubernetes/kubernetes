package integration_test

import (
	"runtime"
	"strings"

	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gbytes"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Running Specs", func() {
	var pathToTest string

	Context("when pointed at the current directory", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
		})

		It("should run the tests in the working directory", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("••••"))
			Ω(output).Should(ContainSubstring("SUCCESS! -- 4 Passed"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})
	})

	Context("when passed an explicit package to run", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
		})

		It("should run the ginkgo style tests", func() {
			session := startGinkgo(tmpDir, "--noColor", pathToTest)
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("••••"))
			Ω(output).Should(ContainSubstring("SUCCESS! -- 4 Passed"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})
	})

	Context("when passed a number of packages to run", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			otherPathToTest := tmpPath("other")
			copyIn("passing_ginkgo_tests", pathToTest)
			copyIn("more_ginkgo_tests", otherPathToTest)
		})

		It("should run the ginkgo style tests", func() {
			session := startGinkgo(tmpDir, "--noColor", "--succinct=false", "ginkgo", "./other")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("Running Suite: More_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})
	})

	Context("when passed a number of packages to run, some of which have focused tests", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			otherPathToTest := tmpPath("other")
			focusedPathToTest := tmpPath("focused")
			copyIn("passing_ginkgo_tests", pathToTest)
			copyIn("more_ginkgo_tests", otherPathToTest)
			copyIn("focused_fixture", focusedPathToTest)
		})

		It("should exit with a status code of 2 and explain why", func() {
			session := startGinkgo(tmpDir, "--noColor", "--succinct=false", "-r")
			Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Running Suite: Passing_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("Running Suite: More_ginkgo_tests Suite"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
			Ω(output).Should(ContainSubstring("Detected Programmatic Focus - setting exit status to %d", types.GINKGO_FOCUS_EXIT_CODE))
		})
	})

	Context("when told to skipPackages", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			otherPathToTest := tmpPath("other")
			focusedPathToTest := tmpPath("focused")
			copyIn("passing_ginkgo_tests", pathToTest)
			copyIn("more_ginkgo_tests", otherPathToTest)
			copyIn("focused_fixture", focusedPathToTest)
		})

		It("should skip packages that match the list", func() {
			session := startGinkgo(tmpDir, "--noColor", "--skipPackage=other,focused", "-r")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Passing_ginkgo_tests Suite"))
			Ω(output).ShouldNot(ContainSubstring("More_ginkgo_tests Suite"))
			Ω(output).ShouldNot(ContainSubstring("Focused_fixture Suite"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})

		Context("when all packages are skipped", func() {
			It("should not run anything, but still exit 0", func() {
				session := startGinkgo(tmpDir, "--noColor", "--skipPackage=other,focused,ginkgo", "-r")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				Ω(output).Should(ContainSubstring("All tests skipped!"))
				Ω(output).ShouldNot(ContainSubstring("Passing_ginkgo_tests Suite"))
				Ω(output).ShouldNot(ContainSubstring("More_ginkgo_tests Suite"))
				Ω(output).ShouldNot(ContainSubstring("Focused_fixture Suite"))
				Ω(output).ShouldNot(ContainSubstring("Test Suite Passed"))
			})
		})
	})

	Context("when there are no tests to run", func() {
		It("should exit 1", func() {
			session := startGinkgo(tmpDir, "--noColor", "--skipPackage=other,focused", "-r")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Err.Contents())

			Ω(output).Should(ContainSubstring("Found no test suites"))
		})
	})

	Context("when told to randomizeSuites", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			otherPathToTest := tmpPath("other")
			copyIn("passing_ginkgo_tests", pathToTest)
			copyIn("more_ginkgo_tests", otherPathToTest)
		})

		It("should skip packages that match the regexp", func() {
			session := startGinkgo(tmpDir, "--noColor", "--randomizeSuites", "-r", "--seed=2")
			Eventually(session).Should(gexec.Exit(0))

			Ω(session).Should(gbytes.Say("More_ginkgo_tests Suite"))
			Ω(session).Should(gbytes.Say("Passing_ginkgo_tests Suite"))

			session = startGinkgo(tmpDir, "--noColor", "--randomizeSuites", "-r", "--seed=3")
			Eventually(session).Should(gexec.Exit(0))

			Ω(session).Should(gbytes.Say("Passing_ginkgo_tests Suite"))
			Ω(session).Should(gbytes.Say("More_ginkgo_tests Suite"))
		})
	})

	Context("when pointed at a package with xunit style tests", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("xunit")
			copyIn("xunit_tests", pathToTest)
		})

		It("should run the xunit style tests", func() {
			session := startGinkgo(pathToTest)
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("--- PASS: TestAlwaysTrue"))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})
	})

	Context("when pointed at a package with no tests", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("no_tests")
			copyIn("no_tests", pathToTest)
		})

		It("should fail", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(1))

			Ω(session.Err.Contents()).Should(ContainSubstring("Found no test suites"))
		})
	})

	Context("when pointed at a package that fails to compile", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("does_not_compile")
			copyIn("does_not_compile", pathToTest)
		})

		It("should fail", func() {
			session := startGinkgo(pathToTest, "--noColor")
			Eventually(session).Should(gexec.Exit(1))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring("Failed to compile"))
		})
	})

	Context("when running in parallel", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
		})

		Context("with a specific number of -nodes", func() {
			It("should use the specified number of nodes", func() {
				session := startGinkgo(pathToTest, "--noColor", "-succinct", "-nodes=2")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				Ω(output).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs - 2 nodes •••• SUCCESS! [\d.µs]+`))
				Ω(output).Should(ContainSubstring("Test Suite Passed"))
			})
		})

		Context("with -p", func() {
			It("it should autocompute the number of nodes", func() {
				session := startGinkgo(pathToTest, "--noColor", "-succinct", "-p")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				nodes := runtime.NumCPU()
				if nodes > 4 {
					nodes = nodes - 1
				}
				Ω(output).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs - %d nodes •••• SUCCESS! [\d.µs]+`, nodes))
				Ω(output).Should(ContainSubstring("Test Suite Passed"))
			})
		})
	})

	Context("when streaming in parallel", func() {
		BeforeEach(func() {
			pathToTest = tmpPath("ginkgo")
			copyIn("passing_ginkgo_tests", pathToTest)
		})

		It("should print output in realtime", func() {
			session := startGinkgo(pathToTest, "--noColor", "-stream", "-nodes=2")
			Eventually(session).Should(gexec.Exit(0))
			output := string(session.Out.Contents())

			Ω(output).Should(ContainSubstring(`[1] Parallel test node 1/2.`))
			Ω(output).Should(ContainSubstring(`[2] Parallel test node 2/2.`))
			Ω(output).Should(ContainSubstring(`[1] SUCCESS!`))
			Ω(output).Should(ContainSubstring(`[2] SUCCESS!`))
			Ω(output).Should(ContainSubstring("Test Suite Passed"))
		})
	})

	Context("when running recursively", func() {
		BeforeEach(func() {
			passingTest := tmpPath("A")
			otherPassingTest := tmpPath("E")
			copyIn("passing_ginkgo_tests", passingTest)
			copyIn("more_ginkgo_tests", otherPassingTest)
		})

		Context("when all the tests pass", func() {
			It("should run all the tests (in succinct mode) and succeed", func() {
				session := startGinkgo(tmpDir, "--noColor", "-r")
				Eventually(session).Should(gexec.Exit(0))
				output := string(session.Out.Contents())

				outputLines := strings.Split(output, "\n")
				Ω(outputLines[0]).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs •••• SUCCESS! [\d.µs]+ PASS`))
				Ω(outputLines[1]).Should(MatchRegexp(`\[\d+\] More_ginkgo_tests Suite - 2/2 specs •• SUCCESS! [\d.µs]+ PASS`))
				Ω(output).Should(ContainSubstring("Test Suite Passed"))
			})
		})

		Context("when one of the packages has a failing tests", func() {
			BeforeEach(func() {
				failingTest := tmpPath("C")
				copyIn("failing_ginkgo_tests", failingTest)
			})

			It("should fail and stop running tests", func() {
				session := startGinkgo(tmpDir, "--noColor", "-r")
				Eventually(session).Should(gexec.Exit(1))
				output := string(session.Out.Contents())

				outputLines := strings.Split(output, "\n")
				Ω(outputLines[0]).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs •••• SUCCESS! [\d.µs]+ PASS`))
				Ω(outputLines[1]).Should(MatchRegexp(`\[\d+\] Failing_ginkgo_tests Suite - 2/2 specs`))
				Ω(output).Should(ContainSubstring("• Failure"))
				Ω(output).ShouldNot(ContainSubstring("More_ginkgo_tests Suite"))
				Ω(output).Should(ContainSubstring("Test Suite Failed"))

				Ω(output).Should(ContainSubstring("Summarizing 1 Failure:"))
				Ω(output).Should(ContainSubstring("[Fail] FailingGinkgoTests [It] should fail"))
			})
		})

		Context("when one of the packages fails to compile", func() {
			BeforeEach(func() {
				doesNotCompileTest := tmpPath("C")
				copyIn("does_not_compile", doesNotCompileTest)
			})

			It("should fail and stop running tests", func() {
				session := startGinkgo(tmpDir, "--noColor", "-r")
				Eventually(session).Should(gexec.Exit(1))
				output := string(session.Out.Contents())

				outputLines := strings.Split(output, "\n")
				Ω(outputLines[0]).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs •••• SUCCESS! [\d.µs]+ PASS`))
				Ω(outputLines[1]).Should(ContainSubstring("Failed to compile C:"))
				Ω(output).ShouldNot(ContainSubstring("More_ginkgo_tests Suite"))
				Ω(output).Should(ContainSubstring("Test Suite Failed"))
			})
		})

		Context("when either is the case, but the keepGoing flag is set", func() {
			BeforeEach(func() {
				doesNotCompileTest := tmpPath("B")
				copyIn("does_not_compile", doesNotCompileTest)

				failingTest := tmpPath("C")
				copyIn("failing_ginkgo_tests", failingTest)
			})

			It("should soldier on", func() {
				session := startGinkgo(tmpDir, "--noColor", "-r", "-keepGoing")
				Eventually(session).Should(gexec.Exit(1))
				output := string(session.Out.Contents())

				outputLines := strings.Split(output, "\n")
				Ω(outputLines[0]).Should(MatchRegexp(`\[\d+\] Passing_ginkgo_tests Suite - 4/4 specs •••• SUCCESS! [\d.µs]+ PASS`))
				Ω(outputLines[1]).Should(ContainSubstring("Failed to compile B:"))
				Ω(output).Should(MatchRegexp(`\[\d+\] Failing_ginkgo_tests Suite - 2/2 specs`))
				Ω(output).Should(ContainSubstring("• Failure"))
				Ω(output).Should(MatchRegexp(`\[\d+\] More_ginkgo_tests Suite - 2/2 specs •• SUCCESS! [\d.µs]+ PASS`))
				Ω(output).Should(ContainSubstring("Test Suite Failed"))
			})
		})
	})

	Context("when told to keep going --untilItFails", func() {
		BeforeEach(func() {
			copyIn("eventually_failing", tmpDir)
		})

		It("should keep rerunning the tests, until a failure occurs", func() {
			session := startGinkgo(tmpDir, "--untilItFails", "--noColor")
			Eventually(session).Should(gexec.Exit(1))
			Ω(session).Should(gbytes.Say("This was attempt #1"))
			Ω(session).Should(gbytes.Say("This was attempt #2"))
			Ω(session).Should(gbytes.Say("Tests failed on attempt #3"))

			//it should change the random seed between each test
			lines := strings.Split(string(session.Out.Contents()), "\n")
			randomSeeds := []string{}
			for _, line := range lines {
				if strings.Contains(line, "Random Seed:") {
					randomSeeds = append(randomSeeds, strings.Split(line, ": ")[1])
				}
			}
			Ω(randomSeeds[0]).ShouldNot(Equal(randomSeeds[1]))
			Ω(randomSeeds[1]).ShouldNot(Equal(randomSeeds[2]))
			Ω(randomSeeds[0]).ShouldNot(Equal(randomSeeds[2]))
		})
	})
})
