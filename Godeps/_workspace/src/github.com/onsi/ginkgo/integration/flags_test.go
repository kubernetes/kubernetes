package integration_test

import (
	"os"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Flags Specs", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("flags")
		copyIn("flags_tests", pathToTest)
	})

	getRandomOrders := func(output string) []int {
		return []int{strings.Index(output, "RANDOM_A"), strings.Index(output, "RANDOM_B"), strings.Index(output, "RANDOM_C")}
	}

	It("normally passes, runs measurements, prints out noisy pendings, does not randomize tests, and honors the programmatic focus", func() {
		session := startGinkgo(pathToTest, "--noColor")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("Ran 3 samples:"), "has a measurement")
		Ω(output).Should(ContainSubstring("10 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("1 Pending"))
		Ω(output).Should(ContainSubstring("2 Skipped"))
		Ω(output).Should(ContainSubstring("[PENDING]"))
		Ω(output).Should(ContainSubstring("marshmallow"))
		Ω(output).Should(ContainSubstring("chocolate"))
		Ω(output).Should(ContainSubstring("CUSTOM_FLAG: default"))
		Ω(output).Should(ContainSubstring("Detected Programmatic Focus - setting exit status to %d", types.GINKGO_FOCUS_EXIT_CODE))
		Ω(output).ShouldNot(ContainSubstring("smores"))
		Ω(output).ShouldNot(ContainSubstring("SLOW TEST"))
		Ω(output).ShouldNot(ContainSubstring("should honor -slowSpecThreshold"))

		orders := getRandomOrders(output)
		Ω(orders[0]).Should(BeNumerically("<", orders[1]))
		Ω(orders[1]).Should(BeNumerically("<", orders[2]))
	})

	It("should run a coverprofile when passed -cover", func() {
		session := startGinkgo(pathToTest, "--noColor", "--cover", "--focus=the focused set")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		_, err := os.Stat(filepath.Join(pathToTest, "flags.coverprofile"))
		Ω(err).ShouldNot(HaveOccurred())
		Ω(output).Should(ContainSubstring("coverage: "))
	})

	It("should fail when there are pending tests and it is passed --failOnPending", func() {
		session := startGinkgo(pathToTest, "--noColor", "--failOnPending")
		Eventually(session).Should(gexec.Exit(1))
	})

	It("should not print out pendings when --noisyPendings=false", func() {
		session := startGinkgo(pathToTest, "--noColor", "--noisyPendings=false")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).ShouldNot(ContainSubstring("[PENDING]"))
		Ω(output).Should(ContainSubstring("1 Pending"))
	})

	It("should override the programmatic focus when told to focus", func() {
		session := startGinkgo(pathToTest, "--noColor", "--focus=smores")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("marshmallow"))
		Ω(output).Should(ContainSubstring("chocolate"))
		Ω(output).Should(ContainSubstring("smores"))
		Ω(output).Should(ContainSubstring("3 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("0 Pending"))
		Ω(output).Should(ContainSubstring("10 Skipped"))
	})

	It("should override the programmatic focus when told to skip", func() {
		session := startGinkgo(pathToTest, "--noColor", "--skip=marshmallow|failing")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).ShouldNot(ContainSubstring("marshmallow"))
		Ω(output).Should(ContainSubstring("chocolate"))
		Ω(output).Should(ContainSubstring("smores"))
		Ω(output).Should(ContainSubstring("10 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
		Ω(output).Should(ContainSubstring("1 Pending"))
		Ω(output).Should(ContainSubstring("2 Skipped"))
	})

	It("should run the race detector when told to", func() {
		session := startGinkgo(pathToTest, "--noColor", "--race")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("WARNING: DATA RACE"))
	})

	It("should randomize tests when told to", func() {
		session := startGinkgo(pathToTest, "--noColor", "--randomizeAllSpecs", "--seed=21")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		orders := getRandomOrders(output)
		Ω(orders[0]).ShouldNot(BeNumerically("<", orders[1]))
	})

	It("should skip measurements when told to", func() {
		session := startGinkgo(pathToTest, "--skipMeasurements")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).ShouldNot(ContainSubstring("Ran 3 samples:"), "has a measurement")
		Ω(output).Should(ContainSubstring("3 Skipped"))
	})

	It("should watch for slow specs", func() {
		session := startGinkgo(pathToTest, "--slowSpecThreshold=0.05")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("SLOW TEST"))
		Ω(output).Should(ContainSubstring("should honor -slowSpecThreshold"))
	})

	It("should pass additional arguments in", func() {
		session := startGinkgo(pathToTest, "--", "--customFlag=madagascar")
		Eventually(session).Should(gexec.Exit(types.GINKGO_FOCUS_EXIT_CODE))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("CUSTOM_FLAG: madagascar"))
	})

	It("should print out full stack traces for failures when told to", func() {
		session := startGinkgo(pathToTest, "--focus=a failing test", "--trace")
		Eventually(session).Should(gexec.Exit(1))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("Full Stack Trace"))
	})

	It("should fail fast when told to", func() {
		pathToTest = tmpPath("fail")
		copyIn("fail_fixture", pathToTest)
		session := startGinkgo(pathToTest, "--failFast")
		Eventually(session).Should(gexec.Exit(1))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("1 Failed"))
		Ω(output).Should(ContainSubstring("15 Skipped"))
	})

	It("should perform a dry run when told to", func() {
		pathToTest = tmpPath("fail")
		copyIn("fail_fixture", pathToTest)
		session := startGinkgo(pathToTest, "--dryRun", "-v")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())

		Ω(output).Should(ContainSubstring("synchronous failures"))
		Ω(output).Should(ContainSubstring("16 Specs"))
		Ω(output).Should(ContainSubstring("0 Passed"))
		Ω(output).Should(ContainSubstring("0 Failed"))
	})
})
