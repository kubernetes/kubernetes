package integration_test

import (
	"os"
	"os/exec"
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gbytes"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("ginkgo build", func() {
	var pathToTest string

	BeforeEach(func() {
		pathToTest = tmpPath("passing_ginkgo_tests")
		copyIn("passing_ginkgo_tests", pathToTest)
		session := startGinkgo(pathToTest, "build")
		Eventually(session).Should(gexec.Exit(0))
		output := string(session.Out.Contents())
		Ω(output).Should(ContainSubstring("Compiling passing_ginkgo_tests"))
		Ω(output).Should(ContainSubstring("compiled passing_ginkgo_tests.test"))
	})

	It("should build a test binary", func() {
		_, err := os.Stat(filepath.Join(pathToTest, "passing_ginkgo_tests.test"))
		Ω(err).ShouldNot(HaveOccurred())
	})

	It("should be possible to run the test binary directly", func() {
		cmd := exec.Command("./passing_ginkgo_tests.test")
		cmd.Dir = pathToTest
		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Ω(err).ShouldNot(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))
		Ω(session).Should(gbytes.Say("Running Suite: Passing_ginkgo_tests Suite"))
	})

	It("should be possible to run the test binary via ginkgo", func() {
		session := startGinkgo(pathToTest, "./passing_ginkgo_tests.test")
		Eventually(session).Should(gexec.Exit(0))
		Ω(session).Should(gbytes.Say("Running Suite: Passing_ginkgo_tests Suite"))
	})

	It("should be possible to run the test binary in parallel", func() {
		session := startGinkgo(pathToTest, "--nodes=4", "--noColor", "./passing_ginkgo_tests.test")
		Eventually(session).Should(gexec.Exit(0))
		Ω(session).Should(gbytes.Say("Running Suite: Passing_ginkgo_tests Suite"))
		Ω(session).Should(gbytes.Say("Running in parallel across 4 nodes"))
	})
})
