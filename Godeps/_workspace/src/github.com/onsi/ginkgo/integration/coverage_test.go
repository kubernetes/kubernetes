package integration_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	"os"
	"os/exec"
)

var _ = Describe("Coverage Specs", func() {
	AfterEach(func() {
		os.RemoveAll("./_fixtures/coverage_fixture/coverage_fixture.coverprofile")
	})

	It("runs coverage analysis in series and in parallel", func() {
		session := startGinkgo("./_fixtures/coverage_fixture", "-cover")
		Eventually(session).Should(gexec.Exit(0))
		output := session.Out.Contents()
		Ω(output).Should(ContainSubstring("coverage: 80.0% of statements"))

		serialCoverProfileOutput, err := exec.Command("go", "tool", "cover", "-func=./_fixtures/coverage_fixture/coverage_fixture.coverprofile").CombinedOutput()
		Ω(err).ShouldNot(HaveOccurred())

		os.RemoveAll("./_fixtures/coverage_fixture/coverage_fixture.coverprofile")

		Eventually(startGinkgo("./_fixtures/coverage_fixture", "-cover", "-nodes=4")).Should(gexec.Exit(0))

		parallelCoverProfileOutput, err := exec.Command("go", "tool", "cover", "-func=./_fixtures/coverage_fixture/coverage_fixture.coverprofile").CombinedOutput()
		Ω(err).ShouldNot(HaveOccurred())

		Ω(parallelCoverProfileOutput).Should(Equal(serialCoverProfileOutput))
	})
})
