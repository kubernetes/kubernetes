package gexec_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe(".Build", func() {
	var packagePath = "./_fixture/firefly"

	Context("when there have been previous calls to Build", func() {
		BeforeEach(func() {
			_, err := gexec.Build(packagePath)
			Ω(err).ShouldNot(HaveOccurred())
		})

		It("compiles the specified package", func() {
			compiledPath, err := gexec.Build(packagePath)
			Ω(err).ShouldNot(HaveOccurred())
			Ω(compiledPath).Should(BeAnExistingFile())
		})

		Context("and CleanupBuildArtifacts has been called", func() {
			BeforeEach(func() {
				gexec.CleanupBuildArtifacts()
			})

			It("compiles the specified package", func() {
				var err error
				fireflyPath, err = gexec.Build(packagePath)
				Ω(err).ShouldNot(HaveOccurred())
				Ω(fireflyPath).Should(BeAnExistingFile())
			})
		})
	})
})
