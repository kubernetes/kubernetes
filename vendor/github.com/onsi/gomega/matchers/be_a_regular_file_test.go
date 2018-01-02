package matchers_test

import (
	"io/ioutil"
	"os"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeARegularFileMatcher", func() {
	Context("when passed a string", func() {
		It("should do the right thing", func() {
			Ω("/dne/test").ShouldNot(BeARegularFile())

			tmpFile, err := ioutil.TempFile("", "gomega-test-tempfile")
			Ω(err).ShouldNot(HaveOccurred())
			defer os.Remove(tmpFile.Name())
			Ω(tmpFile.Name()).Should(BeARegularFile())

			tmpDir, err := ioutil.TempDir("", "gomega-test-tempdir")
			Ω(err).ShouldNot(HaveOccurred())
			defer os.Remove(tmpDir)
			Ω(tmpDir).ShouldNot(BeARegularFile())
		})
	})

	Context("when passed something else", func() {
		It("should error", func() {
			success, err := (&BeARegularFileMatcher{}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeARegularFileMatcher{}).Match(true)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
