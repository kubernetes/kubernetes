package integration_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ginkgo convert", func() {
	var tmpDir string

	readConvertedFileNamed := func(pathComponents ...string) string {
		pathToFile := filepath.Join(tmpDir, "convert_fixtures", filepath.Join(pathComponents...))
		bytes, err := ioutil.ReadFile(pathToFile)
		ExpectWithOffset(1, err).NotTo(HaveOccurred())

		return string(bytes)
	}

	readGoldMasterNamed := func(filename string) string {
		bytes, err := ioutil.ReadFile(filepath.Join("_fixtures", "convert_goldmasters", filename))
		Ω(err).ShouldNot(HaveOccurred())

		return string(bytes)
	}

	BeforeEach(func() {
		var err error

		tmpDir, err = ioutil.TempDir("", "ginkgo-convert")
		Ω(err).ShouldNot(HaveOccurred())

		err = exec.Command("cp", "-r", filepath.Join("_fixtures", "convert_fixtures"), tmpDir).Run()
		Ω(err).ShouldNot(HaveOccurred())
	})

	JustBeforeEach(func() {
		cwd, err := os.Getwd()
		Ω(err).ShouldNot(HaveOccurred())

		relPath, err := filepath.Rel(cwd, filepath.Join(tmpDir, "convert_fixtures"))
		Ω(err).ShouldNot(HaveOccurred())

		cmd := exec.Command(pathToGinkgo, "convert", relPath)
		cmd.Env = os.Environ()
		for i, env := range cmd.Env {
			if strings.HasPrefix(env, "PATH") {
				cmd.Env[i] = cmd.Env[i] + ":" + filepath.Dir(pathToGinkgo)
				break
			}
		}
		err = cmd.Run()
		Ω(err).ShouldNot(HaveOccurred())
	})

	AfterEach(func() {
		err := os.RemoveAll(tmpDir)
		Ω(err).ShouldNot(HaveOccurred())
	})

	It("rewrites xunit tests as ginkgo tests", func() {
		convertedFile := readConvertedFileNamed("xunit_test.go")
		goldMaster := readGoldMasterNamed("xunit_test.go")
		Ω(convertedFile).Should(Equal(goldMaster))
	})

	It("rewrites all usages of *testing.T as mr.T()", func() {
		convertedFile := readConvertedFileNamed("extra_functions_test.go")
		goldMaster := readGoldMasterNamed("extra_functions_test.go")
		Ω(convertedFile).Should(Equal(goldMaster))
	})

	It("rewrites tests in the package dir that belong to other packages", func() {
		convertedFile := readConvertedFileNamed("outside_package_test.go")
		goldMaster := readGoldMasterNamed("outside_package_test.go")
		Ω(convertedFile).Should(Equal(goldMaster))
	})

	It("rewrites tests in nested packages", func() {
		convertedFile := readConvertedFileNamed("nested", "nested_test.go")
		goldMaster := readGoldMasterNamed("nested_test.go")
		Ω(convertedFile).Should(Equal(goldMaster))
	})

	Context("ginkgo test suite files", func() {
		It("creates a ginkgo test suite file for the package you specified", func() {
			testsuite := readConvertedFileNamed("convert_fixtures_suite_test.go")
			goldMaster := readGoldMasterNamed("suite_test.go")
			Ω(testsuite).Should(Equal(goldMaster))
		})

		It("converts go tests in deeply nested packages (some may not contain go files)", func() {
			testsuite := readConvertedFileNamed("nested_without_gofiles", "subpackage", "nested_subpackage_test.go")
			goldMaster := readGoldMasterNamed("nested_subpackage_test.go")
			Ω(testsuite).Should(Equal(goldMaster))
		})

		It("creates ginkgo test suites for all nested packages", func() {
			testsuite := readConvertedFileNamed("nested", "nested_suite_test.go")
			goldMaster := readGoldMasterNamed("nested_suite_test.go")
			Ω(testsuite).Should(Equal(goldMaster))
		})
	})

	Context("with an existing test suite file", func() {
		BeforeEach(func() {
			goldMaster := readGoldMasterNamed("fixtures_suite_test.go")
			err := ioutil.WriteFile(filepath.Join(tmpDir, "convert_fixtures", "tmp_suite_test.go"), []byte(goldMaster), 0600)
			Ω(err).ShouldNot(HaveOccurred())
		})

		It("gracefully handles existing test suite files", func() {
			//nothing should have gone wrong!
		})
	})
})
