package testsuite_test

import (
	"io/ioutil"
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/ginkgo/testsuite"
	. "github.com/onsi/gomega"
)

var _ = Describe("TestSuite", func() {
	var tmpDir string
	var relTmpDir string

	writeFile := func(folder string, filename string, content string, mode os.FileMode) {
		path := filepath.Join(tmpDir, folder)
		err := os.MkdirAll(path, 0700)
		Ω(err).ShouldNot(HaveOccurred())

		path = filepath.Join(path, filename)
		ioutil.WriteFile(path, []byte(content), mode)
	}

	BeforeEach(func() {
		var err error
		tmpDir, err = ioutil.TempDir("/tmp", "ginkgo")
		Ω(err).ShouldNot(HaveOccurred())

		cwd, err := os.Getwd()
		Ω(err).ShouldNot(HaveOccurred())
		relTmpDir, err = filepath.Rel(cwd, tmpDir)
		relTmpDir = "./" + relTmpDir
		Ω(err).ShouldNot(HaveOccurred())

		//go files in the root directory (no tests)
		writeFile("/", "main.go", "package main", 0666)

		//non-go files in a nested directory
		writeFile("/redherring", "big_test.jpg", "package ginkgo", 0666)

		//non-ginkgo tests in a nested directory
		writeFile("/professorplum", "professorplum_test.go", `import "testing"`, 0666)

		//ginkgo tests in a nested directory
		writeFile("/colonelmustard", "colonelmustard_test.go", `import "github.com/onsi/ginkgo"`, 0666)

		//ginkgo tests in a deeply nested directory
		writeFile("/colonelmustard/library", "library_test.go", `import "github.com/onsi/ginkgo"`, 0666)

		//a precompiled ginkgo test
		writeFile("/precompiled-dir", "precompiled.test", `fake-binary-file`, 0777)
		writeFile("/precompiled-dir", "some-other-binary", `fake-binary-file`, 0777)
		writeFile("/precompiled-dir", "nonexecutable.test", `fake-binary-file`, 0666)
	})

	AfterEach(func() {
		os.RemoveAll(tmpDir)
	})

	Describe("Finding precompiled test suites", func() {
		Context("if pointed at an executable file that ends with .test", func() {
			It("should return a precompiled test suite", func() {
				suite, err := PrecompiledTestSuite(filepath.Join(tmpDir, "precompiled-dir", "precompiled.test"))
				Ω(err).ShouldNot(HaveOccurred())
				Ω(suite).Should(Equal(TestSuite{
					Path:        relTmpDir + "/precompiled-dir",
					PackageName: "precompiled",
					IsGinkgo:    true,
					Precompiled: true,
				}))
			})
		})

		Context("if pointed at a directory", func() {
			It("should error", func() {
				suite, err := PrecompiledTestSuite(filepath.Join(tmpDir, "precompiled-dir"))
				Ω(suite).Should(BeZero())
				Ω(err).Should(HaveOccurred())
			})
		})

		Context("if pointed at an executable that doesn't have .test", func() {
			It("should error", func() {
				suite, err := PrecompiledTestSuite(filepath.Join(tmpDir, "precompiled-dir", "some-other-binary"))
				Ω(suite).Should(BeZero())
				Ω(err).Should(HaveOccurred())
			})
		})

		Context("if pointed at a .test that isn't executable", func() {
			It("should error", func() {
				suite, err := PrecompiledTestSuite(filepath.Join(tmpDir, "precompiled-dir", "nonexecutable.test"))
				Ω(suite).Should(BeZero())
				Ω(err).Should(HaveOccurred())
			})
		})

		Context("if pointed at a nonexisting file", func() {
			It("should error", func() {
				suite, err := PrecompiledTestSuite(filepath.Join(tmpDir, "precompiled-dir", "nope-nothing-to-see-here"))
				Ω(suite).Should(BeZero())
				Ω(err).Should(HaveOccurred())
			})
		})
	})

	Describe("scanning for suites in a directory", func() {
		Context("when there are no tests in the specified directory", func() {
			It("should come up empty", func() {
				suites := SuitesInDir(tmpDir, false)
				Ω(suites).Should(BeEmpty())
			})
		})

		Context("when there are ginkgo tests in the specified directory", func() {
			It("should return an appropriately configured suite", func() {
				suites := SuitesInDir(filepath.Join(tmpDir, "colonelmustard"), false)
				Ω(suites).Should(HaveLen(1))

				Ω(suites[0].Path).Should(Equal(relTmpDir + "/colonelmustard"))
				Ω(suites[0].PackageName).Should(Equal("colonelmustard"))
				Ω(suites[0].IsGinkgo).Should(BeTrue())
				Ω(suites[0].Precompiled).Should(BeFalse())
			})
		})

		Context("when there are non-ginkgo tests in the specified directory", func() {
			It("should return an appropriately configured suite", func() {
				suites := SuitesInDir(filepath.Join(tmpDir, "professorplum"), false)
				Ω(suites).Should(HaveLen(1))

				Ω(suites[0].Path).Should(Equal(relTmpDir + "/professorplum"))
				Ω(suites[0].PackageName).Should(Equal("professorplum"))
				Ω(suites[0].IsGinkgo).Should(BeFalse())
				Ω(suites[0].Precompiled).Should(BeFalse())
			})
		})

		Context("when recursively scanning", func() {
			It("should return suites for corresponding test suites, only", func() {
				suites := SuitesInDir(tmpDir, true)
				Ω(suites).Should(HaveLen(3))

				Ω(suites).Should(ContainElement(TestSuite{
					Path:        relTmpDir + "/colonelmustard",
					PackageName: "colonelmustard",
					IsGinkgo:    true,
					Precompiled: false,
				}))
				Ω(suites).Should(ContainElement(TestSuite{
					Path:        relTmpDir + "/professorplum",
					PackageName: "professorplum",
					IsGinkgo:    false,
					Precompiled: false,
				}))
				Ω(suites).Should(ContainElement(TestSuite{
					Path:        relTmpDir + "/colonelmustard/library",
					PackageName: "library",
					IsGinkgo:    true,
					Precompiled: false,
				}))
			})
		})
	})
})
