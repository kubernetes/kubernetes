package integration_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gbytes"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Watch", func() {
	var rootPath string
	var pathA string
	var pathB string
	var pathC string
	var session *gexec.Session

	BeforeEach(func() {
		rootPath = tmpPath("root")
		pathA = filepath.Join(rootPath, "src", "github.com", "onsi", "A")
		pathB = filepath.Join(rootPath, "src", "github.com", "onsi", "B")
		pathC = filepath.Join(rootPath, "src", "github.com", "onsi", "C")

		err := os.MkdirAll(pathA, 0700)
		Ω(err).ShouldNot(HaveOccurred())

		err = os.MkdirAll(pathB, 0700)
		Ω(err).ShouldNot(HaveOccurred())

		err = os.MkdirAll(pathC, 0700)
		Ω(err).ShouldNot(HaveOccurred())

		copyIn(filepath.Join("watch_fixtures", "A"), pathA)
		copyIn(filepath.Join("watch_fixtures", "B"), pathB)
		copyIn(filepath.Join("watch_fixtures", "C"), pathC)
	})

	startGinkgoWithGopath := func(args ...string) *gexec.Session {
		cmd := ginkgoCommand(rootPath, args...)
		cmd.Env = append([]string{"GOPATH=" + rootPath + ":" + os.Getenv("GOPATH")}, os.Environ()...)
		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Ω(err).ShouldNot(HaveOccurred())
		return session
	}

	modifyFile := func(path string) {
		time.Sleep(time.Second)
		content, err := ioutil.ReadFile(path)
		Ω(err).ShouldNot(HaveOccurred())
		content = append(content, []byte("//")...)
		err = ioutil.WriteFile(path, content, 0666)
		Ω(err).ShouldNot(HaveOccurred())
	}

	modifyCode := func(pkgToModify string) {
		modifyFile(filepath.Join(rootPath, "src", "github.com", "onsi", pkgToModify, pkgToModify+".go"))
	}

	modifyTest := func(pkgToModify string) {
		modifyFile(filepath.Join(rootPath, "src", "github.com", "onsi", pkgToModify, pkgToModify+"_test.go"))
	}

	AfterEach(func() {
		if session != nil {
			session.Kill().Wait()
		}
	})

	It("should be set up correctly", func() {
		session = startGinkgoWithGopath("-r")
		Eventually(session).Should(gexec.Exit(0))
		Ω(session.Out.Contents()).Should(ContainSubstring("A Suite"))
		Ω(session.Out.Contents()).Should(ContainSubstring("B Suite"))
		Ω(session.Out.Contents()).Should(ContainSubstring("C Suite"))
		Ω(session.Out.Contents()).Should(ContainSubstring("Ginkgo ran 3 suites"))
	})

	Context("when watching just one test suite", func() {
		It("should immediately run, and should rerun when the test suite changes", func() {
			session = startGinkgoWithGopath("watch", "-succinct", pathA)
			Eventually(session).Should(gbytes.Say("A Suite"))
			modifyCode("A")
			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("A Suite"))
			session.Kill().Wait()
		})
	})

	Context("when watching several test suites", func() {
		It("should not immediately run, but should rerun a test when its code changes", func() {
			session = startGinkgoWithGopath("watch", "-succinct", "-r")
			Eventually(session).Should(gbytes.Say("Identified 3 test suites"))
			Consistently(session).ShouldNot(gbytes.Say("A Suite|B Suite|C Suite"))
			modifyCode("A")
			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("A Suite"))
			Consistently(session).ShouldNot(gbytes.Say("B Suite|C Suite"))
			session.Kill().Wait()
		})
	})

	Describe("watching dependencies", func() {
		Context("with a depth of 2", func() {
			It("should watch down to that depth", func() {
				session = startGinkgoWithGopath("watch", "-succinct", "-r", "-depth=2")
				Eventually(session).Should(gbytes.Say("Identified 3 test suites"))
				Eventually(session).Should(gbytes.Say(`A \[2 dependencies\]`))
				Eventually(session).Should(gbytes.Say(`B \[1 dependency\]`))
				Eventually(session).Should(gbytes.Say(`C \[0 dependencies\]`))

				modifyCode("A")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("A Suite"))
				Consistently(session).ShouldNot(gbytes.Say("B Suite|C Suite"))

				modifyCode("B")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("B Suite"))
				Eventually(session).Should(gbytes.Say("A Suite"))
				Consistently(session).ShouldNot(gbytes.Say("C Suite"))

				modifyCode("C")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("C Suite"))
				Eventually(session).Should(gbytes.Say("B Suite"))
				Eventually(session).Should(gbytes.Say("A Suite"))
			})
		})

		Context("with a depth of 1", func() {
			It("should watch down to that depth", func() {
				session = startGinkgoWithGopath("watch", "-succinct", "-r", "-depth=1")
				Eventually(session).Should(gbytes.Say("Identified 3 test suites"))
				Eventually(session).Should(gbytes.Say(`A \[1 dependency\]`))
				Eventually(session).Should(gbytes.Say(`B \[1 dependency\]`))
				Eventually(session).Should(gbytes.Say(`C \[0 dependencies\]`))

				modifyCode("A")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("A Suite"))
				Consistently(session).ShouldNot(gbytes.Say("B Suite|C Suite"))

				modifyCode("B")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("B Suite"))
				Eventually(session).Should(gbytes.Say("A Suite"))
				Consistently(session).ShouldNot(gbytes.Say("C Suite"))

				modifyCode("C")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("C Suite"))
				Eventually(session).Should(gbytes.Say("B Suite"))
				Consistently(session).ShouldNot(gbytes.Say("A Suite"))
			})
		})

		Context("with a depth of 0", func() {
			It("should not watch any dependencies", func() {
				session = startGinkgoWithGopath("watch", "-succinct", "-r", "-depth=0")
				Eventually(session).Should(gbytes.Say("Identified 3 test suites"))
				Eventually(session).Should(gbytes.Say(`A \[0 dependencies\]`))
				Eventually(session).Should(gbytes.Say(`B \[0 dependencies\]`))
				Eventually(session).Should(gbytes.Say(`C \[0 dependencies\]`))

				modifyCode("A")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("A Suite"))
				Consistently(session).ShouldNot(gbytes.Say("B Suite|C Suite"))

				modifyCode("B")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("B Suite"))
				Consistently(session).ShouldNot(gbytes.Say("A Suite|C Suite"))

				modifyCode("C")
				Eventually(session).Should(gbytes.Say("Detected changes in"))
				Eventually(session).Should(gbytes.Say("C Suite"))
				Consistently(session).ShouldNot(gbytes.Say("A Suite|B Suite"))
			})
		})

		It("should not trigger dependents when tests are changed", func() {
			session = startGinkgoWithGopath("watch", "-succinct", "-r", "-depth=2")
			Eventually(session).Should(gbytes.Say("Identified 3 test suites"))
			Eventually(session).Should(gbytes.Say(`A \[2 dependencies\]`))
			Eventually(session).Should(gbytes.Say(`B \[1 dependency\]`))
			Eventually(session).Should(gbytes.Say(`C \[0 dependencies\]`))

			modifyTest("A")
			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("A Suite"))
			Consistently(session).ShouldNot(gbytes.Say("B Suite|C Suite"))

			modifyTest("B")
			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("B Suite"))
			Consistently(session).ShouldNot(gbytes.Say("A Suite|C Suite"))

			modifyTest("C")
			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("C Suite"))
			Consistently(session).ShouldNot(gbytes.Say("A Suite|B Suite"))
		})
	})

	Describe("when new test suite is added", func() {
		It("should start monitoring that test suite", func() {
			session = startGinkgoWithGopath("watch", "-succinct", "-r")

			Eventually(session).Should(gbytes.Say("Watching 3 suites"))

			pathD := filepath.Join(rootPath, "src", "github.com", "onsi", "D")

			err := os.MkdirAll(pathD, 0700)
			Ω(err).ShouldNot(HaveOccurred())

			copyIn(filepath.Join("watch_fixtures", "D"), pathD)

			Eventually(session).Should(gbytes.Say("Detected 1 new suite"))
			Eventually(session).Should(gbytes.Say(`D \[1 dependency\]`))
			Eventually(session).Should(gbytes.Say("D Suite"))

			modifyCode("D")

			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("D Suite"))

			modifyCode("C")

			Eventually(session).Should(gbytes.Say("Detected changes in"))
			Eventually(session).Should(gbytes.Say("C Suite"))
			Eventually(session).Should(gbytes.Say("D Suite"))
		})
	})
})
