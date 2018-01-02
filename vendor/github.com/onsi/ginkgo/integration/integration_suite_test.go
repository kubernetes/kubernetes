package integration_test

import (
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"

	"testing"
	"time"
)

var tmpDir string
var pathToGinkgo string

func TestIntegration(t *testing.T) {
	SetDefaultEventuallyTimeout(15 * time.Second)
	RegisterFailHandler(Fail)
	RunSpecs(t, "Integration Suite")
}

var _ = SynchronizedBeforeSuite(func() []byte {
	pathToGinkgo, err := gexec.Build("github.com/onsi/ginkgo/ginkgo")
	Ω(err).ShouldNot(HaveOccurred())
	return []byte(pathToGinkgo)
}, func(computedPathToGinkgo []byte) {
	pathToGinkgo = string(computedPathToGinkgo)
})

var _ = BeforeEach(func() {
	var err error
	tmpDir, err = ioutil.TempDir("", "ginkgo-run")
	Ω(err).ShouldNot(HaveOccurred())
})

var _ = AfterEach(func() {
	err := os.RemoveAll(tmpDir)
	Ω(err).ShouldNot(HaveOccurred())
})

var _ = SynchronizedAfterSuite(func() {}, func() {
	gexec.CleanupBuildArtifacts()
})

func tmpPath(destination string) string {
	return filepath.Join(tmpDir, destination)
}

func copyIn(fixture string, destination string) {
	err := os.MkdirAll(destination, 0777)
	Ω(err).ShouldNot(HaveOccurred())

	filepath.Walk(filepath.Join("_fixtures", fixture), func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}

		base := filepath.Base(path)

		src, err := os.Open(path)
		Ω(err).ShouldNot(HaveOccurred())
		defer src.Close()

		dst, err := os.Create(filepath.Join(destination, base))
		Ω(err).ShouldNot(HaveOccurred())
		defer dst.Close()

		_, err = io.Copy(dst, src)
		Ω(err).ShouldNot(HaveOccurred())
		return nil
	})
}

func ginkgoCommand(dir string, args ...string) *exec.Cmd {
	cmd := exec.Command(pathToGinkgo, args...)
	cmd.Dir = dir

	return cmd
}

func startGinkgo(dir string, args ...string) *gexec.Session {
	cmd := ginkgoCommand(dir, args...)
	session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
	Ω(err).ShouldNot(HaveOccurred())
	return session
}
