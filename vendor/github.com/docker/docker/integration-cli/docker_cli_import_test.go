package main

import (
	"bufio"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestImportDisplay(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")
	cleanedContainerID := strings.TrimSpace(out)

	out, _, err := runCommandPipelineWithOutput(
		exec.Command(dockerBinary, "export", cleanedContainerID),
		exec.Command(dockerBinary, "import", "-"),
	)
	if err != nil {
		c.Errorf("import failed with errors: %v, output: %q", err, out)
	}

	if n := strings.Count(out, "\n"); n != 1 {
		c.Fatalf("display is messed up: %d '\\n' instead of 1:\n%s", n, out)
	}
	image := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "--rm", image, "true")
	if out != "" {
		c.Fatalf("command output should've been nothing, was %q", out)
	}
}

func (s *DockerSuite) TestImportBadURL(c *check.C) {
	out, _, err := dockerCmdWithError(c, "import", "http://nourl/bad")
	if err == nil {
		c.Fatal("import was supposed to fail but didn't")
	}
	if !strings.Contains(out, "dial tcp") {
		c.Fatalf("expected an error msg but didn't get one:\n%s", out)
	}
}

func (s *DockerSuite) TestImportFile(c *check.C) {
	dockerCmd(c, "run", "--name", "test-import", "busybox", "true")

	temporaryFile, err := ioutil.TempFile("", "exportImportTest")
	if err != nil {
		c.Fatal("failed to create temporary file", "", err)
	}
	defer os.Remove(temporaryFile.Name())

	runCmd := exec.Command(dockerBinary, "export", "test-import")
	runCmd.Stdout = bufio.NewWriter(temporaryFile)

	_, err = runCommand(runCmd)
	if err != nil {
		c.Fatal("failed to export a container", err)
	}

	out, _ := dockerCmd(c, "import", temporaryFile.Name())
	if n := strings.Count(out, "\n"); n != 1 {
		c.Fatalf("display is messed up: %d '\\n' instead of 1:\n%s", n, out)
	}
	image := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "run", "--rm", image, "true")
	if out != "" {
		c.Fatalf("command output should've been nothing, was %q", out)
	}
}

func (s *DockerSuite) TestImportFileNonExistentFile(c *check.C) {
	_, exitCode, err := dockerCmdWithError(c, "import", "example.com/myImage.tar")
	if exitCode == 0 || err == nil {
		c.Fatalf("import non-existing file must failed")
	}
}
