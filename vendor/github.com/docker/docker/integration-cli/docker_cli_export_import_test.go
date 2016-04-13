package main

import (
	"os"
	"os/exec"
	"strings"

	"github.com/go-check/check"
)

// export an image and try to import it into a new one
func (s *DockerSuite) TestExportContainerAndImportImage(c *check.C) {
	containerID := "testexportcontainerandimportimage"

	dockerCmd(c, "run", "--name", containerID, "busybox", "true")

	out, _ := dockerCmd(c, "export", containerID)

	importCmd := exec.Command(dockerBinary, "import", "-", "repo/testexp:v1")
	importCmd.Stdin = strings.NewReader(out)
	out, _, err := runCommandWithOutput(importCmd)
	if err != nil {
		c.Fatalf("failed to import image: %s, %v", out, err)
	}

	cleanedImageID := strings.TrimSpace(out)
	if cleanedImageID == "" {
		c.Fatalf("output should have been an image id, got: %s", out)
	}
}

// Used to test output flag in the export command
func (s *DockerSuite) TestExportContainerWithOutputAndImportImage(c *check.C) {
	containerID := "testexportcontainerwithoutputandimportimage"

	dockerCmd(c, "run", "--name", containerID, "busybox", "true")
	dockerCmd(c, "export", "--output=testexp.tar", containerID)
	defer os.Remove("testexp.tar")

	out, _, err := runCommandWithOutput(exec.Command("cat", "testexp.tar"))
	if err != nil {
		c.Fatal(out, err)
	}

	importCmd := exec.Command(dockerBinary, "import", "-", "repo/testexp:v1")
	importCmd.Stdin = strings.NewReader(out)
	out, _, err = runCommandWithOutput(importCmd)
	if err != nil {
		c.Fatalf("failed to import image: %s, %v", out, err)
	}

	cleanedImageID := strings.TrimSpace(out)
	if cleanedImageID == "" {
		c.Fatalf("output should have been an image id, got: %s", out)
	}
}
