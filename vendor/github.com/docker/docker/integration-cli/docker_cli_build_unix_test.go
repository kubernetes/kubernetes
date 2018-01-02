// +build !windows

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/cli/build/fakecontext"
	"github.com/docker/docker/pkg/testutil"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/docker/go-units"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestBuildResourceConstraintsAreUsed(c *check.C) {
	testRequires(c, cpuCfsQuota)
	name := "testbuildresourceconstraints"

	ctx := fakecontext.New(c, "", fakecontext.WithDockerfile(`
	FROM hello-world:frozen
	RUN ["/hello"]
	`))
	cli.Docker(
		cli.Args("build", "--no-cache", "--rm=false", "--memory=64m", "--memory-swap=-1", "--cpuset-cpus=0", "--cpuset-mems=0", "--cpu-shares=100", "--cpu-quota=8000", "--ulimit", "nofile=42", "-t", name, "."),
		cli.InDir(ctx.Dir),
	).Assert(c, icmd.Success)

	out := cli.DockerCmd(c, "ps", "-lq").Combined()
	cID := strings.TrimSpace(out)

	type hostConfig struct {
		Memory     int64
		MemorySwap int64
		CpusetCpus string
		CpusetMems string
		CPUShares  int64
		CPUQuota   int64
		Ulimits    []*units.Ulimit
	}

	cfg := inspectFieldJSON(c, cID, "HostConfig")

	var c1 hostConfig
	err := json.Unmarshal([]byte(cfg), &c1)
	c.Assert(err, checker.IsNil, check.Commentf(cfg))

	c.Assert(c1.Memory, checker.Equals, int64(64*1024*1024), check.Commentf("resource constraints not set properly for Memory"))
	c.Assert(c1.MemorySwap, checker.Equals, int64(-1), check.Commentf("resource constraints not set properly for MemorySwap"))
	c.Assert(c1.CpusetCpus, checker.Equals, "0", check.Commentf("resource constraints not set properly for CpusetCpus"))
	c.Assert(c1.CpusetMems, checker.Equals, "0", check.Commentf("resource constraints not set properly for CpusetMems"))
	c.Assert(c1.CPUShares, checker.Equals, int64(100), check.Commentf("resource constraints not set properly for CPUShares"))
	c.Assert(c1.CPUQuota, checker.Equals, int64(8000), check.Commentf("resource constraints not set properly for CPUQuota"))
	c.Assert(c1.Ulimits[0].Name, checker.Equals, "nofile", check.Commentf("resource constraints not set properly for Ulimits"))
	c.Assert(c1.Ulimits[0].Hard, checker.Equals, int64(42), check.Commentf("resource constraints not set properly for Ulimits"))

	// Make sure constraints aren't saved to image
	cli.DockerCmd(c, "run", "--name=test", name)

	cfg = inspectFieldJSON(c, "test", "HostConfig")

	var c2 hostConfig
	err = json.Unmarshal([]byte(cfg), &c2)
	c.Assert(err, checker.IsNil, check.Commentf(cfg))

	c.Assert(c2.Memory, check.Not(checker.Equals), int64(64*1024*1024), check.Commentf("resource leaked from build for Memory"))
	c.Assert(c2.MemorySwap, check.Not(checker.Equals), int64(-1), check.Commentf("resource leaked from build for MemorySwap"))
	c.Assert(c2.CpusetCpus, check.Not(checker.Equals), "0", check.Commentf("resource leaked from build for CpusetCpus"))
	c.Assert(c2.CpusetMems, check.Not(checker.Equals), "0", check.Commentf("resource leaked from build for CpusetMems"))
	c.Assert(c2.CPUShares, check.Not(checker.Equals), int64(100), check.Commentf("resource leaked from build for CPUShares"))
	c.Assert(c2.CPUQuota, check.Not(checker.Equals), int64(8000), check.Commentf("resource leaked from build for CPUQuota"))
	c.Assert(c2.Ulimits, checker.IsNil, check.Commentf("resource leaked from build for Ulimits"))
}

func (s *DockerSuite) TestBuildAddChangeOwnership(c *check.C) {
	testRequires(c, DaemonIsLinux)
	name := "testbuildaddown"

	ctx := func() *fakecontext.Fake {
		dockerfile := `
			FROM busybox
			ADD foo /bar/
			RUN [ $(stat -c %U:%G "/bar") = 'root:root' ]
			RUN [ $(stat -c %U:%G "/bar/foo") = 'root:root' ]
			`
		tmpDir, err := ioutil.TempDir("", "fake-context")
		c.Assert(err, check.IsNil)
		testFile, err := os.Create(filepath.Join(tmpDir, "foo"))
		if err != nil {
			c.Fatalf("failed to create foo file: %v", err)
		}
		defer testFile.Close()

		icmd.RunCmd(icmd.Cmd{
			Command: []string{"chown", "daemon:daemon", "foo"},
			Dir:     tmpDir,
		}).Assert(c, icmd.Success)

		if err := ioutil.WriteFile(filepath.Join(tmpDir, "Dockerfile"), []byte(dockerfile), 0644); err != nil {
			c.Fatalf("failed to open destination dockerfile: %v", err)
		}
		return fakecontext.New(c, tmpDir)
	}()

	defer ctx.Close()

	buildImageSuccessfully(c, name, build.WithExternalBuildContext(ctx))
}

// Test that an infinite sleep during a build is killed if the client disconnects.
// This test is fairly hairy because there are lots of ways to race.
// Strategy:
// * Monitor the output of docker events starting from before
// * Run a 1-year-long sleep from a docker build.
// * When docker events sees container start, close the "docker build" command
// * Wait for docker events to emit a dying event.
func (s *DockerSuite) TestBuildCancellationKillsSleep(c *check.C) {
	testRequires(c, DaemonIsLinux)
	name := "testbuildcancellation"

	observer, err := newEventObserver(c)
	c.Assert(err, checker.IsNil)
	err = observer.Start()
	c.Assert(err, checker.IsNil)
	defer observer.Stop()

	// (Note: one year, will never finish)
	ctx := fakecontext.New(c, "", fakecontext.WithDockerfile("FROM busybox\nRUN sleep 31536000"))
	defer ctx.Close()

	buildCmd := exec.Command(dockerBinary, "build", "-t", name, ".")
	buildCmd.Dir = ctx.Dir

	stdoutBuild, err := buildCmd.StdoutPipe()
	c.Assert(err, checker.IsNil)

	if err := buildCmd.Start(); err != nil {
		c.Fatalf("failed to run build: %s", err)
	}

	matchCID := regexp.MustCompile("Running in (.+)")
	scanner := bufio.NewScanner(stdoutBuild)

	outputBuffer := new(bytes.Buffer)
	var buildID string
	for scanner.Scan() {
		line := scanner.Text()
		outputBuffer.WriteString(line)
		outputBuffer.WriteString("\n")
		if matches := matchCID.FindStringSubmatch(line); len(matches) > 0 {
			buildID = matches[1]
			break
		}
	}

	if buildID == "" {
		c.Fatalf("Unable to find build container id in build output:\n%s", outputBuffer.String())
	}

	testActions := map[string]chan bool{
		"start": make(chan bool, 1),
		"die":   make(chan bool, 1),
	}

	matcher := matchEventLine(buildID, "container", testActions)
	processor := processEventMatch(testActions)
	go observer.Match(matcher, processor)

	select {
	case <-time.After(10 * time.Second):
		observer.CheckEventError(c, buildID, "start", matcher)
	case <-testActions["start"]:
		// ignore, done
	}

	// Send a kill to the `docker build` command.
	// Causes the underlying build to be cancelled due to socket close.
	if err := buildCmd.Process.Kill(); err != nil {
		c.Fatalf("error killing build command: %s", err)
	}

	// Get the exit status of `docker build`, check it exited because killed.
	if err := buildCmd.Wait(); err != nil && !testutil.IsKilled(err) {
		c.Fatalf("wait failed during build run: %T %s", err, err)
	}

	select {
	case <-time.After(10 * time.Second):
		observer.CheckEventError(c, buildID, "die", matcher)
	case <-testActions["die"]:
		// ignore, done
	}
}
