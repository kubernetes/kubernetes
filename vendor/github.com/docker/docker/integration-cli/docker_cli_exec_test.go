// +build !test_no_exec

package main

import (
	"bufio"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestExec(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "sh", "-c", "echo test > /tmp/file && top")

	out, _ := dockerCmd(c, "exec", "testing", "cat", "/tmp/file")
	out = strings.Trim(out, "\r\n")
	if out != "test" {
		c.Errorf("container exec should've printed test but printed %q", out)
	}

}

func (s *DockerSuite) TestExecInteractive(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "sh", "-c", "echo test > /tmp/file && top")

	execCmd := exec.Command(dockerBinary, "exec", "-i", "testing", "sh")
	stdin, err := execCmd.StdinPipe()
	if err != nil {
		c.Fatal(err)
	}
	stdout, err := execCmd.StdoutPipe()
	if err != nil {
		c.Fatal(err)
	}

	if err := execCmd.Start(); err != nil {
		c.Fatal(err)
	}
	if _, err := stdin.Write([]byte("cat /tmp/file\n")); err != nil {
		c.Fatal(err)
	}

	r := bufio.NewReader(stdout)
	line, err := r.ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	line = strings.TrimSpace(line)
	if line != "test" {
		c.Fatalf("Output should be 'test', got '%q'", line)
	}
	if err := stdin.Close(); err != nil {
		c.Fatal(err)
	}
	errChan := make(chan error)
	go func() {
		errChan <- execCmd.Wait()
		close(errChan)
	}()
	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(1 * time.Second):
		c.Fatal("docker exec failed to exit on stdin close")
	}

}

func (s *DockerSuite) TestExecAfterContainerRestart(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "restart", cleanedContainerID)

	out, _ = dockerCmd(c, "exec", cleanedContainerID, "echo", "hello")
	outStr := strings.TrimSpace(out)
	if outStr != "hello" {
		c.Errorf("container should've printed hello, instead printed %q", outStr)
	}
}

func (s *DockerDaemonSuite) TestExecAfterDaemonRestart(c *check.C) {
	testRequires(c, SameHostDaemon)

	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	if out, err := s.d.Cmd("run", "-d", "--name", "top", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top: err=%v\n%s", err, out)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatalf("Could not restart daemon: %v", err)
	}

	if out, err := s.d.Cmd("start", "top"); err != nil {
		c.Fatalf("Could not start top after daemon restart: err=%v\n%s", err, out)
	}

	out, err := s.d.Cmd("exec", "top", "echo", "hello")
	if err != nil {
		c.Fatalf("Could not exec on container top: err=%v\n%s", err, out)
	}

	outStr := strings.TrimSpace(string(out))
	if outStr != "hello" {
		c.Errorf("container should've printed hello, instead printed %q", outStr)
	}
}

// Regression test for #9155, #9044
func (s *DockerSuite) TestExecEnv(c *check.C) {
	dockerCmd(c, "run", "-e", "LALA=value1", "-e", "LALA=value2",
		"-d", "--name", "testing", "busybox", "top")

	out, _ := dockerCmd(c, "exec", "testing", "env")
	if strings.Contains(out, "LALA=value1") ||
		!strings.Contains(out, "LALA=value2") ||
		!strings.Contains(out, "HOME=/root") {
		c.Errorf("exec env(%q), expect %q, %q", out, "LALA=value2", "HOME=/root")
	}
}

func (s *DockerSuite) TestExecExitStatus(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "top", "busybox", "top")

	// Test normal (non-detached) case first
	cmd := exec.Command(dockerBinary, "exec", "top", "sh", "-c", "exit 23")
	ec, _ := runCommand(cmd)
	if ec != 23 {
		c.Fatalf("Should have had an ExitCode of 23, not: %d", ec)
	}
}

func (s *DockerSuite) TestExecPausedContainer(c *check.C) {
	defer unpauseAllContainers()

	out, _ := dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "top")
	ContainerID := strings.TrimSpace(out)

	dockerCmd(c, "pause", "testing")
	out, _, err := dockerCmdWithError(c, "exec", "-i", "-t", ContainerID, "echo", "hello")
	if err == nil {
		c.Fatal("container should fail to exec new command if it is paused")
	}

	expected := ContainerID + " is paused, unpause the container before exec"
	if !strings.Contains(out, expected) {
		c.Fatal("container should not exec new command if it is paused")
	}
}

// regression test for #9476
func (s *DockerSuite) TestExecTtyCloseStdin(c *check.C) {
	dockerCmd(c, "run", "-d", "-it", "--name", "exec_tty_stdin", "busybox")

	cmd := exec.Command(dockerBinary, "exec", "-i", "exec_tty_stdin", "cat")
	stdinRw, err := cmd.StdinPipe()
	if err != nil {
		c.Fatal(err)
	}

	stdinRw.Write([]byte("test"))
	stdinRw.Close()

	if out, _, err := runCommandWithOutput(cmd); err != nil {
		c.Fatal(out, err)
	}

	out, _ := dockerCmd(c, "top", "exec_tty_stdin")
	outArr := strings.Split(out, "\n")
	if len(outArr) > 3 || strings.Contains(out, "nsenter-exec") {
		c.Fatalf("exec process left running\n\t %s", out)
	}
}

func (s *DockerSuite) TestExecTtyWithoutStdin(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-ti", "busybox")
	id := strings.TrimSpace(out)
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	defer func() {
		dockerCmd(c, "kill", id)
	}()

	errChan := make(chan error)
	go func() {
		defer close(errChan)

		cmd := exec.Command(dockerBinary, "exec", "-ti", id, "true")
		if _, err := cmd.StdinPipe(); err != nil {
			errChan <- err
			return
		}

		expected := "cannot enable tty mode"
		if out, _, err := runCommandWithOutput(cmd); err == nil {
			errChan <- fmt.Errorf("exec should have failed")
			return
		} else if !strings.Contains(out, expected) {
			errChan <- fmt.Errorf("exec failed with error %q: expected %q", out, expected)
			return
		}
	}()

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(3 * time.Second):
		c.Fatal("exec is running but should have failed")
	}
}

func (s *DockerSuite) TestExecParseError(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "top", "busybox", "top")

	// Test normal (non-detached) case first
	cmd := exec.Command(dockerBinary, "exec", "top")
	if _, stderr, code, err := runCommandWithStdoutStderr(cmd); err == nil || !strings.Contains(stderr, "See '"+dockerBinary+" exec --help'") || code == 0 {
		c.Fatalf("Should have thrown error & point to help: %s", stderr)
	}
}

func (s *DockerSuite) TestExecStopNotHanging(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "top")

	if err := exec.Command(dockerBinary, "exec", "testing", "top").Start(); err != nil {
		c.Fatal(err)
	}

	type dstop struct {
		out []byte
		err error
	}

	ch := make(chan dstop)
	go func() {
		out, err := exec.Command(dockerBinary, "stop", "testing").CombinedOutput()
		ch <- dstop{out, err}
		close(ch)
	}()
	select {
	case <-time.After(3 * time.Second):
		c.Fatal("Container stop timed out")
	case s := <-ch:
		c.Assert(s.err, check.IsNil)
	}
}

func (s *DockerSuite) TestExecCgroup(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "top")

	out, _ := dockerCmd(c, "exec", "testing", "cat", "/proc/1/cgroup")
	containerCgroups := sort.StringSlice(strings.Split(out, "\n"))

	var wg sync.WaitGroup
	var mu sync.Mutex
	execCgroups := []sort.StringSlice{}
	errChan := make(chan error)
	// exec a few times concurrently to get consistent failure
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			out, _, err := dockerCmdWithError(c, "exec", "testing", "cat", "/proc/self/cgroup")
			if err != nil {
				errChan <- err
				return
			}
			cg := sort.StringSlice(strings.Split(out, "\n"))

			mu.Lock()
			execCgroups = append(execCgroups, cg)
			mu.Unlock()
			wg.Done()
		}()
	}
	wg.Wait()
	close(errChan)

	for err := range errChan {
		c.Assert(err, check.IsNil)
	}

	for _, cg := range execCgroups {
		if !reflect.DeepEqual(cg, containerCgroups) {
			fmt.Println("exec cgroups:")
			for _, name := range cg {
				fmt.Printf(" %s\n", name)
			}

			fmt.Println("container cgroups:")
			for _, name := range containerCgroups {
				fmt.Printf(" %s\n", name)
			}
			c.Fatal("cgroups mismatched")
		}
	}
}

func (s *DockerSuite) TestInspectExecID(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id := strings.TrimSuffix(out, "\n")

	out, err := inspectField(id, "ExecIDs")
	if err != nil {
		c.Fatalf("failed to inspect container: %s, %v", out, err)
	}
	if out != "[]" {
		c.Fatalf("ExecIDs should be empty, got: %s", out)
	}

	// Start an exec, have it block waiting for input so we can do some checking
	cmd := exec.Command(dockerBinary, "exec", "-i", id, "sh", "-c", "read a")
	execStdin, _ := cmd.StdinPipe()

	if err = cmd.Start(); err != nil {
		c.Fatalf("failed to start the exec cmd: %q", err)
	}

	// Give the exec 10 chances/seconds to start then give up and stop the test
	tries := 10
	for i := 0; i < tries; i++ {
		// Since its still running we should see exec as part of the container
		out, err = inspectField(id, "ExecIDs")
		if err != nil {
			c.Fatalf("failed to inspect container: %s, %v", out, err)
		}

		out = strings.TrimSuffix(out, "\n")
		if out != "[]" && out != "<no value>" {
			break
		}
		if i+1 == tries {
			c.Fatalf("ExecIDs should not be empty, got: %s", out)
		}
		time.Sleep(1 * time.Second)
	}

	// Save execID for later
	execID, err := inspectFilter(id, "index .ExecIDs 0")
	if err != nil {
		c.Fatalf("failed to get the exec id: %v", err)
	}

	// End the exec by closing its stdin, and wait for it to end
	execStdin.Close()
	cmd.Wait()

	// All execs for the container should be gone now
	out, err = inspectField(id, "ExecIDs")
	if err != nil {
		c.Fatalf("failed to inspect container: %s, %v", out, err)
	}

	out = strings.TrimSuffix(out, "\n")
	if out != "[]" && out != "<no value>" {
		c.Fatalf("ExecIDs should be empty, got: %s", out)
	}

	// But we should still be able to query the execID
	sc, body, err := sockRequest("GET", "/exec/"+execID+"/json", nil)
	if sc != http.StatusOK {
		c.Fatalf("received status != 200 OK: %d\n%s", sc, body)
	}
}

func (s *DockerSuite) TestLinksPingLinkedContainersOnRename(c *check.C) {
	var out string
	out, _ = dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	idA := strings.TrimSpace(out)
	if idA == "" {
		c.Fatal(out, "id should not be nil")
	}
	out, _ = dockerCmd(c, "run", "-d", "--link", "container1:alias1", "--name", "container2", "busybox", "top")
	idB := strings.TrimSpace(out)
	if idB == "" {
		c.Fatal(out, "id should not be nil")
	}

	dockerCmd(c, "exec", "container2", "ping", "-c", "1", "alias1", "-W", "1")
	dockerCmd(c, "rename", "container1", "container_new")
	dockerCmd(c, "exec", "container2", "ping", "-c", "1", "alias1", "-W", "1")
}

func (s *DockerSuite) TestRunExecDir(c *check.C) {
	testRequires(c, SameHostDaemon)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id := strings.TrimSpace(out)
	execDir := filepath.Join(execDriverPath, id)
	stateFile := filepath.Join(execDir, "state.json")

	{
		fi, err := os.Stat(execDir)
		if err != nil {
			c.Fatal(err)
		}
		if !fi.IsDir() {
			c.Fatalf("%q must be a directory", execDir)
		}
		fi, err = os.Stat(stateFile)
		if err != nil {
			c.Fatal(err)
		}
	}

	dockerCmd(c, "stop", id)
	{
		_, err := os.Stat(execDir)
		if err == nil {
			c.Fatal(err)
		}
		if err == nil {
			c.Fatalf("Exec directory %q exists for removed container!", execDir)
		}
		if !os.IsNotExist(err) {
			c.Fatalf("Error should be about non-existing, got %s", err)
		}
	}
	dockerCmd(c, "start", id)
	{
		fi, err := os.Stat(execDir)
		if err != nil {
			c.Fatal(err)
		}
		if !fi.IsDir() {
			c.Fatalf("%q must be a directory", execDir)
		}
		fi, err = os.Stat(stateFile)
		if err != nil {
			c.Fatal(err)
		}
	}
	dockerCmd(c, "rm", "-f", id)
	{
		_, err := os.Stat(execDir)
		if err == nil {
			c.Fatal(err)
		}
		if err == nil {
			c.Fatalf("Exec directory %q is exists for removed container!", execDir)
		}
		if !os.IsNotExist(err) {
			c.Fatalf("Error should be about non-existing, got %s", err)
		}
	}
}

func (s *DockerSuite) TestRunMutableNetworkFiles(c *check.C) {
	testRequires(c, SameHostDaemon)

	for _, fn := range []string{"resolv.conf", "hosts"} {
		deleteAllContainers()

		content, err := runCommandAndReadContainerFile(fn, exec.Command(dockerBinary, "run", "-d", "--name", "c1", "busybox", "sh", "-c", fmt.Sprintf("echo success >/etc/%s && top", fn)))
		if err != nil {
			c.Fatal(err)
		}

		if strings.TrimSpace(string(content)) != "success" {
			c.Fatal("Content was not what was modified in the container", string(content))
		}

		out, _ := dockerCmd(c, "run", "-d", "--name", "c2", "busybox", "top")
		contID := strings.TrimSpace(out)
		netFilePath := containerStorageFile(contID, fn)

		f, err := os.OpenFile(netFilePath, os.O_WRONLY|os.O_SYNC|os.O_APPEND, 0644)
		if err != nil {
			c.Fatal(err)
		}

		if _, err := f.Seek(0, 0); err != nil {
			f.Close()
			c.Fatal(err)
		}

		if err := f.Truncate(0); err != nil {
			f.Close()
			c.Fatal(err)
		}

		if _, err := f.Write([]byte("success2\n")); err != nil {
			f.Close()
			c.Fatal(err)
		}
		f.Close()

		res, _ := dockerCmd(c, "exec", contID, "cat", "/etc/"+fn)
		if res != "success2\n" {
			c.Fatalf("Expected content of %s: %q, got: %q", fn, "success2\n", res)
		}
	}
}

func (s *DockerSuite) TestExecWithUser(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "parent", "busybox", "top")

	out, _ := dockerCmd(c, "exec", "-u", "1", "parent", "id")
	if !strings.Contains(out, "uid=1(daemon) gid=1(daemon)") {
		c.Fatalf("exec with user by id expected daemon user got %s", out)
	}

	out, _ = dockerCmd(c, "exec", "-u", "root", "parent", "id")
	if !strings.Contains(out, "uid=0(root) gid=0(root)") {
		c.Fatalf("exec with user by root expected root user got %s", out)
	}
}

func (s *DockerSuite) TestExecWithImageUser(c *check.C) {
	name := "testbuilduser"
	_, err := buildImage(name,
		`FROM busybox
		RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
		USER dockerio`,
		true)
	if err != nil {
		c.Fatalf("Could not build image %s: %v", name, err)
	}

	dockerCmd(c, "run", "-d", "--name", "dockerioexec", name, "top")

	out, _ := dockerCmd(c, "exec", "dockerioexec", "whoami")
	if !strings.Contains(out, "dockerio") {
		c.Fatalf("exec with user by id expected dockerio user got %s", out)
	}
}
