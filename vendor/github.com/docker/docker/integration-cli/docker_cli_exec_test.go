// +build !test_no_exec

package main

import (
	"bufio"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestExec(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "sh", "-c", "echo test > /tmp/file && top")
	c.Assert(waitRun(strings.TrimSpace(out)), check.IsNil)

	out, _ = dockerCmd(c, "exec", "testing", "cat", "/tmp/file")
	out = strings.Trim(out, "\r\n")
	c.Assert(out, checker.Equals, "test")

}

func (s *DockerSuite) TestExecInteractive(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "sh", "-c", "echo test > /tmp/file && top")

	execCmd := exec.Command(dockerBinary, "exec", "-i", "testing", "sh")
	stdin, err := execCmd.StdinPipe()
	c.Assert(err, checker.IsNil)
	stdout, err := execCmd.StdoutPipe()
	c.Assert(err, checker.IsNil)

	err = execCmd.Start()
	c.Assert(err, checker.IsNil)
	_, err = stdin.Write([]byte("cat /tmp/file\n"))
	c.Assert(err, checker.IsNil)

	r := bufio.NewReader(stdout)
	line, err := r.ReadString('\n')
	c.Assert(err, checker.IsNil)
	line = strings.TrimSpace(line)
	c.Assert(line, checker.Equals, "test")
	err = stdin.Close()
	c.Assert(err, checker.IsNil)
	errChan := make(chan error)
	go func() {
		errChan <- execCmd.Wait()
		close(errChan)
	}()
	select {
	case err := <-errChan:
		c.Assert(err, checker.IsNil)
	case <-time.After(1 * time.Second):
		c.Fatal("docker exec failed to exit on stdin close")
	}

}

func (s *DockerSuite) TestExecAfterContainerRestart(c *check.C) {
	out := runSleepingContainer(c)
	cleanedContainerID := strings.TrimSpace(out)
	c.Assert(waitRun(cleanedContainerID), check.IsNil)
	dockerCmd(c, "restart", cleanedContainerID)
	c.Assert(waitRun(cleanedContainerID), check.IsNil)

	out, _ = dockerCmd(c, "exec", cleanedContainerID, "echo", "hello")
	outStr := strings.TrimSpace(out)
	c.Assert(outStr, checker.Equals, "hello")
}

func (s *DockerDaemonSuite) TestExecAfterDaemonRestart(c *check.C) {
	// TODO Windows CI: Requires a little work to get this ported.
	testRequires(c, DaemonIsLinux, SameHostDaemon)
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--name", "top", "-p", "80", "busybox:latest", "top")
	c.Assert(err, checker.IsNil, check.Commentf("Could not run top: %s", out))

	s.d.Restart(c)

	out, err = s.d.Cmd("start", "top")
	c.Assert(err, checker.IsNil, check.Commentf("Could not start top after daemon restart: %s", out))

	out, err = s.d.Cmd("exec", "top", "echo", "hello")
	c.Assert(err, checker.IsNil, check.Commentf("Could not exec on container top: %s", out))

	outStr := strings.TrimSpace(string(out))
	c.Assert(outStr, checker.Equals, "hello")
}

// Regression test for #9155, #9044
func (s *DockerSuite) TestExecEnv(c *check.C) {
	// TODO Windows CI: This one is interesting and may just end up being a feature
	// difference between Windows and Linux. On Windows, the environment is passed
	// into the process that is launched, not into the machine environment. Hence
	// a subsequent exec will not have LALA set/
	testRequires(c, DaemonIsLinux)
	runSleepingContainer(c, "-e", "LALA=value1", "-e", "LALA=value2", "-d", "--name", "testing")
	c.Assert(waitRun("testing"), check.IsNil)

	out, _ := dockerCmd(c, "exec", "testing", "env")
	c.Assert(out, checker.Not(checker.Contains), "LALA=value1")
	c.Assert(out, checker.Contains, "LALA=value2")
	c.Assert(out, checker.Contains, "HOME=/root")
}

func (s *DockerSuite) TestExecSetEnv(c *check.C) {
	testRequires(c, DaemonIsLinux)
	runSleepingContainer(c, "-e", "HOME=/root", "-d", "--name", "testing")
	c.Assert(waitRun("testing"), check.IsNil)

	out, _ := dockerCmd(c, "exec", "-e", "HOME=/another", "-e", "ABC=xyz", "testing", "env")
	c.Assert(out, checker.Not(checker.Contains), "HOME=/root")
	c.Assert(out, checker.Contains, "HOME=/another")
	c.Assert(out, checker.Contains, "ABC=xyz")
}

func (s *DockerSuite) TestExecExitStatus(c *check.C) {
	runSleepingContainer(c, "-d", "--name", "top")

	result := icmd.RunCommand(dockerBinary, "exec", "top", "sh", "-c", "exit 23")
	c.Assert(result, icmd.Matches, icmd.Expected{ExitCode: 23, Error: "exit status 23"})
}

func (s *DockerSuite) TestExecPausedContainer(c *check.C) {
	testRequires(c, IsPausable)

	out := runSleepingContainer(c, "-d", "--name", "testing")
	ContainerID := strings.TrimSpace(out)

	dockerCmd(c, "pause", "testing")
	out, _, err := dockerCmdWithError("exec", ContainerID, "echo", "hello")
	c.Assert(err, checker.NotNil, check.Commentf("container should fail to exec new command if it is paused"))

	expected := ContainerID + " is paused, unpause the container before exec"
	c.Assert(out, checker.Contains, expected, check.Commentf("container should not exec new command if it is paused"))
}

// regression test for #9476
func (s *DockerSuite) TestExecTTYCloseStdin(c *check.C) {
	// TODO Windows CI: This requires some work to port to Windows.
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "-it", "--name", "exec_tty_stdin", "busybox")

	cmd := exec.Command(dockerBinary, "exec", "-i", "exec_tty_stdin", "cat")
	stdinRw, err := cmd.StdinPipe()
	c.Assert(err, checker.IsNil)

	stdinRw.Write([]byte("test"))
	stdinRw.Close()

	out, _, err := runCommandWithOutput(cmd)
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, _ = dockerCmd(c, "top", "exec_tty_stdin")
	outArr := strings.Split(out, "\n")
	c.Assert(len(outArr), checker.LessOrEqualThan, 3, check.Commentf("exec process left running"))
	c.Assert(out, checker.Not(checker.Contains), "nsenter-exec")
}

func (s *DockerSuite) TestExecTTYWithoutStdin(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-ti", "busybox")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	errChan := make(chan error)
	go func() {
		defer close(errChan)

		cmd := exec.Command(dockerBinary, "exec", "-ti", id, "true")
		if _, err := cmd.StdinPipe(); err != nil {
			errChan <- err
			return
		}

		expected := "the input device is not a TTY"
		if runtime.GOOS == "windows" {
			expected += ".  If you are using mintty, try prefixing the command with 'winpty'"
		}
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

// FIXME(vdemeester) this should be a unit tests on cli/command/container package
func (s *DockerSuite) TestExecParseError(c *check.C) {
	// TODO Windows CI: Requires some extra work. Consider copying the
	// runSleepingContainer helper to have an exec version.
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name", "top", "busybox", "top")

	// Test normal (non-detached) case first
	icmd.RunCommand(dockerBinary, "exec", "top").Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Err:      "See 'docker exec --help'",
	})
}

func (s *DockerSuite) TestExecStopNotHanging(c *check.C) {
	// TODO Windows CI: Requires some extra work. Consider copying the
	// runSleepingContainer helper to have an exec version.
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "top")

	err := exec.Command(dockerBinary, "exec", "testing", "top").Start()
	c.Assert(err, checker.IsNil)

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
	// Not applicable on Windows - using Linux specific functionality
	testRequires(c, NotUserNamespace)
	testRequires(c, DaemonIsLinux)
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
			out, _, err := dockerCmdWithError("exec", "testing", "cat", "/proc/self/cgroup")
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
		c.Assert(err, checker.IsNil)
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

func (s *DockerSuite) TestExecInspectID(c *check.C) {
	out := runSleepingContainer(c, "-d")
	id := strings.TrimSuffix(out, "\n")

	out = inspectField(c, id, "ExecIDs")
	c.Assert(out, checker.Equals, "[]", check.Commentf("ExecIDs should be empty, got: %s", out))

	// Start an exec, have it block waiting so we can do some checking
	cmd := exec.Command(dockerBinary, "exec", id, "sh", "-c",
		"while ! test -e /execid1; do sleep 1; done")

	err := cmd.Start()
	c.Assert(err, checker.IsNil, check.Commentf("failed to start the exec cmd"))

	// Give the exec 10 chances/seconds to start then give up and stop the test
	tries := 10
	for i := 0; i < tries; i++ {
		// Since its still running we should see exec as part of the container
		out = strings.TrimSpace(inspectField(c, id, "ExecIDs"))

		if out != "[]" && out != "<no value>" {
			break
		}
		c.Assert(i+1, checker.Not(checker.Equals), tries, check.Commentf("ExecIDs still empty after 10 second"))
		time.Sleep(1 * time.Second)
	}

	// Save execID for later
	execID, err := inspectFilter(id, "index .ExecIDs 0")
	c.Assert(err, checker.IsNil, check.Commentf("failed to get the exec id"))

	// End the exec by creating the missing file
	err = exec.Command(dockerBinary, "exec", id,
		"sh", "-c", "touch /execid1").Run()

	c.Assert(err, checker.IsNil, check.Commentf("failed to run the 2nd exec cmd"))

	// Wait for 1st exec to complete
	cmd.Wait()

	// Give the exec 10 chances/seconds to stop then give up and stop the test
	for i := 0; i < tries; i++ {
		// Since its still running we should see exec as part of the container
		out = strings.TrimSpace(inspectField(c, id, "ExecIDs"))

		if out == "[]" {
			break
		}
		c.Assert(i+1, checker.Not(checker.Equals), tries, check.Commentf("ExecIDs still not empty after 10 second"))
		time.Sleep(1 * time.Second)
	}

	// But we should still be able to query the execID
	sc, body, _ := request.SockRequest("GET", "/exec/"+execID+"/json", nil, daemonHost())

	c.Assert(sc, checker.Equals, http.StatusOK, check.Commentf("received status != 200 OK: %d\n%s", sc, body))

	// Now delete the container and then an 'inspect' on the exec should
	// result in a 404 (not 'container not running')
	out, ec := dockerCmd(c, "rm", "-f", id)
	c.Assert(ec, checker.Equals, 0, check.Commentf("error removing container: %s", out))
	sc, body, _ = request.SockRequest("GET", "/exec/"+execID+"/json", nil, daemonHost())
	c.Assert(sc, checker.Equals, http.StatusNotFound, check.Commentf("received status != 404: %d\n%s", sc, body))
}

func (s *DockerSuite) TestLinksPingLinkedContainersOnRename(c *check.C) {
	// Problematic on Windows as Windows does not support links
	testRequires(c, DaemonIsLinux)
	var out string
	out, _ = dockerCmd(c, "run", "-d", "--name", "container1", "busybox", "top")
	idA := strings.TrimSpace(out)
	c.Assert(idA, checker.Not(checker.Equals), "", check.Commentf("%s, id should not be nil", out))
	out, _ = dockerCmd(c, "run", "-d", "--link", "container1:alias1", "--name", "container2", "busybox", "top")
	idB := strings.TrimSpace(out)
	c.Assert(idB, checker.Not(checker.Equals), "", check.Commentf("%s, id should not be nil", out))

	dockerCmd(c, "exec", "container2", "ping", "-c", "1", "alias1", "-W", "1")
	dockerCmd(c, "rename", "container1", "container_new")
	dockerCmd(c, "exec", "container2", "ping", "-c", "1", "alias1", "-W", "1")
}

func (s *DockerSuite) TestRunMutableNetworkFiles(c *check.C) {
	// Not applicable on Windows to Windows CI.
	testRequires(c, SameHostDaemon, DaemonIsLinux)
	for _, fn := range []string{"resolv.conf", "hosts"} {
		containers := cli.DockerCmd(c, "ps", "-q", "-a").Combined()
		if containers != "" {
			cli.DockerCmd(c, append([]string{"rm", "-fv"}, strings.Split(strings.TrimSpace(containers), "\n")...)...)
		}

		content := runCommandAndReadContainerFile(c, fn, dockerBinary, "run", "-d", "--name", "c1", "busybox", "sh", "-c", fmt.Sprintf("echo success >/etc/%s && top", fn))

		c.Assert(strings.TrimSpace(string(content)), checker.Equals, "success", check.Commentf("Content was not what was modified in the container", string(content)))

		out, _ := dockerCmd(c, "run", "-d", "--name", "c2", "busybox", "top")
		contID := strings.TrimSpace(out)
		netFilePath := containerStorageFile(contID, fn)

		f, err := os.OpenFile(netFilePath, os.O_WRONLY|os.O_SYNC|os.O_APPEND, 0644)
		c.Assert(err, checker.IsNil)

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
		c.Assert(res, checker.Equals, "success2\n")
	}
}

func (s *DockerSuite) TestExecWithUser(c *check.C) {
	// TODO Windows CI: This may be fixable in the future once Windows
	// supports users
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "-d", "--name", "parent", "busybox", "top")

	out, _ := dockerCmd(c, "exec", "-u", "1", "parent", "id")
	c.Assert(out, checker.Contains, "uid=1(daemon) gid=1(daemon)")

	out, _ = dockerCmd(c, "exec", "-u", "root", "parent", "id")
	c.Assert(out, checker.Contains, "uid=0(root) gid=0(root)", check.Commentf("exec with user by id expected daemon user got %s", out))
}

func (s *DockerSuite) TestExecWithPrivileged(c *check.C) {
	// Not applicable on Windows
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	// Start main loop which attempts mknod repeatedly
	dockerCmd(c, "run", "-d", "--name", "parent", "--cap-drop=ALL", "busybox", "sh", "-c", `while (true); do if [ -e /exec_priv ]; then cat /exec_priv && mknod /tmp/sda b 8 0 && echo "Success"; else echo "Privileged exec has not run yet"; fi; usleep 10000; done`)

	// Check exec mknod doesn't work
	icmd.RunCommand(dockerBinary, "exec", "parent", "sh", "-c", "mknod /tmp/sdb b 8 16").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})

	// Check exec mknod does work with --privileged
	result := icmd.RunCommand(dockerBinary, "exec", "--privileged", "parent", "sh", "-c", `echo "Running exec --privileged" > /exec_priv && mknod /tmp/sdb b 8 16 && usleep 50000 && echo "Finished exec --privileged" > /exec_priv && echo ok`)
	result.Assert(c, icmd.Success)

	actual := strings.TrimSpace(result.Combined())
	c.Assert(actual, checker.Equals, "ok", check.Commentf("exec mknod in --cap-drop=ALL container with --privileged failed, output: %q", result.Combined()))

	// Check subsequent unprivileged exec cannot mknod
	icmd.RunCommand(dockerBinary, "exec", "parent", "sh", "-c", "mknod /tmp/sdc b 8 32").Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "Operation not permitted",
	})
	// Confirm at no point was mknod allowed
	result = icmd.RunCommand(dockerBinary, "logs", "parent")
	result.Assert(c, icmd.Success)
	c.Assert(result.Combined(), checker.Not(checker.Contains), "Success")

}

func (s *DockerSuite) TestExecWithImageUser(c *check.C) {
	// Not applicable on Windows
	testRequires(c, DaemonIsLinux)
	name := "testbuilduser"
	buildImageSuccessfully(c, name, build.WithDockerfile(`FROM busybox
		RUN echo 'dockerio:x:1001:1001::/bin:/bin/false' >> /etc/passwd
		USER dockerio`))
	dockerCmd(c, "run", "-d", "--name", "dockerioexec", name, "top")

	out, _ := dockerCmd(c, "exec", "dockerioexec", "whoami")
	c.Assert(out, checker.Contains, "dockerio", check.Commentf("exec with user by id expected dockerio user got %s", out))
}

func (s *DockerSuite) TestExecOnReadonlyContainer(c *check.C) {
	// Windows does not support read-only
	// --read-only + userns has remount issues
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	dockerCmd(c, "run", "-d", "--read-only", "--name", "parent", "busybox", "top")
	dockerCmd(c, "exec", "parent", "true")
}

func (s *DockerSuite) TestExecUlimits(c *check.C) {
	testRequires(c, DaemonIsLinux)
	name := "testexeculimits"
	runSleepingContainer(c, "-d", "--ulimit", "nofile=511:511", "--name", name)
	c.Assert(waitRun(name), checker.IsNil)

	out, _, err := dockerCmdWithError("exec", name, "sh", "-c", "ulimit -n")
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, "511")
}

// #15750
func (s *DockerSuite) TestExecStartFails(c *check.C) {
	// TODO Windows CI. This test should be portable. Figure out why it fails
	// currently.
	testRequires(c, DaemonIsLinux)
	name := "exec-15750"
	runSleepingContainer(c, "-d", "--name", name)
	c.Assert(waitRun(name), checker.IsNil)

	out, _, err := dockerCmdWithError("exec", name, "no-such-cmd")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "executable file not found")
}

// Fix regression in https://github.com/docker/docker/pull/26461#issuecomment-250287297
func (s *DockerSuite) TestExecWindowsPathNotWiped(c *check.C) {
	testRequires(c, DaemonIsWindows)
	out, _ := dockerCmd(c, "run", "-d", "--name", "testing", minimalBaseImage(), "powershell", "start-sleep", "60")
	c.Assert(waitRun(strings.TrimSpace(out)), check.IsNil)

	out, _ = dockerCmd(c, "exec", "testing", "powershell", "write-host", "$env:PATH")
	out = strings.ToLower(strings.Trim(out, "\r\n"))
	c.Assert(out, checker.Contains, `windowspowershell\v1.0`)
}

func (s *DockerSuite) TestExecEnvLinksHost(c *check.C) {
	testRequires(c, DaemonIsLinux)
	runSleepingContainer(c, "-d", "--name", "foo")
	runSleepingContainer(c, "-d", "--link", "foo:db", "--hostname", "myhost", "--name", "bar")
	out, _ := dockerCmd(c, "exec", "bar", "env")
	c.Assert(out, checker.Contains, "HOSTNAME=myhost")
	c.Assert(out, checker.Contains, "DB_NAME=/bar/db")
}

func (s *DockerSuite) TestExecWindowsOpenHandles(c *check.C) {
	testRequires(c, DaemonIsWindows)
	runSleepingContainer(c, "-d", "--name", "test")
	exec := make(chan bool)
	go func() {
		dockerCmd(c, "exec", "test", "cmd", "/c", "start sleep 10")
		exec <- true
	}()

	count := 0
	for {
		top := make(chan string)
		var out string
		go func() {
			out, _ := dockerCmd(c, "top", "test")
			top <- out
		}()

		select {
		case <-time.After(time.Second * 5):
			c.Fatal("timed out waiting for top while exec is exiting")
		case out = <-top:
			break
		}

		if strings.Count(out, "busybox.exe") == 2 && !strings.Contains(out, "cmd.exe") {
			// The initial exec process (cmd.exe) has exited, and both sleeps are currently running
			break
		}
		count++
		if count >= 30 {
			c.Fatal("too many retries")
		}
		time.Sleep(1 * time.Second)
	}

	inspect := make(chan bool)
	go func() {
		dockerCmd(c, "inspect", "test")
		inspect <- true
	}()

	select {
	case <-time.After(time.Second * 5):
		c.Fatal("timed out waiting for inspect while exec is exiting")
	case <-inspect:
		break
	}

	// Ensure the background sleep is still running
	out, _ := dockerCmd(c, "top", "test")
	c.Assert(strings.Count(out, "busybox.exe"), checker.Equals, 2)

	// The exec should exit when the background sleep exits
	select {
	case <-time.After(time.Second * 15):
		c.Fatal("timed out waiting for async exec to exit")
	case <-exec:
		// Ensure the background sleep has actually exited
		out, _ := dockerCmd(c, "top", "test")
		c.Assert(strings.Count(out, "busybox.exe"), checker.Equals, 1)
		break
	}
}
