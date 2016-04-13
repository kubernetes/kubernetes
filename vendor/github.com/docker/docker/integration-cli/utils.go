package main

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http/httptest"
	"os"
	"os/exec"
	"path"
	"reflect"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/pkg/stringutils"
)

func getExitCode(err error) (int, error) {
	exitCode := 0
	if exiterr, ok := err.(*exec.ExitError); ok {
		if procExit, ok := exiterr.Sys().(syscall.WaitStatus); ok {
			return procExit.ExitStatus(), nil
		}
	}
	return exitCode, fmt.Errorf("failed to get exit code")
}

func processExitCode(err error) (exitCode int) {
	if err != nil {
		var exiterr error
		if exitCode, exiterr = getExitCode(err); exiterr != nil {
			// TODO: Fix this so we check the error's text.
			// we've failed to retrieve exit code, so we set it to 127
			exitCode = 127
		}
	}
	return
}

func IsKilled(err error) bool {
	if exitErr, ok := err.(*exec.ExitError); ok {
		status, ok := exitErr.Sys().(syscall.WaitStatus)
		if !ok {
			return false
		}
		// status.ExitStatus() is required on Windows because it does not
		// implement Signal() nor Signaled(). Just check it had a bad exit
		// status could mean it was killed (and in tests we do kill)
		return (status.Signaled() && status.Signal() == os.Kill) || status.ExitStatus() != 0
	}
	return false
}

func runCommandWithOutput(cmd *exec.Cmd) (output string, exitCode int, err error) {
	exitCode = 0
	out, err := cmd.CombinedOutput()
	exitCode = processExitCode(err)
	output = string(out)
	return
}

func runCommandWithStdoutStderr(cmd *exec.Cmd) (stdout string, stderr string, exitCode int, err error) {
	var (
		stderrBuffer, stdoutBuffer bytes.Buffer
	)
	exitCode = 0
	cmd.Stderr = &stderrBuffer
	cmd.Stdout = &stdoutBuffer
	err = cmd.Run()
	exitCode = processExitCode(err)

	stdout = stdoutBuffer.String()
	stderr = stderrBuffer.String()
	return
}

func runCommandWithOutputForDuration(cmd *exec.Cmd, duration time.Duration) (output string, exitCode int, timedOut bool, err error) {
	var outputBuffer bytes.Buffer
	if cmd.Stdout != nil {
		err = errors.New("cmd.Stdout already set")
		return
	}
	cmd.Stdout = &outputBuffer

	if cmd.Stderr != nil {
		err = errors.New("cmd.Stderr already set")
		return
	}
	cmd.Stderr = &outputBuffer

	done := make(chan error)
	go func() {
		exitErr := cmd.Run()
		exitCode = processExitCode(exitErr)
		done <- exitErr
	}()

	select {
	case <-time.After(duration):
		killErr := cmd.Process.Kill()
		if killErr != nil {
			fmt.Printf("failed to kill (pid=%d): %v\n", cmd.Process.Pid, killErr)
		}
		timedOut = true
		break
	case err = <-done:
		break
	}
	output = outputBuffer.String()
	return
}

var ErrCmdTimeout = fmt.Errorf("command timed out")

func runCommandWithOutputAndTimeout(cmd *exec.Cmd, timeout time.Duration) (output string, exitCode int, err error) {
	var timedOut bool
	output, exitCode, timedOut, err = runCommandWithOutputForDuration(cmd, timeout)
	if timedOut {
		err = ErrCmdTimeout
	}
	return
}

func runCommand(cmd *exec.Cmd) (exitCode int, err error) {
	exitCode = 0
	err = cmd.Run()
	exitCode = processExitCode(err)
	return
}

func runCommandPipelineWithOutput(cmds ...*exec.Cmd) (output string, exitCode int, err error) {
	if len(cmds) < 2 {
		return "", 0, errors.New("pipeline does not have multiple cmds")
	}

	// connect stdin of each cmd to stdout pipe of previous cmd
	for i, cmd := range cmds {
		if i > 0 {
			prevCmd := cmds[i-1]
			cmd.Stdin, err = prevCmd.StdoutPipe()

			if err != nil {
				return "", 0, fmt.Errorf("cannot set stdout pipe for %s: %v", cmd.Path, err)
			}
		}
	}

	// start all cmds except the last
	for _, cmd := range cmds[:len(cmds)-1] {
		if err = cmd.Start(); err != nil {
			return "", 0, fmt.Errorf("starting %s failed with error: %v", cmd.Path, err)
		}
	}

	defer func() {
		// wait all cmds except the last to release their resources
		for _, cmd := range cmds[:len(cmds)-1] {
			cmd.Wait()
		}
	}()

	// wait on last cmd
	return runCommandWithOutput(cmds[len(cmds)-1])
}

func unmarshalJSON(data []byte, result interface{}) error {
	if err := json.Unmarshal(data, result); err != nil {
		return err
	}

	return nil
}

func convertSliceOfStringsToMap(input []string) map[string]struct{} {
	output := make(map[string]struct{})
	for _, v := range input {
		output[v] = struct{}{}
	}
	return output
}

func waitForContainer(contID string, args ...string) error {
	args = append([]string{"run", "--name", contID}, args...)
	cmd := exec.Command(dockerBinary, args...)
	if _, err := runCommand(cmd); err != nil {
		return err
	}

	if err := waitRun(contID); err != nil {
		return err
	}

	return nil
}

func waitRun(contID string) error {
	return waitInspect(contID, "{{.State.Running}}", "true", 5)
}

func waitInspect(name, expr, expected string, timeout int) error {
	after := time.After(time.Duration(timeout) * time.Second)

	for {
		cmd := exec.Command(dockerBinary, "inspect", "-f", expr, name)
		out, _, err := runCommandWithOutput(cmd)
		if err != nil {
			if !strings.Contains(out, "No such") {
				return fmt.Errorf("error executing docker inspect: %v\n%s", err, out)
			}
			select {
			case <-after:
				return err
			default:
				time.Sleep(10 * time.Millisecond)
				continue
			}
		}

		out = strings.TrimSpace(out)
		if out == expected {
			break
		}

		select {
		case <-after:
			return fmt.Errorf("condition \"%q == %q\" not true in time", out, expected)
		default:
		}

		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

func compareDirectoryEntries(e1 []os.FileInfo, e2 []os.FileInfo) error {
	var (
		e1Entries = make(map[string]struct{})
		e2Entries = make(map[string]struct{})
	)
	for _, e := range e1 {
		e1Entries[e.Name()] = struct{}{}
	}
	for _, e := range e2 {
		e2Entries[e.Name()] = struct{}{}
	}
	if !reflect.DeepEqual(e1Entries, e2Entries) {
		return fmt.Errorf("entries differ")
	}
	return nil
}

func ListTar(f io.Reader) ([]string, error) {
	tr := tar.NewReader(f)
	var entries []string

	for {
		th, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			return entries, nil
		}
		if err != nil {
			return entries, err
		}
		entries = append(entries, th.Name)
	}
}

type FileServer struct {
	*httptest.Server
}

// randomUnixTmpDirPath provides a temporary unix path with rand string appended.
// does not create or checks if it exists.
func randomUnixTmpDirPath(s string) string {
	return path.Join("/tmp", fmt.Sprintf("%s.%s", s, stringutils.GenerateRandomAlphaOnlyString(10)))
}

// Reads chunkSize bytes from reader after every interval.
// Returns total read bytes.
func consumeWithSpeed(reader io.Reader, chunkSize int, interval time.Duration, stop chan bool) (n int, err error) {
	buffer := make([]byte, chunkSize)
	for {
		select {
		case <-stop:
			return
		default:
			var readBytes int
			readBytes, err = reader.Read(buffer)
			n += readBytes
			if err != nil {
				if err == io.EOF {
					err = nil
				}
				return
			}
			time.Sleep(interval)
		}
	}
}

// Parses 'procCgroupData', which is output of '/proc/<pid>/cgroup', and returns
// a map which cgroup name as key and path as value.
func parseCgroupPaths(procCgroupData string) map[string]string {
	cgroupPaths := map[string]string{}
	for _, line := range strings.Split(procCgroupData, "\n") {
		parts := strings.Split(line, ":")
		if len(parts) != 3 {
			continue
		}
		cgroupPaths[parts[1]] = parts[2]
	}
	return cgroupPaths
}

type channelBuffer struct {
	c chan []byte
}

func (c *channelBuffer) Write(b []byte) (int, error) {
	c.c <- b
	return len(b), nil
}

func (c *channelBuffer) Close() error {
	close(c.c)
	return nil
}

func (c *channelBuffer) ReadTimeout(p []byte, n time.Duration) (int, error) {
	select {
	case b := <-c.c:
		return copy(p[0:], b), nil
	case <-time.After(n):
		return -1, fmt.Errorf("timeout reading from channel")
	}
}
