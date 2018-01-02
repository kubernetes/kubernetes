package testutil

import (
	"archive/tar"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/pkg/stringutils"
	"github.com/docker/docker/pkg/system"
)

// IsKilled process the specified error and returns whether the process was killed or not.
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
	out, err := cmd.CombinedOutput()
	exitCode = system.ProcessExitCode(err)
	output = string(out)
	return
}

// RunCommandPipelineWithOutput runs the array of commands with the output
// of each pipelined with the following (like cmd1 | cmd2 | cmd3 would do).
// It returns the final output, the exitCode different from 0 and the error
// if something bad happened.
func RunCommandPipelineWithOutput(cmds ...*exec.Cmd) (output string, exitCode int, err error) {
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
		var pipeErrMsgs []string
		// wait all cmds except the last to release their resources
		for _, cmd := range cmds[:len(cmds)-1] {
			if pipeErr := cmd.Wait(); pipeErr != nil {
				pipeErrMsgs = append(pipeErrMsgs, fmt.Sprintf("command %s failed with error: %v", cmd.Path, pipeErr))
			}
		}
		if len(pipeErrMsgs) > 0 && err == nil {
			err = fmt.Errorf("pipelineError from Wait: %v", strings.Join(pipeErrMsgs, ", "))
		}
	}()

	// wait on last cmd
	return runCommandWithOutput(cmds[len(cmds)-1])
}

// ConvertSliceOfStringsToMap converts a slices of string in a map
// with the strings as key and an empty string as values.
func ConvertSliceOfStringsToMap(input []string) map[string]struct{} {
	output := make(map[string]struct{})
	for _, v := range input {
		output[v] = struct{}{}
	}
	return output
}

// CompareDirectoryEntries compares two sets of FileInfo (usually taken from a directory)
// and returns an error if different.
func CompareDirectoryEntries(e1 []os.FileInfo, e2 []os.FileInfo) error {
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

// ListTar lists the entries of a tar.
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

// RandomTmpDirPath provides a temporary path with rand string appended.
// does not create or checks if it exists.
func RandomTmpDirPath(s string, platform string) string {
	tmp := "/tmp"
	if platform == "windows" {
		tmp = os.Getenv("TEMP")
	}
	path := filepath.Join(tmp, fmt.Sprintf("%s.%s", s, stringutils.GenerateRandomAlphaOnlyString(10)))
	if platform == "windows" {
		return filepath.FromSlash(path) // Using \
	}
	return filepath.ToSlash(path) // Using /
}

// ConsumeWithSpeed reads chunkSize bytes from reader before sleeping
// for interval duration. Returns total read bytes. Send true to the
// stop channel to return before reading to EOF on the reader.
func ConsumeWithSpeed(reader io.Reader, chunkSize int, interval time.Duration, stop chan bool) (n int, err error) {
	buffer := make([]byte, chunkSize)
	for {
		var readBytes int
		readBytes, err = reader.Read(buffer)
		n += readBytes
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return
		}
		select {
		case <-stop:
			return
		case <-time.After(interval):
		}
	}
}

// ParseCgroupPaths parses 'procCgroupData', which is output of '/proc/<pid>/cgroup', and returns
// a map which cgroup name as key and path as value.
func ParseCgroupPaths(procCgroupData string) map[string]string {
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

// ChannelBuffer holds a chan of byte array that can be populate in a goroutine.
type ChannelBuffer struct {
	C chan []byte
}

// Write implements Writer.
func (c *ChannelBuffer) Write(b []byte) (int, error) {
	c.C <- b
	return len(b), nil
}

// Close closes the go channel.
func (c *ChannelBuffer) Close() error {
	close(c.C)
	return nil
}

// ReadTimeout reads the content of the channel in the specified byte array with
// the specified duration as timeout.
func (c *ChannelBuffer) ReadTimeout(p []byte, n time.Duration) (int, error) {
	select {
	case b := <-c.C:
		return copy(p[0:], b), nil
	case <-time.After(n):
		return -1, fmt.Errorf("timeout reading from channel")
	}
}

// ReadBody read the specified ReadCloser content and returns it
func ReadBody(b io.ReadCloser) ([]byte, error) {
	defer b.Close()
	return ioutil.ReadAll(b)
}
