package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/daemon"
	"github.com/docker/docker/integration-cli/registry"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

// Deprecated
func daemonHost() string {
	return request.DaemonHost()
}

func deleteImages(images ...string) error {
	args := []string{dockerBinary, "rmi", "-f"}
	return icmd.RunCmd(icmd.Cmd{Command: append(args, images...)}).Error
}

// Deprecated: use cli.Docker or cli.DockerCmd
func dockerCmdWithError(args ...string) (string, int, error) {
	result := cli.Docker(cli.Args(args...))
	if result.Error != nil {
		return result.Combined(), result.ExitCode, result.Compare(icmd.Success)
	}
	return result.Combined(), result.ExitCode, result.Error
}

// Deprecated: use cli.Docker or cli.DockerCmd
func dockerCmd(c *check.C, args ...string) (string, int) {
	result := cli.DockerCmd(c, args...)
	return result.Combined(), result.ExitCode
}

// Deprecated: use cli.Docker or cli.DockerCmd
func dockerCmdWithResult(args ...string) *icmd.Result {
	return cli.Docker(cli.Args(args...))
}

func findContainerIP(c *check.C, id string, network string) string {
	out, _ := dockerCmd(c, "inspect", fmt.Sprintf("--format='{{ .NetworkSettings.Networks.%s.IPAddress }}'", network), id)
	return strings.Trim(out, " \r\n'")
}

func getContainerCount(c *check.C) int {
	const containers = "Containers:"

	result := icmd.RunCommand(dockerBinary, "info")
	result.Assert(c, icmd.Success)

	lines := strings.Split(result.Combined(), "\n")
	for _, line := range lines {
		if strings.Contains(line, containers) {
			output := strings.TrimSpace(line)
			output = strings.TrimLeft(output, containers)
			output = strings.Trim(output, " ")
			containerCount, err := strconv.Atoi(output)
			c.Assert(err, checker.IsNil)
			return containerCount
		}
	}
	return 0
}

func inspectFieldAndUnmarshall(c *check.C, name, field string, output interface{}) {
	str := inspectFieldJSON(c, name, field)
	err := json.Unmarshal([]byte(str), output)
	if c != nil {
		c.Assert(err, check.IsNil, check.Commentf("failed to unmarshal: %v", err))
	}
}

// Deprecated: use cli.Inspect
func inspectFilter(name, filter string) (string, error) {
	format := fmt.Sprintf("{{%s}}", filter)
	result := icmd.RunCommand(dockerBinary, "inspect", "-f", format, name)
	if result.Error != nil || result.ExitCode != 0 {
		return "", fmt.Errorf("failed to inspect %s: %s", name, result.Combined())
	}
	return strings.TrimSpace(result.Combined()), nil
}

// Deprecated: use cli.Inspect
func inspectFieldWithError(name, field string) (string, error) {
	return inspectFilter(name, fmt.Sprintf(".%s", field))
}

// Deprecated: use cli.Inspect
func inspectField(c *check.C, name, field string) string {
	out, err := inspectFilter(name, fmt.Sprintf(".%s", field))
	if c != nil {
		c.Assert(err, check.IsNil)
	}
	return out
}

// Deprecated: use cli.Inspect
func inspectFieldJSON(c *check.C, name, field string) string {
	out, err := inspectFilter(name, fmt.Sprintf("json .%s", field))
	if c != nil {
		c.Assert(err, check.IsNil)
	}
	return out
}

// Deprecated: use cli.Inspect
func inspectFieldMap(c *check.C, name, path, field string) string {
	out, err := inspectFilter(name, fmt.Sprintf("index .%s %q", path, field))
	if c != nil {
		c.Assert(err, check.IsNil)
	}
	return out
}

// Deprecated: use cli.Inspect
func inspectMountSourceField(name, destination string) (string, error) {
	m, err := inspectMountPoint(name, destination)
	if err != nil {
		return "", err
	}
	return m.Source, nil
}

// Deprecated: use cli.Inspect
func inspectMountPoint(name, destination string) (types.MountPoint, error) {
	out, err := inspectFilter(name, "json .Mounts")
	if err != nil {
		return types.MountPoint{}, err
	}

	return inspectMountPointJSON(out, destination)
}

var errMountNotFound = errors.New("mount point not found")

// Deprecated: use cli.Inspect
func inspectMountPointJSON(j, destination string) (types.MountPoint, error) {
	var mp []types.MountPoint
	if err := json.Unmarshal([]byte(j), &mp); err != nil {
		return types.MountPoint{}, err
	}

	var m *types.MountPoint
	for _, c := range mp {
		if c.Destination == destination {
			m = &c
			break
		}
	}

	if m == nil {
		return types.MountPoint{}, errMountNotFound
	}

	return *m, nil
}

// Deprecated: use cli.Inspect
func inspectImage(c *check.C, name, filter string) string {
	args := []string{"inspect", "--type", "image"}
	if filter != "" {
		format := fmt.Sprintf("{{%s}}", filter)
		args = append(args, "-f", format)
	}
	args = append(args, name)
	result := icmd.RunCommand(dockerBinary, args...)
	result.Assert(c, icmd.Success)
	return strings.TrimSpace(result.Combined())
}

func getIDByName(c *check.C, name string) string {
	id, err := inspectFieldWithError(name, "Id")
	c.Assert(err, checker.IsNil)
	return id
}

// Deprecated: use cli.Build
func buildImageSuccessfully(c *check.C, name string, cmdOperators ...cli.CmdOperator) {
	buildImage(name, cmdOperators...).Assert(c, icmd.Success)
}

// Deprecated: use cli.Build
func buildImage(name string, cmdOperators ...cli.CmdOperator) *icmd.Result {
	return cli.Docker(cli.Build(name), cmdOperators...)
}

// Deprecated: use trustedcmd
func trustedBuild(cmd *icmd.Cmd) func() {
	trustedCmd(cmd)
	return nil
}

// Write `content` to the file at path `dst`, creating it if necessary,
// as well as any missing directories.
// The file is truncated if it already exists.
// Fail the test when error occurs.
func writeFile(dst, content string, c *check.C) {
	// Create subdirectories if necessary
	c.Assert(os.MkdirAll(path.Dir(dst), 0700), check.IsNil)
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0700)
	c.Assert(err, check.IsNil)
	defer f.Close()
	// Write content (truncate if it exists)
	_, err = io.Copy(f, strings.NewReader(content))
	c.Assert(err, check.IsNil)
}

// Return the contents of file at path `src`.
// Fail the test when error occurs.
func readFile(src string, c *check.C) (content string) {
	data, err := ioutil.ReadFile(src)
	c.Assert(err, check.IsNil)

	return string(data)
}

func containerStorageFile(containerID, basename string) string {
	return filepath.Join(testEnv.ContainerStoragePath(), containerID, basename)
}

// docker commands that use this function must be run with the '-d' switch.
func runCommandAndReadContainerFile(c *check.C, filename string, command string, args ...string) []byte {
	result := icmd.RunCommand(command, args...)
	result.Assert(c, icmd.Success)
	contID := strings.TrimSpace(result.Combined())
	if err := waitRun(contID); err != nil {
		c.Fatalf("%v: %q", contID, err)
	}
	return readContainerFile(c, contID, filename)
}

func readContainerFile(c *check.C, containerID, filename string) []byte {
	f, err := os.Open(containerStorageFile(containerID, filename))
	c.Assert(err, checker.IsNil)
	defer f.Close()

	content, err := ioutil.ReadAll(f)
	c.Assert(err, checker.IsNil)
	return content
}

func readContainerFileWithExec(c *check.C, containerID, filename string) []byte {
	result := icmd.RunCommand(dockerBinary, "exec", containerID, "cat", filename)
	result.Assert(c, icmd.Success)
	return []byte(result.Combined())
}

// daemonTime provides the current time on the daemon host
func daemonTime(c *check.C) time.Time {
	if testEnv.LocalDaemon() {
		return time.Now()
	}

	status, body, err := request.SockRequest("GET", "/info", nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	type infoJSON struct {
		SystemTime string
	}
	var info infoJSON
	err = json.Unmarshal(body, &info)
	c.Assert(err, check.IsNil, check.Commentf("unable to unmarshal GET /info response"))

	dt, err := time.Parse(time.RFC3339Nano, info.SystemTime)
	c.Assert(err, check.IsNil, check.Commentf("invalid time format in GET /info response"))
	return dt
}

// daemonUnixTime returns the current time on the daemon host with nanoseconds precision.
// It return the time formatted how the client sends timestamps to the server.
func daemonUnixTime(c *check.C) string {
	return parseEventTime(daemonTime(c))
}

func parseEventTime(t time.Time) string {
	return fmt.Sprintf("%d.%09d", t.Unix(), int64(t.Nanosecond()))
}

func setupRegistry(c *check.C, schema1 bool, auth, tokenURL string) *registry.V2 {
	reg, err := registry.NewV2(schema1, auth, tokenURL, privateRegistryURL)
	c.Assert(err, check.IsNil)

	// Wait for registry to be ready to serve requests.
	for i := 0; i != 50; i++ {
		if err = reg.Ping(); err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	c.Assert(err, check.IsNil, check.Commentf("Timeout waiting for test registry to become available: %v", err))
	return reg
}

func setupNotary(c *check.C) *testNotary {
	ts, err := newTestNotary(c)
	c.Assert(err, check.IsNil)

	return ts
}

// appendBaseEnv appends the minimum set of environment variables to exec the
// docker cli binary for testing with correct configuration to the given env
// list.
func appendBaseEnv(isTLS bool, env ...string) []string {
	preserveList := []string{
		// preserve remote test host
		"DOCKER_HOST",

		// windows: requires preserving SystemRoot, otherwise dial tcp fails
		// with "GetAddrInfoW: A non-recoverable error occurred during a database lookup."
		"SystemRoot",

		// testing help text requires the $PATH to dockerd is set
		"PATH",
	}
	if isTLS {
		preserveList = append(preserveList, "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
	}

	for _, key := range preserveList {
		if val := os.Getenv(key); val != "" {
			env = append(env, fmt.Sprintf("%s=%s", key, val))
		}
	}
	return env
}

func createTmpFile(c *check.C, content string) string {
	f, err := ioutil.TempFile("", "testfile")
	c.Assert(err, check.IsNil)

	filename := f.Name()

	err = ioutil.WriteFile(filename, []byte(content), 0644)
	c.Assert(err, check.IsNil)

	return filename
}

// waitRun will wait for the specified container to be running, maximum 5 seconds.
// Deprecated: use cli.WaitFor
func waitRun(contID string) error {
	return waitInspect(contID, "{{.State.Running}}", "true", 5*time.Second)
}

// waitInspect will wait for the specified container to have the specified string
// in the inspect output. It will wait until the specified timeout (in seconds)
// is reached.
// Deprecated: use cli.WaitFor
func waitInspect(name, expr, expected string, timeout time.Duration) error {
	return waitInspectWithArgs(name, expr, expected, timeout)
}

// Deprecated: use cli.WaitFor
func waitInspectWithArgs(name, expr, expected string, timeout time.Duration, arg ...string) error {
	return daemon.WaitInspectWithArgs(dockerBinary, name, expr, expected, timeout, arg...)
}

func getInspectBody(c *check.C, version, id string) []byte {
	endpoint := fmt.Sprintf("/%s/containers/%s/json", version, id)
	status, body, err := request.SockRequest("GET", endpoint, nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)
	return body
}

// Run a long running idle task in a background container using the
// system-specific default image and command.
func runSleepingContainer(c *check.C, extraArgs ...string) string {
	return runSleepingContainerInImage(c, defaultSleepImage, extraArgs...)
}

// Run a long running idle task in a background container using the specified
// image and the system-specific command.
func runSleepingContainerInImage(c *check.C, image string, extraArgs ...string) string {
	args := []string{"run", "-d"}
	args = append(args, extraArgs...)
	args = append(args, image)
	args = append(args, sleepCommandForDaemonPlatform()...)
	return strings.TrimSpace(cli.DockerCmd(c, args...).Combined())
}

// minimalBaseImage returns the name of the minimal base image for the current
// daemon platform.
func minimalBaseImage() string {
	return testEnv.MinimalBaseImage()
}

func getGoroutineNumber() (int, error) {
	i := struct {
		NGoroutines int
	}{}
	status, b, err := request.SockRequest("GET", "/info", nil, daemonHost())
	if err != nil {
		return 0, err
	}
	if status != http.StatusOK {
		return 0, fmt.Errorf("http status code: %d", status)
	}
	if err := json.Unmarshal(b, &i); err != nil {
		return 0, err
	}
	return i.NGoroutines, nil
}

func waitForGoroutines(expected int) error {
	t := time.After(30 * time.Second)
	for {
		select {
		case <-t:
			n, err := getGoroutineNumber()
			if err != nil {
				return err
			}
			if n > expected {
				return fmt.Errorf("leaked goroutines: expected less than or equal to %d, got: %d", expected, n)
			}
		default:
			n, err := getGoroutineNumber()
			if err != nil {
				return err
			}
			if n <= expected {
				return nil
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}

// getErrorMessage returns the error message from an error API response
func getErrorMessage(c *check.C, body []byte) string {
	var resp types.ErrorResponse
	c.Assert(json.Unmarshal(body, &resp), check.IsNil)
	return strings.TrimSpace(resp.Message)
}

func waitAndAssert(c *check.C, timeout time.Duration, f checkF, checker check.Checker, args ...interface{}) {
	after := time.After(timeout)
	for {
		v, comment := f(c)
		assert, _ := checker.Check(append([]interface{}{v}, args...), checker.Info().Params)
		select {
		case <-after:
			assert = true
		default:
		}
		if assert {
			if comment != nil {
				args = append(args, comment)
			}
			c.Assert(v, checker, args...)
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
}

type checkF func(*check.C) (interface{}, check.CommentInterface)
type reducer func(...interface{}) interface{}

func reducedCheck(r reducer, funcs ...checkF) checkF {
	return func(c *check.C) (interface{}, check.CommentInterface) {
		var values []interface{}
		var comments []string
		for _, f := range funcs {
			v, comment := f(c)
			values = append(values, v)
			if comment != nil {
				comments = append(comments, comment.CheckCommentString())
			}
		}
		return r(values...), check.Commentf("%v", strings.Join(comments, ", "))
	}
}

func sumAsIntegers(vals ...interface{}) interface{} {
	var s int
	for _, v := range vals {
		s += v.(int)
	}
	return s
}
