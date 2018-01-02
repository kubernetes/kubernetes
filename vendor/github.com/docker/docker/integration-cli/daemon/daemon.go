package daemon

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/testutil"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/go-check/check"
	"github.com/pkg/errors"
)

type testingT interface {
	logT
	Fatalf(string, ...interface{})
}

type logT interface {
	Logf(string, ...interface{})
}

// SockRoot holds the path of the default docker integration daemon socket
var SockRoot = filepath.Join(os.TempDir(), "docker-integration")

var errDaemonNotStarted = errors.New("daemon not started")

// Daemon represents a Docker daemon for the testing framework.
type Daemon struct {
	GlobalFlags       []string
	Root              string
	Folder            string
	Wait              chan error
	UseDefaultHost    bool
	UseDefaultTLSHost bool

	id             string
	logFile        *os.File
	stdin          io.WriteCloser
	stdout, stderr io.ReadCloser
	cmd            *exec.Cmd
	storageDriver  string
	userlandProxy  bool
	execRoot       string
	experimental   bool
	dockerBinary   string
	dockerdBinary  string
	log            logT
}

// Config holds docker daemon integration configuration
type Config struct {
	Experimental bool
}

type clientConfig struct {
	transport *http.Transport
	scheme    string
	addr      string
}

// New returns a Daemon instance to be used for testing.
// This will create a directory such as d123456789 in the folder specified by $DOCKER_INTEGRATION_DAEMON_DEST or $DEST.
// The daemon will not automatically start.
func New(t testingT, dockerBinary string, dockerdBinary string, config Config) *Daemon {
	dest := os.Getenv("DOCKER_INTEGRATION_DAEMON_DEST")
	if dest == "" {
		dest = os.Getenv("DEST")
	}
	if dest == "" {
		t.Fatalf("Please set the DOCKER_INTEGRATION_DAEMON_DEST or the DEST environment variable")
	}

	if err := os.MkdirAll(SockRoot, 0700); err != nil {
		t.Fatalf("could not create daemon socket root")
	}

	id := fmt.Sprintf("d%s", stringid.TruncateID(stringid.GenerateRandomID()))
	dir := filepath.Join(dest, id)
	daemonFolder, err := filepath.Abs(dir)
	if err != nil {
		t.Fatalf("Could not make %q an absolute path", dir)
	}
	daemonRoot := filepath.Join(daemonFolder, "root")

	if err := os.MkdirAll(daemonRoot, 0755); err != nil {
		t.Fatalf("Could not create daemon root %q", dir)
	}

	userlandProxy := true
	if env := os.Getenv("DOCKER_USERLANDPROXY"); env != "" {
		if val, err := strconv.ParseBool(env); err != nil {
			userlandProxy = val
		}
	}

	return &Daemon{
		id:            id,
		Folder:        daemonFolder,
		Root:          daemonRoot,
		storageDriver: os.Getenv("DOCKER_GRAPHDRIVER"),
		userlandProxy: userlandProxy,
		execRoot:      filepath.Join(os.TempDir(), "docker-execroot", id),
		dockerBinary:  dockerBinary,
		dockerdBinary: dockerdBinary,
		experimental:  config.Experimental,
		log:           t,
	}
}

// RootDir returns the root directory of the daemon.
func (d *Daemon) RootDir() string {
	return d.Root
}

// ID returns the generated id of the daemon
func (d *Daemon) ID() string {
	return d.id
}

// StorageDriver returns the configured storage driver of the daemon
func (d *Daemon) StorageDriver() string {
	return d.storageDriver
}

// CleanupExecRoot cleans the daemon exec root (network namespaces, ...)
func (d *Daemon) CleanupExecRoot(c *check.C) {
	cleanupExecRoot(c, d.execRoot)
}

func (d *Daemon) getClientConfig() (*clientConfig, error) {
	var (
		transport *http.Transport
		scheme    string
		addr      string
		proto     string
	)
	if d.UseDefaultTLSHost {
		option := &tlsconfig.Options{
			CAFile:   "fixtures/https/ca.pem",
			CertFile: "fixtures/https/client-cert.pem",
			KeyFile:  "fixtures/https/client-key.pem",
		}
		tlsConfig, err := tlsconfig.Client(*option)
		if err != nil {
			return nil, err
		}
		transport = &http.Transport{
			TLSClientConfig: tlsConfig,
		}
		addr = fmt.Sprintf("%s:%d", opts.DefaultHTTPHost, opts.DefaultTLSHTTPPort)
		scheme = "https"
		proto = "tcp"
	} else if d.UseDefaultHost {
		addr = opts.DefaultUnixSocket
		proto = "unix"
		scheme = "http"
		transport = &http.Transport{}
	} else {
		addr = d.sockPath()
		proto = "unix"
		scheme = "http"
		transport = &http.Transport{}
	}

	if err := sockets.ConfigureTransport(transport, proto, addr); err != nil {
		return nil, err
	}
	transport.DisableKeepAlives = true

	return &clientConfig{
		transport: transport,
		scheme:    scheme,
		addr:      addr,
	}, nil
}

// Start starts the daemon and return once it is ready to receive requests.
func (d *Daemon) Start(t testingT, args ...string) {
	if err := d.StartWithError(args...); err != nil {
		t.Fatalf("Error starting daemon with arguments: %v", args)
	}
}

// StartWithError starts the daemon and return once it is ready to receive requests.
// It returns an error in case it couldn't start.
func (d *Daemon) StartWithError(args ...string) error {
	logFile, err := os.OpenFile(filepath.Join(d.Folder, "docker.log"), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0600)
	if err != nil {
		return errors.Wrapf(err, "[%s] Could not create %s/docker.log", d.id, d.Folder)
	}

	return d.StartWithLogFile(logFile, args...)
}

// StartWithLogFile will start the daemon and attach its streams to a given file.
func (d *Daemon) StartWithLogFile(out *os.File, providedArgs ...string) error {
	dockerdBinary, err := exec.LookPath(d.dockerdBinary)
	if err != nil {
		return errors.Wrapf(err, "[%s] could not find docker binary in $PATH", d.id)
	}
	args := append(d.GlobalFlags,
		"--containerd", "/var/run/docker/libcontainerd/docker-containerd.sock",
		"--data-root", d.Root,
		"--exec-root", d.execRoot,
		"--pidfile", fmt.Sprintf("%s/docker.pid", d.Folder),
		fmt.Sprintf("--userland-proxy=%t", d.userlandProxy),
	)
	if d.experimental {
		args = append(args, "--experimental", "--init")
	}
	if !(d.UseDefaultHost || d.UseDefaultTLSHost) {
		args = append(args, []string{"--host", d.Sock()}...)
	}
	if root := os.Getenv("DOCKER_REMAP_ROOT"); root != "" {
		args = append(args, []string{"--userns-remap", root}...)
	}

	// If we don't explicitly set the log-level or debug flag(-D) then
	// turn on debug mode
	foundLog := false
	foundSd := false
	for _, a := range providedArgs {
		if strings.Contains(a, "--log-level") || strings.Contains(a, "-D") || strings.Contains(a, "--debug") {
			foundLog = true
		}
		if strings.Contains(a, "--storage-driver") {
			foundSd = true
		}
	}
	if !foundLog {
		args = append(args, "--debug")
	}
	if d.storageDriver != "" && !foundSd {
		args = append(args, "--storage-driver", d.storageDriver)
	}

	args = append(args, providedArgs...)
	d.cmd = exec.Command(dockerdBinary, args...)
	d.cmd.Env = append(os.Environ(), "DOCKER_SERVICE_PREFER_OFFLINE_IMAGE=1")
	d.cmd.Stdout = out
	d.cmd.Stderr = out
	d.logFile = out

	if err := d.cmd.Start(); err != nil {
		return errors.Errorf("[%s] could not start daemon container: %v", d.id, err)
	}

	wait := make(chan error)

	go func() {
		wait <- d.cmd.Wait()
		d.log.Logf("[%s] exiting daemon", d.id)
		close(wait)
	}()

	d.Wait = wait

	tick := time.Tick(500 * time.Millisecond)
	// make sure daemon is ready to receive requests
	startTime := time.Now().Unix()
	for {
		d.log.Logf("[%s] waiting for daemon to start", d.id)
		if time.Now().Unix()-startTime > 5 {
			// After 5 seconds, give up
			return errors.Errorf("[%s] Daemon exited and never started", d.id)
		}
		select {
		case <-time.After(2 * time.Second):
			return errors.Errorf("[%s] timeout: daemon does not respond", d.id)
		case <-tick:
			clientConfig, err := d.getClientConfig()
			if err != nil {
				return err
			}

			client := &http.Client{
				Transport: clientConfig.transport,
			}

			req, err := http.NewRequest("GET", "/_ping", nil)
			if err != nil {
				return errors.Wrapf(err, "[%s] could not create new request", d.id)
			}
			req.URL.Host = clientConfig.addr
			req.URL.Scheme = clientConfig.scheme
			resp, err := client.Do(req)
			if err != nil {
				continue
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				d.log.Logf("[%s] received status != 200 OK: %s\n", d.id, resp.Status)
			}
			d.log.Logf("[%s] daemon started\n", d.id)
			d.Root, err = d.queryRootDir()
			if err != nil {
				return errors.Errorf("[%s] error querying daemon for root directory: %v", d.id, err)
			}
			return nil
		case <-d.Wait:
			return errors.Errorf("[%s] Daemon exited during startup", d.id)
		}
	}
}

// StartWithBusybox will first start the daemon with Daemon.Start()
// then save the busybox image from the main daemon and load it into this Daemon instance.
func (d *Daemon) StartWithBusybox(t testingT, arg ...string) {
	d.Start(t, arg...)
	if err := d.LoadBusybox(); err != nil {
		t.Fatalf("Error loading busybox image to current daemon: %s\n%v", d.id, err)
	}
}

// Kill will send a SIGKILL to the daemon
func (d *Daemon) Kill() error {
	if d.cmd == nil || d.Wait == nil {
		return errDaemonNotStarted
	}

	defer func() {
		d.logFile.Close()
		d.cmd = nil
	}()

	if err := d.cmd.Process.Kill(); err != nil {
		return err
	}

	if err := os.Remove(fmt.Sprintf("%s/docker.pid", d.Folder)); err != nil {
		return err
	}

	return nil
}

// Pid returns the pid of the daemon
func (d *Daemon) Pid() int {
	return d.cmd.Process.Pid
}

// Interrupt stops the daemon by sending it an Interrupt signal
func (d *Daemon) Interrupt() error {
	return d.Signal(os.Interrupt)
}

// Signal sends the specified signal to the daemon if running
func (d *Daemon) Signal(signal os.Signal) error {
	if d.cmd == nil || d.Wait == nil {
		return errDaemonNotStarted
	}
	return d.cmd.Process.Signal(signal)
}

// DumpStackAndQuit sends SIGQUIT to the daemon, which triggers it to dump its
// stack to its log file and exit
// This is used primarily for gathering debug information on test timeout
func (d *Daemon) DumpStackAndQuit() {
	if d.cmd == nil || d.cmd.Process == nil {
		return
	}
	SignalDaemonDump(d.cmd.Process.Pid)
}

// Stop will send a SIGINT every second and wait for the daemon to stop.
// If it times out, a SIGKILL is sent.
// Stop will not delete the daemon directory. If a purged daemon is needed,
// instantiate a new one with NewDaemon.
// If an error occurs while starting the daemon, the test will fail.
func (d *Daemon) Stop(t testingT) {
	err := d.StopWithError()
	if err != nil {
		if err != errDaemonNotStarted {
			t.Fatalf("Error while stopping the daemon %s : %v", d.id, err)
		} else {
			t.Logf("Daemon %s is not started", d.id)
		}
	}
}

// StopWithError will send a SIGINT every second and wait for the daemon to stop.
// If it timeouts, a SIGKILL is sent.
// Stop will not delete the daemon directory. If a purged daemon is needed,
// instantiate a new one with NewDaemon.
func (d *Daemon) StopWithError() error {
	if d.cmd == nil || d.Wait == nil {
		return errDaemonNotStarted
	}

	defer func() {
		d.logFile.Close()
		d.cmd = nil
	}()

	i := 1
	tick := time.Tick(time.Second)

	if err := d.cmd.Process.Signal(os.Interrupt); err != nil {
		if strings.Contains(err.Error(), "os: process already finished") {
			return errDaemonNotStarted
		}
		return errors.Errorf("could not send signal: %v", err)
	}
out1:
	for {
		select {
		case err := <-d.Wait:
			return err
		case <-time.After(20 * time.Second):
			// time for stopping jobs and run onShutdown hooks
			d.log.Logf("[%s] daemon started", d.id)
			break out1
		}
	}

out2:
	for {
		select {
		case err := <-d.Wait:
			return err
		case <-tick:
			i++
			if i > 5 {
				d.log.Logf("tried to interrupt daemon for %d times, now try to kill it", i)
				break out2
			}
			d.log.Logf("Attempt #%d: daemon is still running with pid %d", i, d.cmd.Process.Pid)
			if err := d.cmd.Process.Signal(os.Interrupt); err != nil {
				return errors.Errorf("could not send signal: %v", err)
			}
		}
	}

	if err := d.cmd.Process.Kill(); err != nil {
		d.log.Logf("Could not kill daemon: %v", err)
		return err
	}

	if err := os.Remove(fmt.Sprintf("%s/docker.pid", d.Folder)); err != nil {
		return err
	}

	return nil
}

// Restart will restart the daemon by first stopping it and the starting it.
// If an error occurs while starting the daemon, the test will fail.
func (d *Daemon) Restart(t testingT, args ...string) {
	d.Stop(t)
	d.handleUserns()
	d.Start(t, args...)
}

// RestartWithError will restart the daemon by first stopping it and then starting it.
func (d *Daemon) RestartWithError(arg ...string) error {
	if err := d.StopWithError(); err != nil {
		return err
	}
	d.handleUserns()
	return d.StartWithError(arg...)
}

func (d *Daemon) handleUserns() {
	// in the case of tests running a user namespace-enabled daemon, we have resolved
	// d.Root to be the actual final path of the graph dir after the "uid.gid" of
	// remapped root is added--we need to subtract it from the path before calling
	// start or else we will continue making subdirectories rather than truly restarting
	// with the same location/root:
	if root := os.Getenv("DOCKER_REMAP_ROOT"); root != "" {
		d.Root = filepath.Dir(d.Root)
	}
}

// LoadBusybox will load the stored busybox into a newly started daemon
func (d *Daemon) LoadBusybox() error {
	bb := filepath.Join(d.Folder, "busybox.tar")
	if _, err := os.Stat(bb); err != nil {
		if !os.IsNotExist(err) {
			return errors.Errorf("unexpected error on busybox.tar stat: %v", err)
		}
		// saving busybox image from main daemon
		if out, err := exec.Command(d.dockerBinary, "save", "--output", bb, "busybox:latest").CombinedOutput(); err != nil {
			imagesOut, _ := exec.Command(d.dockerBinary, "images", "--format", "{{ .Repository }}:{{ .Tag }}").CombinedOutput()
			return errors.Errorf("could not save busybox image: %s\n%s", string(out), strings.TrimSpace(string(imagesOut)))
		}
	}
	// loading busybox image to this daemon
	if out, err := d.Cmd("load", "--input", bb); err != nil {
		return errors.Errorf("could not load busybox image: %s", out)
	}
	if err := os.Remove(bb); err != nil {
		return err
	}
	return nil
}

func (d *Daemon) queryRootDir() (string, error) {
	// update daemon root by asking /info endpoint (to support user
	// namespaced daemon with root remapped uid.gid directory)
	clientConfig, err := d.getClientConfig()
	if err != nil {
		return "", err
	}

	client := &http.Client{
		Transport: clientConfig.transport,
	}

	req, err := http.NewRequest("GET", "/info", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.URL.Host = clientConfig.addr
	req.URL.Scheme = clientConfig.scheme

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	body := ioutils.NewReadCloserWrapper(resp.Body, func() error {
		return resp.Body.Close()
	})

	type Info struct {
		DockerRootDir string
	}
	var b []byte
	var i Info
	b, err = testutil.ReadBody(body)
	if err == nil && resp.StatusCode == http.StatusOK {
		// read the docker root dir
		if err = json.Unmarshal(b, &i); err == nil {
			return i.DockerRootDir, nil
		}
	}
	return "", err
}

// Sock returns the socket path of the daemon
func (d *Daemon) Sock() string {
	return fmt.Sprintf("unix://" + d.sockPath())
}

func (d *Daemon) sockPath() string {
	return filepath.Join(SockRoot, d.id+".sock")
}

// WaitRun waits for a container to be running for 10s
func (d *Daemon) WaitRun(contID string) error {
	args := []string{"--host", d.Sock()}
	return WaitInspectWithArgs(d.dockerBinary, contID, "{{.State.Running}}", "true", 10*time.Second, args...)
}

// GetBaseDeviceSize returns the base device size of the daemon
func (d *Daemon) GetBaseDeviceSize(c *check.C) int64 {
	infoCmdOutput, _, err := testutil.RunCommandPipelineWithOutput(
		exec.Command(d.dockerBinary, "-H", d.Sock(), "info"),
		exec.Command("grep", "Base Device Size"),
	)
	c.Assert(err, checker.IsNil)
	basesizeSlice := strings.Split(infoCmdOutput, ":")
	basesize := strings.Trim(basesizeSlice[1], " ")
	basesize = strings.Trim(basesize, "\n")[:len(basesize)-3]
	basesizeFloat, err := strconv.ParseFloat(strings.Trim(basesize, " "), 64)
	c.Assert(err, checker.IsNil)
	basesizeBytes := int64(basesizeFloat) * (1024 * 1024 * 1024)
	return basesizeBytes
}

// Cmd executes a docker CLI command against this daemon.
// Example: d.Cmd("version") will run docker -H unix://path/to/unix.sock version
func (d *Daemon) Cmd(args ...string) (string, error) {
	result := icmd.RunCmd(d.Command(args...))
	return result.Combined(), result.Error
}

// Command creates a docker CLI command against this daemon, to be executed later.
// Example: d.Command("version") creates a command to run "docker -H unix://path/to/unix.sock version"
func (d *Daemon) Command(args ...string) icmd.Cmd {
	return icmd.Command(d.dockerBinary, d.PrependHostArg(args)...)
}

// PrependHostArg prepend the specified arguments by the daemon host flags
func (d *Daemon) PrependHostArg(args []string) []string {
	for _, arg := range args {
		if arg == "--host" || arg == "-H" {
			return args
		}
	}
	return append([]string{"--host", d.Sock()}, args...)
}

// SockRequest executes a socket request on a daemon and returns statuscode and output.
func (d *Daemon) SockRequest(method, endpoint string, data interface{}) (int, []byte, error) {
	jsonData := bytes.NewBuffer(nil)
	if err := json.NewEncoder(jsonData).Encode(data); err != nil {
		return -1, nil, err
	}

	res, body, err := d.SockRequestRaw(method, endpoint, jsonData, "application/json")
	if err != nil {
		return -1, nil, err
	}
	b, err := testutil.ReadBody(body)
	return res.StatusCode, b, err
}

// SockRequestRaw executes a socket request on a daemon and returns an http
// response and a reader for the output data.
// Deprecated: use request package instead
func (d *Daemon) SockRequestRaw(method, endpoint string, data io.Reader, ct string) (*http.Response, io.ReadCloser, error) {
	return request.SockRequestRaw(method, endpoint, data, ct, d.Sock())
}

// LogFileName returns the path the daemon's log file
func (d *Daemon) LogFileName() string {
	return d.logFile.Name()
}

// GetIDByName returns the ID of an object (container, volume, …) given its name
func (d *Daemon) GetIDByName(name string) (string, error) {
	return d.inspectFieldWithError(name, "Id")
}

// ActiveContainers returns the list of ids of the currently running containers
func (d *Daemon) ActiveContainers() (ids []string) {
	// FIXME(vdemeester) shouldn't ignore the error
	out, _ := d.Cmd("ps", "-q")
	for _, id := range strings.Split(out, "\n") {
		if id = strings.TrimSpace(id); id != "" {
			ids = append(ids, id)
		}
	}
	return
}

// ReadLogFile returns the content of the daemon log file
func (d *Daemon) ReadLogFile() ([]byte, error) {
	return ioutil.ReadFile(d.logFile.Name())
}

// InspectField returns the field filter by 'filter'
func (d *Daemon) InspectField(name, filter string) (string, error) {
	return d.inspectFilter(name, filter)
}

func (d *Daemon) inspectFilter(name, filter string) (string, error) {
	format := fmt.Sprintf("{{%s}}", filter)
	out, err := d.Cmd("inspect", "-f", format, name)
	if err != nil {
		return "", errors.Errorf("failed to inspect %s: %s", name, out)
	}
	return strings.TrimSpace(out), nil
}

func (d *Daemon) inspectFieldWithError(name, field string) (string, error) {
	return d.inspectFilter(name, fmt.Sprintf(".%s", field))
}

// FindContainerIP returns the ip of the specified container
func (d *Daemon) FindContainerIP(id string) (string, error) {
	out, err := d.Cmd("inspect", "--format='{{ .NetworkSettings.Networks.bridge.IPAddress }}'", id)
	if err != nil {
		return "", err
	}
	return strings.Trim(out, " \r\n'"), nil
}

// BuildImageWithOut builds an image with the specified dockerfile and options and returns the output
func (d *Daemon) BuildImageWithOut(name, dockerfile string, useCache bool, buildFlags ...string) (string, int, error) {
	buildCmd := BuildImageCmdWithHost(d.dockerBinary, name, dockerfile, d.Sock(), useCache, buildFlags...)
	result := icmd.RunCmd(icmd.Cmd{
		Command: buildCmd.Args,
		Env:     buildCmd.Env,
		Dir:     buildCmd.Dir,
		Stdin:   buildCmd.Stdin,
		Stdout:  buildCmd.Stdout,
	})
	return result.Combined(), result.ExitCode, result.Error
}

// CheckActiveContainerCount returns the number of active containers
// FIXME(vdemeester) should re-use ActivateContainers in some way
func (d *Daemon) CheckActiveContainerCount(c *check.C) (interface{}, check.CommentInterface) {
	out, err := d.Cmd("ps", "-q")
	c.Assert(err, checker.IsNil)
	if len(strings.TrimSpace(out)) == 0 {
		return 0, nil
	}
	return len(strings.Split(strings.TrimSpace(out), "\n")), check.Commentf("output: %q", string(out))
}

// ReloadConfig asks the daemon to reload its configuration
func (d *Daemon) ReloadConfig() error {
	if d.cmd == nil || d.cmd.Process == nil {
		return errors.New("daemon is not running")
	}

	errCh := make(chan error)
	started := make(chan struct{})
	go func() {
		_, body, err := request.DoOnHost(d.Sock(), "/events", request.Method(http.MethodGet))
		close(started)
		if err != nil {
			errCh <- err
		}
		defer body.Close()
		dec := json.NewDecoder(body)
		for {
			var e events.Message
			if err := dec.Decode(&e); err != nil {
				errCh <- err
				return
			}
			if e.Type != events.DaemonEventType {
				continue
			}
			if e.Action != "reload" {
				continue
			}
			close(errCh) // notify that we are done
			return
		}
	}()

	<-started
	if err := signalDaemonReload(d.cmd.Process.Pid); err != nil {
		return errors.Errorf("error signaling daemon reload: %v", err)
	}
	select {
	case err := <-errCh:
		if err != nil {
			return errors.Errorf("error waiting for daemon reload event: %v", err)
		}
	case <-time.After(30 * time.Second):
		return errors.New("timeout waiting for daemon reload event")
	}
	return nil
}

// WaitInspectWithArgs waits for the specified expression to be equals to the specified expected string in the given time.
// Deprecated: use cli.WaitCmd instead
func WaitInspectWithArgs(dockerBinary, name, expr, expected string, timeout time.Duration, arg ...string) error {
	after := time.After(timeout)

	args := append(arg, "inspect", "-f", expr, name)
	for {
		result := icmd.RunCommand(dockerBinary, args...)
		if result.Error != nil {
			if !strings.Contains(strings.ToLower(result.Stderr()), "no such") {
				return errors.Errorf("error executing docker inspect: %v\n%s",
					result.Stderr(), result.Stdout())
			}
			select {
			case <-after:
				return result.Error
			default:
				time.Sleep(10 * time.Millisecond)
				continue
			}
		}

		out := strings.TrimSpace(result.Stdout())
		if out == expected {
			break
		}

		select {
		case <-after:
			return errors.Errorf("condition \"%q == %q\" not true in time (%v)", out, expected, timeout)
		default:
		}

		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

// BuildImageCmdWithHost create a build command with the specified arguments.
// Deprecated
// FIXME(vdemeester) move this away
func BuildImageCmdWithHost(dockerBinary, name, dockerfile, host string, useCache bool, buildFlags ...string) *exec.Cmd {
	args := []string{}
	if host != "" {
		args = append(args, "--host", host)
	}
	args = append(args, "build", "-t", name)
	if !useCache {
		args = append(args, "--no-cache")
	}
	args = append(args, buildFlags...)
	args = append(args, "-")
	buildCmd := exec.Command(dockerBinary, args...)
	buildCmd.Stdin = strings.NewReader(dockerfile)
	return buildCmd
}
