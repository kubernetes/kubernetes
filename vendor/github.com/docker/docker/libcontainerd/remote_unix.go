// +build linux solaris

package libcontainerd

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/docker/docker/pkg/locker"
	"github.com/docker/docker/pkg/system"
	"github.com/golang/protobuf/ptypes"
	"github.com/golang/protobuf/ptypes/timestamp"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"golang.org/x/sys/unix"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/transport"
)

const (
	maxConnectionRetryCount      = 3
	containerdHealthCheckTimeout = 3 * time.Second
	containerdShutdownTimeout    = 15 * time.Second
	containerdBinary             = "docker-containerd"
	containerdPidFilename        = "docker-containerd.pid"
	containerdSockFilename       = "docker-containerd.sock"
	containerdStateDir           = "containerd"
	eventTimestampFilename       = "event.ts"
)

type remote struct {
	sync.RWMutex
	apiClient            containerd.APIClient
	daemonPid            int
	stateDir             string
	rpcAddr              string
	startDaemon          bool
	closedManually       bool
	debugLog             bool
	rpcConn              *grpc.ClientConn
	clients              []*client
	eventTsPath          string
	runtime              string
	runtimeArgs          []string
	daemonWaitCh         chan struct{}
	liveRestore          bool
	oomScore             int
	restoreFromTimestamp *timestamp.Timestamp
}

// New creates a fresh instance of libcontainerd remote.
func New(stateDir string, options ...RemoteOption) (_ Remote, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("Failed to connect to containerd. Please make sure containerd is installed in your PATH or you have specified the correct address. Got error: %v", err)
		}
	}()
	r := &remote{
		stateDir:    stateDir,
		daemonPid:   -1,
		eventTsPath: filepath.Join(stateDir, eventTimestampFilename),
	}
	for _, option := range options {
		if err := option.Apply(r); err != nil {
			return nil, err
		}
	}

	if err := system.MkdirAll(stateDir, 0700, ""); err != nil {
		return nil, err
	}

	if r.rpcAddr == "" {
		r.rpcAddr = filepath.Join(stateDir, containerdSockFilename)
	}

	if r.startDaemon {
		if err := r.runContainerdDaemon(); err != nil {
			return nil, err
		}
	}

	// don't output the grpc reconnect logging
	grpclog.SetLogger(log.New(ioutil.Discard, "", log.LstdFlags))
	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithBackoffMaxDelay(2 * time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	}
	conn, err := grpc.Dial(r.rpcAddr, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("error connecting to containerd: %v", err)
	}

	r.rpcConn = conn
	r.apiClient = containerd.NewAPIClient(conn)

	// Get the timestamp to restore from
	t := r.getLastEventTimestamp()
	tsp, err := ptypes.TimestampProto(t)
	if err != nil {
		logrus.Errorf("libcontainerd: failed to convert timestamp: %q", err)
	}
	r.restoreFromTimestamp = tsp

	go r.handleConnectionChange()

	if err := r.startEventsMonitor(); err != nil {
		return nil, err
	}

	return r, nil
}

func (r *remote) UpdateOptions(options ...RemoteOption) error {
	for _, option := range options {
		if err := option.Apply(r); err != nil {
			return err
		}
	}
	return nil
}

func (r *remote) handleConnectionChange() {
	var transientFailureCount = 0

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	healthClient := grpc_health_v1.NewHealthClient(r.rpcConn)

	for {
		<-ticker.C
		ctx, cancel := context.WithTimeout(context.Background(), containerdHealthCheckTimeout)
		_, err := healthClient.Check(ctx, &grpc_health_v1.HealthCheckRequest{})
		cancel()
		if err == nil {
			continue
		}

		logrus.Debugf("libcontainerd: containerd health check returned error: %v", err)

		if r.daemonPid != -1 {
			if r.closedManually {
				// Well, we asked for it to stop, just return
				return
			}
			// all other errors are transient
			// Reset state to be notified of next failure
			transientFailureCount++
			if transientFailureCount >= maxConnectionRetryCount {
				transientFailureCount = 0
				if system.IsProcessAlive(r.daemonPid) {
					system.KillProcess(r.daemonPid)
				}
				<-r.daemonWaitCh
				if err := r.runContainerdDaemon(); err != nil { //FIXME: Handle error
					logrus.Errorf("libcontainerd: error restarting containerd: %v", err)
				}
				continue
			}
		}
	}
}

func (r *remote) Cleanup() {
	if r.daemonPid == -1 {
		return
	}
	r.closedManually = true
	r.rpcConn.Close()
	// Ask the daemon to quit
	unix.Kill(r.daemonPid, unix.SIGTERM)

	// Wait up to 15secs for it to stop
	for i := time.Duration(0); i < containerdShutdownTimeout; i += time.Second {
		if !system.IsProcessAlive(r.daemonPid) {
			break
		}
		time.Sleep(time.Second)
	}

	if system.IsProcessAlive(r.daemonPid) {
		logrus.Warnf("libcontainerd: containerd (%d) didn't stop within 15 secs, killing it\n", r.daemonPid)
		unix.Kill(r.daemonPid, unix.SIGKILL)
	}

	// cleanup some files
	os.Remove(filepath.Join(r.stateDir, containerdPidFilename))
	os.Remove(filepath.Join(r.stateDir, containerdSockFilename))
}

func (r *remote) Client(b Backend) (Client, error) {
	c := &client{
		clientCommon: clientCommon{
			backend:    b,
			containers: make(map[string]*container),
			locker:     locker.New(),
		},
		remote:        r,
		exitNotifiers: make(map[string]*exitNotifier),
		liveRestore:   r.liveRestore,
	}

	r.Lock()
	r.clients = append(r.clients, c)
	r.Unlock()
	return c, nil
}

func (r *remote) updateEventTimestamp(t time.Time) {
	f, err := os.OpenFile(r.eventTsPath, unix.O_CREAT|unix.O_WRONLY|unix.O_TRUNC, 0600)
	if err != nil {
		logrus.Warnf("libcontainerd: failed to open event timestamp file: %v", err)
		return
	}
	defer f.Close()

	b, err := t.MarshalText()
	if err != nil {
		logrus.Warnf("libcontainerd: failed to encode timestamp: %v", err)
		return
	}

	n, err := f.Write(b)
	if err != nil || n != len(b) {
		logrus.Warnf("libcontainerd: failed to update event timestamp file: %v", err)
		f.Truncate(0)
		return
	}
}

func (r *remote) getLastEventTimestamp() time.Time {
	t := time.Now()

	fi, err := os.Stat(r.eventTsPath)
	if os.IsNotExist(err) || fi.Size() == 0 {
		return t
	}

	f, err := os.Open(r.eventTsPath)
	if err != nil {
		logrus.Warnf("libcontainerd: Unable to access last event ts: %v", err)
		return t
	}
	defer f.Close()

	b := make([]byte, fi.Size())
	n, err := f.Read(b)
	if err != nil || n != len(b) {
		logrus.Warnf("libcontainerd: Unable to read last event ts: %v", err)
		return t
	}

	t.UnmarshalText(b)

	return t
}

func (r *remote) startEventsMonitor() error {
	// First, get past events
	t := r.getLastEventTimestamp()
	tsp, err := ptypes.TimestampProto(t)
	if err != nil {
		logrus.Errorf("libcontainerd: failed to convert timestamp: %q", err)
	}
	er := &containerd.EventsRequest{
		Timestamp: tsp,
	}

	var events containerd.API_EventsClient
	for {
		events, err = r.apiClient.Events(context.Background(), er, grpc.FailFast(false))
		if err == nil {
			break
		}
		logrus.Warnf("libcontainerd: failed to get events from containerd: %q", err)

		if r.closedManually {
			// ignore error if grpc remote connection is closed manually
			return nil
		}

		<-time.After(100 * time.Millisecond)
	}

	go r.handleEventStream(events)
	return nil
}

func (r *remote) handleEventStream(events containerd.API_EventsClient) {
	for {
		e, err := events.Recv()
		if err != nil {
			if grpc.ErrorDesc(err) == transport.ErrConnClosing.Desc &&
				r.closedManually {
				// ignore error if grpc remote connection is closed manually
				return
			}
			logrus.Errorf("libcontainerd: failed to receive event from containerd: %v", err)
			go r.startEventsMonitor()
			return
		}

		logrus.Debugf("libcontainerd: received containerd event: %#v", e)

		var container *container
		var c *client
		r.RLock()
		for _, c = range r.clients {
			container, err = c.getContainer(e.Id)
			if err == nil {
				break
			}
		}
		r.RUnlock()
		if container == nil {
			logrus.Warnf("libcontainerd: unknown container %s", e.Id)
			continue
		}

		if err := container.handleEvent(e); err != nil {
			logrus.Errorf("libcontainerd: error processing state change for %s: %v", e.Id, err)
		}

		tsp, err := ptypes.Timestamp(e.Timestamp)
		if err != nil {
			logrus.Errorf("libcontainerd: failed to convert event timestamp: %q", err)
			continue
		}

		r.updateEventTimestamp(tsp)
	}
}

func (r *remote) runContainerdDaemon() error {
	pidFilename := filepath.Join(r.stateDir, containerdPidFilename)
	f, err := os.OpenFile(pidFilename, os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	// File exist, check if the daemon is alive
	b := make([]byte, 8)
	n, err := f.Read(b)
	if err != nil && err != io.EOF {
		return err
	}

	if n > 0 {
		pid, err := strconv.ParseUint(string(b[:n]), 10, 64)
		if err != nil {
			return err
		}
		if system.IsProcessAlive(int(pid)) {
			logrus.Infof("libcontainerd: previous instance of containerd still alive (%d)", pid)
			r.daemonPid = int(pid)
			return nil
		}
	}

	// rewind the file
	_, err = f.Seek(0, os.SEEK_SET)
	if err != nil {
		return err
	}

	// Truncate it
	err = f.Truncate(0)
	if err != nil {
		return err
	}

	// Start a new instance
	args := []string{
		"-l", fmt.Sprintf("unix://%s", r.rpcAddr),
		"--metrics-interval=0",
		"--start-timeout", "2m",
		"--state-dir", filepath.Join(r.stateDir, containerdStateDir),
	}
	if goruntime.GOOS == "solaris" {
		args = append(args, "--shim", "containerd-shim", "--runtime", "runc")
	} else {
		args = append(args, "--shim", "docker-containerd-shim")
		if r.runtime != "" {
			args = append(args, "--runtime")
			args = append(args, r.runtime)
		}
	}
	if r.debugLog {
		args = append(args, "--debug")
	}
	if len(r.runtimeArgs) > 0 {
		for _, v := range r.runtimeArgs {
			args = append(args, "--runtime-args")
			args = append(args, v)
		}
		logrus.Debugf("libcontainerd: runContainerdDaemon: runtimeArgs: %s", args)
	}

	cmd := exec.Command(containerdBinary, args...)
	// redirect containerd logs to docker logs
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = setSysProcAttr(true)
	cmd.Env = nil
	// clear the NOTIFY_SOCKET from the env when starting containerd
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "NOTIFY_SOCKET") {
			cmd.Env = append(cmd.Env, e)
		}
	}
	if err := cmd.Start(); err != nil {
		return err
	}

	// unless strictly necessary, do not add anything in between here
	// as the reaper goroutine below needs to kick in as soon as possible
	// and any "return" from code paths added here will defeat the reaper
	// process.

	r.daemonWaitCh = make(chan struct{})
	go func() {
		cmd.Wait()
		close(r.daemonWaitCh)
	}() // Reap our child when needed

	logrus.Infof("libcontainerd: new containerd process, pid: %d", cmd.Process.Pid)
	if err := setOOMScore(cmd.Process.Pid, r.oomScore); err != nil {
		system.KillProcess(cmd.Process.Pid)
		return err
	}
	if _, err := f.WriteString(fmt.Sprintf("%d", cmd.Process.Pid)); err != nil {
		system.KillProcess(cmd.Process.Pid)
		return err
	}

	r.daemonPid = cmd.Process.Pid
	return nil
}

// WithRemoteAddr sets the external containerd socket to connect to.
func WithRemoteAddr(addr string) RemoteOption {
	return rpcAddr(addr)
}

type rpcAddr string

func (a rpcAddr) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.rpcAddr = string(a)
		return nil
	}
	return fmt.Errorf("WithRemoteAddr option not supported for this remote")
}

// WithRuntimePath sets the path of the runtime to be used as the
// default by containerd
func WithRuntimePath(rt string) RemoteOption {
	return runtimePath(rt)
}

type runtimePath string

func (rt runtimePath) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.runtime = string(rt)
		return nil
	}
	return fmt.Errorf("WithRuntime option not supported for this remote")
}

// WithRuntimeArgs sets the list of runtime args passed to containerd
func WithRuntimeArgs(args []string) RemoteOption {
	return runtimeArgs(args)
}

type runtimeArgs []string

func (rt runtimeArgs) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.runtimeArgs = rt
		return nil
	}
	return fmt.Errorf("WithRuntimeArgs option not supported for this remote")
}

// WithStartDaemon defines if libcontainerd should also run containerd daemon.
func WithStartDaemon(start bool) RemoteOption {
	return startDaemon(start)
}

type startDaemon bool

func (s startDaemon) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.startDaemon = bool(s)
		return nil
	}
	return fmt.Errorf("WithStartDaemon option not supported for this remote")
}

// WithDebugLog defines if containerd debug logs will be enabled for daemon.
func WithDebugLog(debug bool) RemoteOption {
	return debugLog(debug)
}

type debugLog bool

func (d debugLog) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.debugLog = bool(d)
		return nil
	}
	return fmt.Errorf("WithDebugLog option not supported for this remote")
}

// WithLiveRestore defines if containers are stopped on shutdown or restored.
func WithLiveRestore(v bool) RemoteOption {
	return liveRestore(v)
}

type liveRestore bool

func (l liveRestore) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.liveRestore = bool(l)
		for _, c := range remote.clients {
			c.liveRestore = bool(l)
		}
		return nil
	}
	return fmt.Errorf("WithLiveRestore option not supported for this remote")
}

// WithOOMScore defines the oom_score_adj to set for the containerd process.
func WithOOMScore(score int) RemoteOption {
	return oomScore(score)
}

type oomScore int

func (o oomScore) Apply(r Remote) error {
	if remote, ok := r.(*remote); ok {
		remote.oomScore = int(o)
		return nil
	}
	return fmt.Errorf("WithOOMScore option not supported for this remote")
}
