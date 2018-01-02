package daemon

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/strslice"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/exec"
	"github.com/sirupsen/logrus"
)

const (
	// Longest healthcheck probe output message to store. Longer messages will be truncated.
	maxOutputLen = 4096

	// Default interval between probe runs (from the end of the first to the start of the second).
	// Also the time before the first probe.
	defaultProbeInterval = 30 * time.Second

	// The maximum length of time a single probe run should take. If the probe takes longer
	// than this, the check is considered to have failed.
	defaultProbeTimeout = 30 * time.Second

	// The time given for the container to start before the health check starts considering
	// the container unstable. Defaults to none.
	defaultStartPeriod = 0 * time.Second

	// Default number of consecutive failures of the health check
	// for the container to be considered unhealthy.
	defaultProbeRetries = 3

	// Maximum number of entries to record
	maxLogEntries = 5
)

const (
	// Exit status codes that can be returned by the probe command.

	exitStatusHealthy   = 0 // Container is healthy
	exitStatusUnhealthy = 1 // Container is unhealthy
)

// probe implementations know how to run a particular type of probe.
type probe interface {
	// Perform one run of the check. Returns the exit code and an optional
	// short diagnostic string.
	run(context.Context, *Daemon, *container.Container) (*types.HealthcheckResult, error)
}

// cmdProbe implements the "CMD" probe type.
type cmdProbe struct {
	// Run the command with the system's default shell instead of execing it directly.
	shell bool
}

// exec the healthcheck command in the container.
// Returns the exit code and probe output (if any)
func (p *cmdProbe) run(ctx context.Context, d *Daemon, cntr *container.Container) (*types.HealthcheckResult, error) {
	cmdSlice := strslice.StrSlice(cntr.Config.Healthcheck.Test)[1:]
	if p.shell {
		cmdSlice = append(getShell(cntr.Config), cmdSlice...)
	}
	entrypoint, args := d.getEntrypointAndArgs(strslice.StrSlice{}, cmdSlice)
	execConfig := exec.NewConfig()
	execConfig.OpenStdin = false
	execConfig.OpenStdout = true
	execConfig.OpenStderr = true
	execConfig.ContainerID = cntr.ID
	execConfig.DetachKeys = []byte{}
	execConfig.Entrypoint = entrypoint
	execConfig.Args = args
	execConfig.Tty = false
	execConfig.Privileged = false
	execConfig.User = cntr.Config.User

	linkedEnv, err := d.setupLinkedContainers(cntr)
	if err != nil {
		return nil, err
	}
	execConfig.Env = container.ReplaceOrAppendEnvValues(cntr.CreateDaemonEnvironment(execConfig.Tty, linkedEnv), execConfig.Env)

	d.registerExecCommand(cntr, execConfig)
	d.LogContainerEvent(cntr, "exec_create: "+execConfig.Entrypoint+" "+strings.Join(execConfig.Args, " "))

	output := &limitedBuffer{}
	err = d.ContainerExecStart(ctx, execConfig.ID, nil, output, output)
	if err != nil {
		return nil, err
	}
	info, err := d.getExecConfig(execConfig.ID)
	if err != nil {
		return nil, err
	}
	if info.ExitCode == nil {
		return nil, fmt.Errorf("Healthcheck for container %s has no exit code!", cntr.ID)
	}
	// Note: Go's json package will handle invalid UTF-8 for us
	out := output.String()
	return &types.HealthcheckResult{
		End:      time.Now(),
		ExitCode: *info.ExitCode,
		Output:   out,
	}, nil
}

// Update the container's Status.Health struct based on the latest probe's result.
func handleProbeResult(d *Daemon, c *container.Container, result *types.HealthcheckResult, done chan struct{}) {
	c.Lock()
	defer c.Unlock()

	// probe may have been cancelled while waiting on lock. Ignore result then
	select {
	case <-done:
		return
	default:
	}

	retries := c.Config.Healthcheck.Retries
	if retries <= 0 {
		retries = defaultProbeRetries
	}

	h := c.State.Health
	oldStatus := h.Status

	if len(h.Log) >= maxLogEntries {
		h.Log = append(h.Log[len(h.Log)+1-maxLogEntries:], result)
	} else {
		h.Log = append(h.Log, result)
	}

	if result.ExitCode == exitStatusHealthy {
		h.FailingStreak = 0
		h.Status = types.Healthy
	} else { // Failure (including invalid exit code)
		shouldIncrementStreak := true

		// If the container is starting (i.e. we never had a successful health check)
		// then we check if we are within the start period of the container in which
		// case we do not increment the failure streak.
		if h.Status == types.Starting {
			startPeriod := timeoutWithDefault(c.Config.Healthcheck.StartPeriod, defaultStartPeriod)
			timeSinceStart := result.Start.Sub(c.State.StartedAt)

			// If still within the start period, then don't increment failing streak.
			if timeSinceStart < startPeriod {
				shouldIncrementStreak = false
			}
		}

		if shouldIncrementStreak {
			h.FailingStreak++

			if h.FailingStreak >= retries {
				h.Status = types.Unhealthy
			}
		}
		// Else we're starting or healthy. Stay in that state.
	}

	// replicate Health status changes
	if err := c.CheckpointTo(d.containersReplica); err != nil {
		// queries will be inconsistent until the next probe runs or other state mutations
		// checkpoint the container
		logrus.Errorf("Error replicating health state for container %s: %v", c.ID, err)
	}

	if oldStatus != h.Status {
		d.LogContainerEvent(c, "health_status: "+h.Status)
	}
}

// Run the container's monitoring thread until notified via "stop".
// There is never more than one monitor thread running per container at a time.
func monitor(d *Daemon, c *container.Container, stop chan struct{}, probe probe) {
	probeTimeout := timeoutWithDefault(c.Config.Healthcheck.Timeout, defaultProbeTimeout)
	probeInterval := timeoutWithDefault(c.Config.Healthcheck.Interval, defaultProbeInterval)
	for {
		select {
		case <-stop:
			logrus.Debugf("Stop healthcheck monitoring for container %s (received while idle)", c.ID)
			return
		case <-time.After(probeInterval):
			logrus.Debugf("Running health check for container %s ...", c.ID)
			startTime := time.Now()
			ctx, cancelProbe := context.WithTimeout(context.Background(), probeTimeout)
			results := make(chan *types.HealthcheckResult, 1)
			go func() {
				healthChecksCounter.Inc()
				result, err := probe.run(ctx, d, c)
				if err != nil {
					healthChecksFailedCounter.Inc()
					logrus.Warnf("Health check for container %s error: %v", c.ID, err)
					results <- &types.HealthcheckResult{
						ExitCode: -1,
						Output:   err.Error(),
						Start:    startTime,
						End:      time.Now(),
					}
				} else {
					result.Start = startTime
					logrus.Debugf("Health check for container %s done (exitCode=%d)", c.ID, result.ExitCode)
					results <- result
				}
				close(results)
			}()
			select {
			case <-stop:
				logrus.Debugf("Stop healthcheck monitoring for container %s (received while probing)", c.ID)
				cancelProbe()
				// Wait for probe to exit (it might take a while to respond to the TERM
				// signal and we don't want dying probes to pile up).
				<-results
				return
			case result := <-results:
				handleProbeResult(d, c, result, stop)
				// Stop timeout
				cancelProbe()
			case <-ctx.Done():
				logrus.Debugf("Health check for container %s taking too long", c.ID)
				handleProbeResult(d, c, &types.HealthcheckResult{
					ExitCode: -1,
					Output:   fmt.Sprintf("Health check exceeded timeout (%v)", probeTimeout),
					Start:    startTime,
					End:      time.Now(),
				}, stop)
				cancelProbe()
				// Wait for probe to exit (it might take a while to respond to the TERM
				// signal and we don't want dying probes to pile up).
				<-results
			}
		}
	}
}

// Get a suitable probe implementation for the container's healthcheck configuration.
// Nil will be returned if no healthcheck was configured or NONE was set.
func getProbe(c *container.Container) probe {
	config := c.Config.Healthcheck
	if config == nil || len(config.Test) == 0 {
		return nil
	}
	switch config.Test[0] {
	case "CMD":
		return &cmdProbe{shell: false}
	case "CMD-SHELL":
		return &cmdProbe{shell: true}
	default:
		logrus.Warnf("Unknown healthcheck type '%s' (expected 'CMD') in container %s", config.Test[0], c.ID)
		return nil
	}
}

// Ensure the health-check monitor is running or not, depending on the current
// state of the container.
// Called from monitor.go, with c locked.
func (d *Daemon) updateHealthMonitor(c *container.Container) {
	h := c.State.Health
	if h == nil {
		return // No healthcheck configured
	}

	probe := getProbe(c)
	wantRunning := c.Running && !c.Paused && probe != nil
	if wantRunning {
		if stop := h.OpenMonitorChannel(); stop != nil {
			go monitor(d, c, stop, probe)
		}
	} else {
		h.CloseMonitorChannel()
	}
}

// Reset the health state for a newly-started, restarted or restored container.
// initHealthMonitor is called from monitor.go and we should never be running
// two instances at once.
// Called with c locked.
func (d *Daemon) initHealthMonitor(c *container.Container) {
	// If no healthcheck is setup then don't init the monitor
	if getProbe(c) == nil {
		return
	}

	// This is needed in case we're auto-restarting
	d.stopHealthchecks(c)

	if h := c.State.Health; h != nil {
		h.Status = types.Starting
		h.FailingStreak = 0
	} else {
		h := &container.Health{}
		h.Status = types.Starting
		c.State.Health = h
	}

	d.updateHealthMonitor(c)
}

// Called when the container is being stopped (whether because the health check is
// failing or for any other reason).
func (d *Daemon) stopHealthchecks(c *container.Container) {
	h := c.State.Health
	if h != nil {
		h.CloseMonitorChannel()
	}
}

// Buffer up to maxOutputLen bytes. Further data is discarded.
type limitedBuffer struct {
	buf       bytes.Buffer
	mu        sync.Mutex
	truncated bool // indicates that data has been lost
}

// Append to limitedBuffer while there is room.
func (b *limitedBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	bufLen := b.buf.Len()
	dataLen := len(data)
	keep := min(maxOutputLen-bufLen, dataLen)
	if keep > 0 {
		b.buf.Write(data[:keep])
	}
	if keep < dataLen {
		b.truncated = true
	}
	return dataLen, nil
}

// The contents of the buffer, with "..." appended if it overflowed.
func (b *limitedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	out := b.buf.String()
	if b.truncated {
		out = out + "..."
	}
	return out
}

// If configuredValue is zero, use defaultValue instead.
func timeoutWithDefault(configuredValue time.Duration, defaultValue time.Duration) time.Duration {
	if configuredValue == 0 {
		return defaultValue
	}
	return configuredValue
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func getShell(config *containertypes.Config) []string {
	if len(config.Shell) != 0 {
		return config.Shell
	}
	if runtime.GOOS != "windows" {
		return []string{"/bin/sh", "-c"}
	}
	return []string{"cmd", "/S", "/C"}
}
