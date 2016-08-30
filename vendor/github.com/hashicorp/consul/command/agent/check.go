package agent

import (
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/armon/circbuf"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/go-cleanhttp"
)

const (
	// Do not allow for a interval below this value.
	// Otherwise we risk fork bombing a system.
	MinInterval = time.Second

	// Limit the size of a check's output to the
	// last CheckBufSize. Prevents an enormous buffer
	// from being captured
	CheckBufSize = 4 * 1024 // 4KB

	// Use this user agent when doing requests for
	// HTTP health checks.
	HttpUserAgent = "Consul Health Check"
)

// CheckType is used to create either the CheckMonitor
// or the CheckTTL.
// Five types are supported: Script, HTTP, TCP, Docker and TTL
// Script, HTTP, Docker and TCP all require Interval
// Only one of the types needs to be provided
// TTL or Script/Interval or HTTP/Interval or TCP/Interval or Docker/Interval
type CheckType struct {
	Script            string
	HTTP              string
	TCP               string
	Interval          time.Duration
	DockerContainerID string
	Shell             string

	Timeout time.Duration
	TTL     time.Duration

	Status string

	Notes string
}
type CheckTypes []*CheckType

// Valid checks if the CheckType is valid
func (c *CheckType) Valid() bool {
	return c.IsTTL() || c.IsMonitor() || c.IsHTTP() || c.IsTCP() || c.IsDocker()
}

// IsTTL checks if this is a TTL type
func (c *CheckType) IsTTL() bool {
	return c.TTL != 0
}

// IsMonitor checks if this is a Monitor type
func (c *CheckType) IsMonitor() bool {
	return c.Script != "" && c.DockerContainerID == "" && c.Interval != 0
}

// IsHTTP checks if this is a HTTP type
func (c *CheckType) IsHTTP() bool {
	return c.HTTP != "" && c.Interval != 0
}

// IsTCP checks if this is a TCP type
func (c *CheckType) IsTCP() bool {
	return c.TCP != "" && c.Interval != 0
}

func (c *CheckType) IsDocker() bool {
	return c.DockerContainerID != "" && c.Script != "" && c.Interval != 0
}

// CheckNotifier interface is used by the CheckMonitor
// to notify when a check has a status update. The update
// should take care to be idempotent.
type CheckNotifier interface {
	UpdateCheck(checkID, status, output string)
}

// CheckMonitor is used to periodically invoke a script to
// determine the health of a given check. It is compatible with
// nagios plugins and expects the output in the same format.
type CheckMonitor struct {
	Notify   CheckNotifier
	CheckID  string
	Script   string
	Interval time.Duration
	Logger   *log.Logger
	ReapLock *sync.RWMutex

	stop     bool
	stopCh   chan struct{}
	stopLock sync.Mutex
}

// Start is used to start a check monitor.
// Monitor runs until stop is called
func (c *CheckMonitor) Start() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	c.stop = false
	c.stopCh = make(chan struct{})
	go c.run()
}

// Stop is used to stop a check monitor.
func (c *CheckMonitor) Stop() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	if !c.stop {
		c.stop = true
		close(c.stopCh)
	}
}

// run is invoked by a goroutine to run until Stop() is called
func (c *CheckMonitor) run() {
	// Get the randomized initial pause time
	initialPauseTime := lib.RandomStagger(c.Interval)
	c.Logger.Printf("[DEBUG] agent: pausing %v before first invocation of %s", initialPauseTime, c.Script)
	next := time.After(initialPauseTime)
	for {
		select {
		case <-next:
			c.check()
			next = time.After(c.Interval)
		case <-c.stopCh:
			return
		}
	}
}

// check is invoked periodically to perform the script check
func (c *CheckMonitor) check() {
	// Disable child process reaping so that we can get this command's
	// return value. Note that we take the read lock here since we are
	// waiting on a specific PID and don't need to serialize all waits.
	c.ReapLock.RLock()
	defer c.ReapLock.RUnlock()

	// Create the command
	cmd, err := ExecScript(c.Script)
	if err != nil {
		c.Logger.Printf("[ERR] agent: failed to setup invoke '%s': %s", c.Script, err)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, err.Error())
		return
	}

	// Collect the output
	output, _ := circbuf.NewBuffer(CheckBufSize)
	cmd.Stdout = output
	cmd.Stderr = output

	// Start the check
	if err := cmd.Start(); err != nil {
		c.Logger.Printf("[ERR] agent: failed to invoke '%s': %s", c.Script, err)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, err.Error())
		return
	}

	// Wait for the check to complete
	errCh := make(chan error, 2)
	go func() {
		errCh <- cmd.Wait()
	}()
	go func() {
		time.Sleep(30 * time.Second)
		errCh <- fmt.Errorf("Timed out running check '%s'", c.Script)
	}()
	err = <-errCh

	// Get the output, add a message about truncation
	outputStr := string(output.Bytes())
	if output.TotalWritten() > output.Size() {
		outputStr = fmt.Sprintf("Captured %d of %d bytes\n...\n%s",
			output.Size(), output.TotalWritten(), outputStr)
	}

	c.Logger.Printf("[DEBUG] agent: check '%s' script '%s' output: %s",
		c.CheckID, c.Script, outputStr)

	// Check if the check passed
	if err == nil {
		c.Logger.Printf("[DEBUG] agent: Check '%v' is passing", c.CheckID)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthPassing, outputStr)
		return
	}

	// If the exit code is 1, set check as warning
	exitErr, ok := err.(*exec.ExitError)
	if ok {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
			code := status.ExitStatus()
			if code == 1 {
				c.Logger.Printf("[WARN] agent: Check '%v' is now warning", c.CheckID)
				c.Notify.UpdateCheck(c.CheckID, structs.HealthWarning, outputStr)
				return
			}
		}
	}

	// Set the health as critical
	c.Logger.Printf("[WARN] agent: Check '%v' is now critical", c.CheckID)
	c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, outputStr)
}

// CheckTTL is used to apply a TTL to check status,
// and enables clients to set the status of a check
// but upon the TTL expiring, the check status is
// automatically set to critical.
type CheckTTL struct {
	Notify  CheckNotifier
	CheckID string
	TTL     time.Duration
	Logger  *log.Logger

	timer *time.Timer

	lastOutput     string
	lastOutputLock sync.RWMutex

	stop     bool
	stopCh   chan struct{}
	stopLock sync.Mutex
}

// Start is used to start a check ttl, runs until Stop()
func (c *CheckTTL) Start() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	c.stop = false
	c.stopCh = make(chan struct{})
	c.timer = time.NewTimer(c.TTL)
	go c.run()
}

// Stop is used to stop a check ttl.
func (c *CheckTTL) Stop() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	if !c.stop {
		c.timer.Stop()
		c.stop = true
		close(c.stopCh)
	}
}

// run is used to handle TTL expiration and to update the check status
func (c *CheckTTL) run() {
	for {
		select {
		case <-c.timer.C:
			c.Logger.Printf("[WARN] agent: Check '%v' missed TTL, is now critical",
				c.CheckID)
			c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, c.getExpiredOutput())

		case <-c.stopCh:
			return
		}
	}
}

// getExpiredOutput formats the output for the case when the TTL is expired.
func (c *CheckTTL) getExpiredOutput() string {
	c.lastOutputLock.RLock()
	defer c.lastOutputLock.RUnlock()

	const prefix = "TTL expired"
	if c.lastOutput == "" {
		return prefix
	}

	return fmt.Sprintf("%s (last output before timeout follows): %s", prefix, c.lastOutput)
}

// SetStatus is used to update the status of the check,
// and to renew the TTL. If expired, TTL is restarted.
func (c *CheckTTL) SetStatus(status, output string) {
	c.Logger.Printf("[DEBUG] agent: Check '%v' status is now %v",
		c.CheckID, status)
	c.Notify.UpdateCheck(c.CheckID, status, output)

	// Store the last output so we can retain it if the TTL expires.
	c.lastOutputLock.Lock()
	c.lastOutput = output
	c.lastOutputLock.Unlock()

	c.timer.Reset(c.TTL)
}

// persistedCheck is used to serialize a check and write it to disk
// so that it may be restored later on.
type persistedCheck struct {
	Check   *structs.HealthCheck
	ChkType *CheckType
	Token   string
}

// persistedCheckState is used to persist the current state of a given
// check. This is different from the check definition, and includes an
// expiration timestamp which is used to determine staleness on later
// agent restarts.
type persistedCheckState struct {
	CheckID string
	Output  string
	Status  string
	Expires int64
}

// CheckHTTP is used to periodically make an HTTP request to
// determine the health of a given check.
// The check is passing if the response code is 2XX.
// The check is warning if the response code is 429.
// The check is critical if the response code is anything else
// or if the request returns an error
type CheckHTTP struct {
	Notify   CheckNotifier
	CheckID  string
	HTTP     string
	Interval time.Duration
	Timeout  time.Duration
	Logger   *log.Logger

	httpClient *http.Client
	stop       bool
	stopCh     chan struct{}
	stopLock   sync.Mutex
}

// Start is used to start an HTTP check.
// The check runs until stop is called
func (c *CheckHTTP) Start() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()

	if c.httpClient == nil {
		// Create the transport. We disable HTTP Keep-Alive's to prevent
		// failing checks due to the keepalive interval.
		trans := cleanhttp.DefaultTransport()
		trans.DisableKeepAlives = true

		// Create the HTTP client.
		c.httpClient = &http.Client{
			Timeout:   10 * time.Second,
			Transport: trans,
		}

		// For long (>10s) interval checks the http timeout is 10s, otherwise the
		// timeout is the interval. This means that a check *should* return
		// before the next check begins.
		if c.Timeout > 0 && c.Timeout < c.Interval {
			c.httpClient.Timeout = c.Timeout
		} else if c.Interval < 10*time.Second {
			c.httpClient.Timeout = c.Interval
		}
	}

	c.stop = false
	c.stopCh = make(chan struct{})
	go c.run()
}

// Stop is used to stop an HTTP check.
func (c *CheckHTTP) Stop() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	if !c.stop {
		c.stop = true
		close(c.stopCh)
	}
}

// run is invoked by a goroutine to run until Stop() is called
func (c *CheckHTTP) run() {
	// Get the randomized initial pause time
	initialPauseTime := lib.RandomStagger(c.Interval)
	c.Logger.Printf("[DEBUG] agent: pausing %v before first HTTP request of %s", initialPauseTime, c.HTTP)
	next := time.After(initialPauseTime)
	for {
		select {
		case <-next:
			c.check()
			next = time.After(c.Interval)
		case <-c.stopCh:
			return
		}
	}
}

// check is invoked periodically to perform the HTTP check
func (c *CheckHTTP) check() {
	req, err := http.NewRequest("GET", c.HTTP, nil)
	if err != nil {
		c.Logger.Printf("[WARN] agent: http request failed '%s': %s", c.HTTP, err)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, err.Error())
		return
	}

	req.Header.Set("User-Agent", HttpUserAgent)
	req.Header.Set("Accept", "text/plain, text/*, */*")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.Logger.Printf("[WARN] agent: http request failed '%s': %s", c.HTTP, err)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, err.Error())
		return
	}
	defer resp.Body.Close()

	// Format the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		c.Logger.Printf("[WARN] agent: check '%v': Get error while reading body: %s", c.CheckID, err)
		body = []byte{}
	}
	result := fmt.Sprintf("HTTP GET %s: %s Output: %s", c.HTTP, resp.Status, body)

	if resp.StatusCode >= 200 && resp.StatusCode <= 299 {
		// PASSING (2xx)
		c.Logger.Printf("[DEBUG] agent: check '%v' is passing", c.CheckID)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthPassing, result)

	} else if resp.StatusCode == 429 {
		// WARNING
		// 429 Too Many Requests (RFC 6585)
		// The user has sent too many requests in a given amount of time.
		c.Logger.Printf("[WARN] agent: check '%v' is now warning", c.CheckID)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthWarning, result)

	} else {
		// CRITICAL
		c.Logger.Printf("[WARN] agent: check '%v' is now critical", c.CheckID)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, result)
	}
}

// CheckTCP is used to periodically make an TCP/UDP connection to
// determine the health of a given check.
// The check is passing if the connection succeeds
// The check is critical if the connection returns an error
type CheckTCP struct {
	Notify   CheckNotifier
	CheckID  string
	TCP      string
	Interval time.Duration
	Timeout  time.Duration
	Logger   *log.Logger

	dialer   *net.Dialer
	stop     bool
	stopCh   chan struct{}
	stopLock sync.Mutex
}

// Start is used to start a TCP check.
// The check runs until stop is called
func (c *CheckTCP) Start() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()

	if c.dialer == nil {
		// Create the socket dialer
		c.dialer = &net.Dialer{DualStack: true}

		// For long (>10s) interval checks the socket timeout is 10s, otherwise
		// the timeout is the interval. This means that a check *should* return
		// before the next check begins.
		if c.Timeout > 0 && c.Timeout < c.Interval {
			c.dialer.Timeout = c.Timeout
		} else if c.Interval < 10*time.Second {
			c.dialer.Timeout = c.Interval
		}
	}

	c.stop = false
	c.stopCh = make(chan struct{})
	go c.run()
}

// Stop is used to stop a TCP check.
func (c *CheckTCP) Stop() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	if !c.stop {
		c.stop = true
		close(c.stopCh)
	}
}

// run is invoked by a goroutine to run until Stop() is called
func (c *CheckTCP) run() {
	// Get the randomized initial pause time
	initialPauseTime := lib.RandomStagger(c.Interval)
	c.Logger.Printf("[DEBUG] agent: pausing %v before first socket connection of %s", initialPauseTime, c.TCP)
	next := time.After(initialPauseTime)
	for {
		select {
		case <-next:
			c.check()
			next = time.After(c.Interval)
		case <-c.stopCh:
			return
		}
	}
}

// check is invoked periodically to perform the TCP check
func (c *CheckTCP) check() {
	conn, err := c.dialer.Dial(`tcp`, c.TCP)
	if err != nil {
		c.Logger.Printf("[WARN] agent: socket connection failed '%s': %s", c.TCP, err)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, err.Error())
		return
	}
	conn.Close()
	c.Logger.Printf("[DEBUG] agent: check '%v' is passing", c.CheckID)
	c.Notify.UpdateCheck(c.CheckID, structs.HealthPassing, fmt.Sprintf("TCP connect %s: Success", c.TCP))
}

// A custom interface since go-dockerclient doesn't have one
// We will use this interface in our test to inject a fake client
type DockerClient interface {
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
	InspectExec(string) (*docker.ExecInspect, error)
}

// CheckDocker is used to periodically invoke a script to
// determine the health of an application running inside a
// Docker Container. We assume that the script is compatible
// with nagios plugins and expects the output in the same format.
type CheckDocker struct {
	Notify            CheckNotifier
	CheckID           string
	Script            string
	DockerContainerID string
	Shell             string
	Interval          time.Duration
	Logger            *log.Logger

	dockerClient DockerClient
	cmd          []string
	stop         bool
	stopCh       chan struct{}
	stopLock     sync.Mutex
}

//Initializes the Docker Client
func (c *CheckDocker) Init() error {
	//create the docker client
	var err error
	c.dockerClient, err = docker.NewClientFromEnv()
	if err != nil {
		c.Logger.Printf("[DEBUG] Error creating the Docker client: %s", err.Error())
		return err
	}
	return nil
}

// Start is used to start checks.
// Docker Checks runs until stop is called
func (c *CheckDocker) Start() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()

	//figure out the shell
	if c.Shell == "" {
		c.Shell = shell()
	}

	c.cmd = []string{c.Shell, "-c", c.Script}

	c.stop = false
	c.stopCh = make(chan struct{})
	go c.run()
}

// Stop is used to stop a docker check.
func (c *CheckDocker) Stop() {
	c.stopLock.Lock()
	defer c.stopLock.Unlock()
	if !c.stop {
		c.stop = true
		close(c.stopCh)
	}
}

// run is invoked by a goroutine to run until Stop() is called
func (c *CheckDocker) run() {
	// Get the randomized initial pause time
	initialPauseTime := lib.RandomStagger(c.Interval)
	c.Logger.Printf("[DEBUG] agent: pausing %v before first invocation of %s -c %s in container %s", initialPauseTime, c.Shell, c.Script, c.DockerContainerID)
	next := time.After(initialPauseTime)
	for {
		select {
		case <-next:
			c.check()
			next = time.After(c.Interval)
		case <-c.stopCh:
			return
		}
	}
}

func (c *CheckDocker) check() {
	//Set up the Exec since
	execOpts := docker.CreateExecOptions{
		AttachStdin:  false,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
		Cmd:          c.cmd,
		Container:    c.DockerContainerID,
	}
	var (
		exec *docker.Exec
		err  error
	)
	if exec, err = c.dockerClient.CreateExec(execOpts); err != nil {
		c.Logger.Printf("[DEBUG] agent: Error while creating Exec: %s", err.Error())
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, fmt.Sprintf("Unable to create Exec, error: %s", err.Error()))
		return
	}

	// Collect the output
	output, _ := circbuf.NewBuffer(CheckBufSize)

	err = c.dockerClient.StartExec(exec.ID, docker.StartExecOptions{Detach: false, Tty: false, OutputStream: output, ErrorStream: output})
	if err != nil {
		c.Logger.Printf("[DEBUG] Error in executing health checks: %s", err.Error())
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, fmt.Sprintf("Unable to start Exec: %s", err.Error()))
		return
	}

	// Get the output, add a message about truncation
	outputStr := string(output.Bytes())
	if output.TotalWritten() > output.Size() {
		outputStr = fmt.Sprintf("Captured %d of %d bytes\n...\n%s",
			output.Size(), output.TotalWritten(), outputStr)
	}

	c.Logger.Printf("[DEBUG] agent: check '%s' script '%s' output: %s",
		c.CheckID, c.Script, outputStr)

	execInfo, err := c.dockerClient.InspectExec(exec.ID)
	if err != nil {
		c.Logger.Printf("[DEBUG] Error in inspecting check result : %s", err.Error())
		c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, fmt.Sprintf("Unable to inspect Exec: %s", err.Error()))
		return
	}

	// Sets the status of the check to healthy if exit code is 0
	if execInfo.ExitCode == 0 {
		c.Notify.UpdateCheck(c.CheckID, structs.HealthPassing, outputStr)
		return
	}

	// Set the status of the check to Warning if exit code is 1
	if execInfo.ExitCode == 1 {
		c.Logger.Printf("[DEBUG] Check failed with exit code: %d", execInfo.ExitCode)
		c.Notify.UpdateCheck(c.CheckID, structs.HealthWarning, outputStr)
		return
	}

	// Set the health as critical
	c.Logger.Printf("[WARN] agent: Check '%v' is now critical", c.CheckID)
	c.Notify.UpdateCheck(c.CheckID, structs.HealthCritical, outputStr)
}

func shell() string {
	if otherShell := os.Getenv("SHELL"); otherShell != "" {
		return otherShell
	} else {
		return "/bin/sh"
	}
}
