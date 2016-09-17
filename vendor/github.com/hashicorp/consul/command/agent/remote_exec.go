package agent

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/hashicorp/consul/consul/structs"
)

const (
	// remoteExecFileName is the name of the file we append to
	// the path, e.g. _rexec/session_id/job
	remoteExecFileName = "job"

	// rExecAck is the suffix added to an ack path
	remoteExecAckSuffix = "ack"

	// remoteExecAck is the suffix added to an exit code
	remoteExecExitSuffix = "exit"

	// remoteExecOutputDivider is used to namespace the output
	remoteExecOutputDivider = "out"

	// remoteExecOutputSize is the size we chunk output too
	remoteExecOutputSize = 4 * 1024

	// remoteExecOutputDeadline is how long we wait before uploading
	// less than the chunk size
	remoteExecOutputDeadline = 500 * time.Millisecond
)

// remoteExecEvent is used as the payload of the user event to transmit
// what we need to know about the event
type remoteExecEvent struct {
	Prefix  string
	Session string
}

// remoteExecSpec is used as the specification of the remote exec.
// It is stored in the KV store
type remoteExecSpec struct {
	Command string
	Script  []byte
	Wait    time.Duration
}

type rexecWriter struct {
	BufCh    chan []byte
	BufSize  int
	BufIdle  time.Duration
	CancelCh chan struct{}

	buf     []byte
	bufLen  int
	bufLock sync.Mutex
	flush   *time.Timer
}

func (r *rexecWriter) Write(b []byte) (int, error) {
	r.bufLock.Lock()
	defer r.bufLock.Unlock()
	if r.flush != nil {
		r.flush.Stop()
		r.flush = nil
	}
	inpLen := len(b)
	if r.buf == nil {
		r.buf = make([]byte, r.BufSize)
	}

COPY:
	remain := len(r.buf) - r.bufLen
	if remain > len(b) {
		copy(r.buf[r.bufLen:], b)
		r.bufLen += len(b)
	} else {
		copy(r.buf[r.bufLen:], b[:remain])
		b = b[remain:]
		r.bufLen += remain
		r.bufLock.Unlock()
		r.Flush()
		r.bufLock.Lock()
		goto COPY
	}

	r.flush = time.AfterFunc(r.BufIdle, r.Flush)
	return inpLen, nil
}

func (r *rexecWriter) Flush() {
	r.bufLock.Lock()
	defer r.bufLock.Unlock()
	if r.flush != nil {
		r.flush.Stop()
		r.flush = nil
	}
	if r.bufLen == 0 {
		return
	}
	select {
	case r.BufCh <- r.buf[:r.bufLen]:
		r.buf = make([]byte, r.BufSize)
		r.bufLen = 0
	case <-r.CancelCh:
		r.bufLen = 0
	}
}

// handleRemoteExec is invoked when a new remote exec request is received
func (a *Agent) handleRemoteExec(msg *UserEvent) {
	a.logger.Printf("[DEBUG] agent: received remote exec event (ID: %s)", msg.ID)
	// Decode the event payload
	var event remoteExecEvent
	if err := json.Unmarshal(msg.Payload, &event); err != nil {
		a.logger.Printf("[ERR] agent: failed to decode remote exec event: %v", err)
		return
	}

	// Read the job specification
	var spec remoteExecSpec
	if !a.remoteExecGetSpec(&event, &spec) {
		return
	}

	// Write the acknowledgement
	if !a.remoteExecWriteAck(&event) {
		return
	}

	// Disable child process reaping so that we can get this command's
	// return value. Note that we take the read lock here since we are
	// waiting on a specific PID and don't need to serialize all waits.
	a.reapLock.RLock()
	defer a.reapLock.RUnlock()

	// Ensure we write out an exit code
	exitCode := 0
	defer a.remoteExecWriteExitCode(&event, &exitCode)

	// Check if this is a script, we may need to spill to disk
	var script string
	if len(spec.Script) != 0 {
		tmpFile, err := ioutil.TempFile("", "rexec")
		if err != nil {
			a.logger.Printf("[DEBUG] agent: failed to make tmp file: %v", err)
			exitCode = 255
			return
		}
		defer os.Remove(tmpFile.Name())
		os.Chmod(tmpFile.Name(), 0750)
		tmpFile.Write(spec.Script)
		tmpFile.Close()
		script = tmpFile.Name()
	} else {
		script = spec.Command
	}

	// Create the exec.Cmd
	a.logger.Printf("[INFO] agent: remote exec '%s'", script)
	cmd, err := ExecScript(script)
	if err != nil {
		a.logger.Printf("[DEBUG] agent: failed to start remote exec: %v", err)
		exitCode = 255
		return
	}

	// Setup the output streaming
	writer := &rexecWriter{
		BufCh:    make(chan []byte, 16),
		BufSize:  remoteExecOutputSize,
		BufIdle:  remoteExecOutputDeadline,
		CancelCh: make(chan struct{}),
	}
	cmd.Stdout = writer
	cmd.Stderr = writer

	// Start execution
	err = cmd.Start()
	if err != nil {
		a.logger.Printf("[DEBUG] agent: failed to start remote exec: %v", err)
		exitCode = 255
		return
	}

	// Wait for the process to exit
	exitCh := make(chan int, 1)
	go func() {
		err := cmd.Wait()
		writer.Flush()
		close(writer.BufCh)
		if err == nil {
			exitCh <- 0
			return
		}

		// Try to determine the exit code
		if exitErr, ok := err.(*exec.ExitError); ok {
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				exitCh <- status.ExitStatus()
				return
			}
		}
		exitCh <- 1
	}()

	// Wait until we are complete, uploading as we go
WAIT:
	for num := 0; ; num++ {
		select {
		case out := <-writer.BufCh:
			if out == nil {
				break WAIT
			}
			if !a.remoteExecWriteOutput(&event, num, out) {
				close(writer.CancelCh)
				exitCode = 255
				return
			}
		case <-time.After(spec.Wait):
			// Acts like a heartbeat, since there is no output
			if !a.remoteExecWriteOutput(&event, num, nil) {
				close(writer.CancelCh)
				exitCode = 255
				return
			}
		}
	}

	// Get the exit code
	exitCode = <-exitCh
}

// remoteExecGetSpec is used to get the exec specification.
// Returns if execution should continue
func (a *Agent) remoteExecGetSpec(event *remoteExecEvent, spec *remoteExecSpec) bool {
	get := structs.KeyRequest{
		Datacenter: a.config.Datacenter,
		Key:        path.Join(event.Prefix, event.Session, remoteExecFileName),
		QueryOptions: structs.QueryOptions{
			AllowStale: true, // Stale read for scale! Retry on failure.
		},
	}
	get.Token = a.config.ACLToken
	var out structs.IndexedDirEntries
QUERY:
	if err := a.RPC("KVS.Get", &get, &out); err != nil {
		a.logger.Printf("[ERR] agent: failed to get remote exec job: %v", err)
		return false
	}
	if len(out.Entries) == 0 {
		// If the initial read was stale and had no data, retry as a consistent read
		if get.QueryOptions.AllowStale {
			a.logger.Printf("[DEBUG] agent: trying consistent fetch of remote exec job spec")
			get.QueryOptions.AllowStale = false
			goto QUERY
		} else {
			a.logger.Printf("[DEBUG] agent: remote exec aborted, job spec missing")
			return false
		}
	}
	if err := json.Unmarshal(out.Entries[0].Value, &spec); err != nil {
		a.logger.Printf("[ERR] agent: failed to decode remote exec spec: %v", err)
		return false
	}
	return true
}

// remoteExecWriteAck is used to write an ack. Returns if execution should
// continue.
func (a *Agent) remoteExecWriteAck(event *remoteExecEvent) bool {
	if err := a.remoteExecWriteKey(event, remoteExecAckSuffix, nil); err != nil {
		a.logger.Printf("[ERR] agent: failed to ack remote exec job: %v", err)
		return false
	}
	return true
}

// remoteExecWriteOutput is used to write output
func (a *Agent) remoteExecWriteOutput(event *remoteExecEvent, num int, output []byte) bool {
	suffix := path.Join(remoteExecOutputDivider, fmt.Sprintf("%05x", num))
	if err := a.remoteExecWriteKey(event, suffix, output); err != nil {
		a.logger.Printf("[ERR] agent: failed to write output for remote exec job: %v", err)
		return false
	}
	return true
}

// remoteExecWriteExitCode is used to write an exit code
func (a *Agent) remoteExecWriteExitCode(event *remoteExecEvent, exitCode *int) bool {
	val := []byte(strconv.FormatInt(int64(*exitCode), 10))
	if err := a.remoteExecWriteKey(event, remoteExecExitSuffix, val); err != nil {
		a.logger.Printf("[ERR] agent: failed to write exit code for remote exec job: %v", err)
		return false
	}
	return true
}

// remoteExecWriteKey is used to write an output key for a remote exec job
func (a *Agent) remoteExecWriteKey(event *remoteExecEvent, suffix string, val []byte) error {
	key := path.Join(event.Prefix, event.Session, a.config.NodeName, suffix)
	write := structs.KVSRequest{
		Datacenter: a.config.Datacenter,
		Op:         structs.KVSLock,
		DirEnt: structs.DirEntry{
			Key:     key,
			Value:   val,
			Session: event.Session,
		},
	}
	write.Token = a.config.ACLToken
	var success bool
	if err := a.RPC("KVS.Apply", &write, &success); err != nil {
		return err
	}
	if !success {
		return fmt.Errorf("write failed")
	}
	return nil
}
