/*
Copyright (c) 2017-2021 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package process

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/vmware/govmomi/toolbox/hgfs"
	"github.com/vmware/govmomi/toolbox/vix"
)

var (
	EscapeXML *strings.Replacer

	shell = "/bin/sh"

	defaultOwner = os.Getenv("USER")
)

func init() {
	// See: VixToolsEscapeXMLString
	chars := []string{
		`"`,
		"%",
		"&",
		"'",
		"<",
		">",
	}

	replace := make([]string, 0, len(chars)*2)

	for _, c := range chars {
		replace = append(replace, c)
		replace = append(replace, url.QueryEscape(c))
	}

	EscapeXML = strings.NewReplacer(replace...)

	// See procMgrPosix.c:ProcMgrStartProcess:
	// Prefer bash -c as is uses exec() to replace itself,
	// whereas bourne shell does a fork & exec, so two processes are started.
	if sh, err := exec.LookPath("bash"); err != nil {
		shell = sh
	}

	if defaultOwner == "" {
		defaultOwner = "toolbox"
	}
}

// IO encapsulates IO for Go functions and OS commands such that they can interact via the OperationsManager
// without file system disk IO.
type IO struct {
	In struct {
		io.Writer
		io.Reader
		io.Closer // Closer for the write side of the pipe, can be closed via hgfs ops (FileTranfserToGuest)
	}

	Out *bytes.Buffer
	Err *bytes.Buffer
}

// State is the toolbox representation of the GuestProcessInfo type
type State struct {
	StartTime int64 // (keep first to ensure 64-bit alignment)
	EndTime   int64 // (keep first to ensure 64-bit alignment)

	Name     string
	Args     string
	Owner    string
	Pid      int64
	ExitCode int32

	IO *IO
}

// WithIO enables toolbox Process IO without file system disk IO.
func (p *Process) WithIO() *Process {
	p.IO = &IO{
		Out: new(bytes.Buffer),
		Err: new(bytes.Buffer),
	}

	return p
}

// File implements the os.FileInfo interface to enable toolbox interaction with virtual files.
type File struct {
	io.Reader
	io.Writer
	io.Closer

	name string
	size int
}

// Name implementation of the os.FileInfo interface method.
func (a *File) Name() string {
	return a.name
}

// Size implementation of the os.FileInfo interface method.
func (a *File) Size() int64 {
	return int64(a.size)
}

// Mode implementation of the os.FileInfo interface method.
func (a *File) Mode() os.FileMode {
	if strings.HasSuffix(a.name, "stdin") {
		return 0200
	}
	return 0400
}

// ModTime implementation of the os.FileInfo interface method.
func (a *File) ModTime() time.Time {
	return time.Now()
}

// IsDir implementation of the os.FileInfo interface method.
func (a *File) IsDir() bool {
	return false
}

// Sys implementation of the os.FileInfo interface method.
func (a *File) Sys() interface{} {
	return nil
}

func (s *State) toXML() string {
	const format = "<proc>" +
		"<cmd>%s</cmd>" +
		"<name>%s</name>" +
		"<pid>%d</pid>" +
		"<user>%s</user>" +
		"<start>%d</start>" +
		"<eCode>%d</eCode>" +
		"<eTime>%d</eTime>" +
		"</proc>"

	name := filepath.Base(s.Name)

	argv := []string{s.Name}

	if len(s.Args) != 0 {
		argv = append(argv, EscapeXML.Replace(s.Args))
	}

	args := strings.Join(argv, " ")

	return fmt.Sprintf(format, name, args, s.Pid, s.Owner, s.StartTime, s.ExitCode, s.EndTime)
}

// Process managed by the process Manager.
type Process struct {
	State

	Start func(*Process, *vix.StartProgramRequest) (int64, error)
	Wait  func() error
	Kill  context.CancelFunc

	ctx context.Context
}

// Error can be returned by the Process.Wait function to propagate ExitCode to process State.
type Error struct {
	Err      error
	ExitCode int32
}

func (e *Error) Error() string {
	return e.Err.Error()
}

// Manager manages processes within the guest.
// See: http://pubs.vmware.com/vsphere-60/topic/com.vmware.wssdk.apiref.doc/vim.vm.guest.Manager.html
type Manager struct {
	wg      sync.WaitGroup
	mu      sync.Mutex
	expire  time.Duration
	entries map[int64]*Process
	pids    sync.Pool
}

// NewManager creates a new process Manager instance.
func NewManager() *Manager {
	// We use pseudo PIDs that don't conflict with OS PIDs, so they can live in the same table.
	// For the pseudo PIDs, we use a sync.Pool rather than a plain old counter to avoid the unlikely,
	// but possible wrapping should such a counter exceed MaxInt64.
	pid := int64(32768) // TODO: /proc/sys/kernel/pid_max

	return &Manager{
		expire:  time.Minute * 5,
		entries: make(map[int64]*Process),
		pids: sync.Pool{
			New: func() interface{} {
				return atomic.AddInt64(&pid, 1)
			},
		},
	}
}

// Start calls the Process.Start function, returning the pid on success or an error.
// A goroutine is started that calls the Process.Wait function.  After Process.Wait has
// returned, the process State EndTime and ExitCode fields are set.  The process state can be
// queried via ListProcessesInGuest until it is removed, 5 minutes after Wait returns.
func (m *Manager) Start(r *vix.StartProgramRequest, p *Process) (int64, error) {
	p.Name = r.ProgramPath
	p.Args = r.Arguments

	// Owner is cosmetic, but useful for example with: govc guest.ps -U $uid
	if p.Owner == "" {
		p.Owner = defaultOwner
	}

	p.StartTime = time.Now().Unix()

	p.ctx, p.Kill = context.WithCancel(context.Background())

	pid, err := p.Start(p, r)
	if err != nil {
		return -1, err
	}

	if pid == 0 {
		p.Pid = m.pids.Get().(int64) // pseudo pid for funcs
	} else {
		p.Pid = pid
	}

	m.mu.Lock()
	m.entries[p.Pid] = p
	m.mu.Unlock()

	m.wg.Add(1)
	go func() {
		werr := p.Wait()

		m.mu.Lock()
		p.EndTime = time.Now().Unix()

		if werr != nil {
			rc := int32(1)
			if xerr, ok := werr.(*Error); ok {
				rc = xerr.ExitCode
			}

			p.ExitCode = rc
		}

		m.mu.Unlock()
		m.wg.Done()
		p.Kill() // cancel context for those waiting on p.ctx.Done()

		// See: http://pubs.vmware.com/vsphere-65/topic/com.vmware.wssdk.apiref.doc/vim.vm.guest.ProcessManager.ProcessInfo.html
		// "If the process was started using StartProgramInGuest then the process completion time
		//  will be available if queried within 5 minutes after it completes."
		<-time.After(m.expire)

		m.mu.Lock()
		delete(m.entries, p.Pid)
		m.mu.Unlock()

		if pid == 0 {
			m.pids.Put(p.Pid) // pseudo pid can be reused now
		}
	}()

	return p.Pid, nil
}

// Kill cancels the Process Context.
// Returns true if pid exists in the process table, false otherwise.
func (m *Manager) Kill(pid int64) bool {
	m.mu.Lock()
	entry, ok := m.entries[pid]
	m.mu.Unlock()

	if ok {
		entry.Kill()
		return true
	}

	return false
}

// ListProcesses marshals the process State for the given pids.
// If no pids are specified, all current processes are included.
// The return value can be used for responding to a VixMsgListProcessesExRequest.
func (m *Manager) ListProcesses(pids []int64) []byte {
	w := new(bytes.Buffer)

	for _, p := range m.List(pids) {
		_, _ = w.WriteString(p.toXML())
	}

	return w.Bytes()
}

// List the process State for the given pids.
func (m *Manager) List(pids []int64) []State {
	var list []State

	m.mu.Lock()

	if len(pids) == 0 {
		for _, p := range m.entries {
			list = append(list, p.State)
		}
	} else {
		for _, id := range pids {
			p, ok := m.entries[id]
			if !ok {
				continue
			}

			list = append(list, p.State)
		}
	}

	m.mu.Unlock()

	return list
}

type procFileInfo struct {
	os.FileInfo
}

// Size returns hgfs.LargePacketMax such that InitiateFileTransferFromGuest can download a /proc/ file from the guest.
// If we were to return the size '0' here, then a 'Content-Length: 0' header is returned by VC/ESX.
func (p procFileInfo) Size() int64 {
	return hgfs.LargePacketMax // Remember, Sully, when I promised to kill you last?  I lied.
}

// Stat implements hgfs.FileHandler.Stat
func (m *Manager) Stat(u *url.URL) (os.FileInfo, error) {
	name := path.Join("/proc", u.Path)

	info, err := os.Stat(name)
	if err == nil && info.Size() == 0 {
		// This is a real /proc file
		return &procFileInfo{info}, nil
	}

	dir, file := path.Split(u.Path)

	pid, err := strconv.ParseInt(path.Base(dir), 10, 64)
	if err != nil {
		return nil, os.ErrNotExist
	}

	m.mu.Lock()
	p := m.entries[pid]
	m.mu.Unlock()

	if p == nil || p.IO == nil {
		return nil, os.ErrNotExist
	}

	pf := &File{
		name:   name,
		Closer: ioutil.NopCloser(nil), // via hgfs, nop for stdout and stderr
	}

	var r *bytes.Buffer

	switch file {
	case "stdin":
		pf.Writer = p.IO.In.Writer
		pf.Closer = p.IO.In.Closer
		return pf, nil
	case "stdout":
		r = p.IO.Out
	case "stderr":
		r = p.IO.Err
	default:
		return nil, os.ErrNotExist
	}

	select {
	case <-p.ctx.Done():
	case <-time.After(time.Second):
		// The vmx guest RPC calls are queue based, serialized on the vmx side.
		// There are 5 seconds between "ping" RPC calls and after a few misses,
		// the vmx considers tools as not running.  In this case, the vmx would timeout
		// a file transfer after 60 seconds.
		//
		// vix.FileAccessError is converted to a CannotAccessFile fault,
		// so the client can choose to retry the transfer in this case.
		// Would have preferred vix.ObjectIsBusy (EBUSY), but VC/ESX converts that
		// to a general SystemErrorFault with nothing but a localized string message
		// to check against: "<reason>vix error codes = (5, 0).</reason>"
		// Is standard vmware-tools, EACCES is converted to a CannotAccessFile fault.
		return nil, vix.Error(vix.FileAccessError)
	}

	pf.Reader = r
	pf.size = r.Len()

	return pf, nil
}

// Open implements hgfs.FileHandler.Open
func (m *Manager) Open(u *url.URL, mode int32) (hgfs.File, error) {
	info, err := m.Stat(u)
	if err != nil {
		return nil, err
	}

	pinfo, ok := info.(*File)

	if !ok {
		return nil, os.ErrNotExist // fall through to default os.Open
	}

	switch path.Base(u.Path) {
	case "stdin":
		if mode != hgfs.OpenModeWriteOnly {
			return nil, vix.Error(vix.InvalidArg)
		}
	case "stdout", "stderr":
		if mode != hgfs.OpenModeReadOnly {
			return nil, vix.Error(vix.InvalidArg)
		}
	}

	return pinfo, nil
}

type processFunc struct {
	wg sync.WaitGroup

	run func(context.Context, string) error

	err error
}

// NewFunc creates a new Process, where the Start function calls the given run function within a goroutine.
// The Wait function waits for the goroutine to finish and returns the error returned by run.
// The run ctx param may be used to return early via the process Manager.Kill method.
// The run args command is that of the VixMsgStartProgramRequest.Arguments field.
func NewFunc(run func(ctx context.Context, args string) error) *Process {
	f := &processFunc{run: run}

	return &Process{
		Start: f.start,
		Wait:  f.wait,
	}
}

// FuncIO is the Context key to access optional ProcessIO
var FuncIO = struct {
	key int64
}{vix.CommandMagicWord}

func (f *processFunc) start(p *Process, r *vix.StartProgramRequest) (int64, error) {
	f.wg.Add(1)

	var c io.Closer

	if p.IO != nil {
		pr, pw := io.Pipe()

		p.IO.In.Reader, p.IO.In.Writer = pr, pw
		c, p.IO.In.Closer = pr, pw

		p.ctx = context.WithValue(p.ctx, FuncIO, p.IO)
	}

	go func() {
		f.err = f.run(p.ctx, r.Arguments)

		if p.IO != nil {
			_ = c.Close()

			if f.err != nil && p.IO.Err.Len() == 0 {
				p.IO.Err.WriteString(f.err.Error())
			}
		}

		f.wg.Done()
	}()

	return 0, nil
}

func (f *processFunc) wait() error {
	f.wg.Wait()
	return f.err
}

type processCmd struct {
	cmd *exec.Cmd
}

// New creates a new Process, where the Start function use exec.CommandContext to create and start the process.
// The Wait function waits for the process to finish and returns the error returned by exec.Cmd.Wait().
// Prior to Wait returning, the exec.Cmd.Wait() error is used to set the Process.ExitCode, if error is of type exec.ExitError.
// The ctx param may be used to kill the process via the process Manager.Kill method.
// The VixMsgStartProgramRequest param fields are mapped to the exec.Cmd counterpart fields.
// Processes are started within a sub-shell, allowing for i/o redirection, just as with the C version of vmware-tools.
func New() *Process {
	c := new(processCmd)

	return &Process{
		Start: c.start,
		Wait:  c.wait,
	}
}

func (c *processCmd) start(p *Process, r *vix.StartProgramRequest) (int64, error) {
	name, err := exec.LookPath(r.ProgramPath)
	if err != nil {
		return -1, err
	}
	// #nosec: Subprocess launching with variable
	// Note that processCmd is currently used only for testing.
	c.cmd = exec.CommandContext(p.ctx, shell, "-c", fmt.Sprintf("%s %s", name, r.Arguments))
	c.cmd.Dir = r.WorkingDir
	c.cmd.Env = r.EnvVars

	if p.IO != nil {
		in, perr := c.cmd.StdinPipe()
		if perr != nil {
			return -1, perr
		}

		p.IO.In.Writer = in
		p.IO.In.Closer = in

		// Note we currently use a Buffer in addition to the os.Pipe so that:
		// - Stat() can provide a size
		// - FileTransferFromGuest won't block
		// - Can't use the exec.Cmd.Std{out,err}Pipe methods since Wait() closes the pipes.
		//   We could use os.Pipe directly, but toolbox needs to take care of closing both ends,
		//   but also need to prevent FileTransferFromGuest from blocking.
		c.cmd.Stdout = p.IO.Out
		c.cmd.Stderr = p.IO.Err
	}

	err = c.cmd.Start()
	if err != nil {
		return -1, err
	}

	return int64(c.cmd.Process.Pid), nil
}

func (c *processCmd) wait() error {
	err := c.cmd.Wait()
	if err != nil {
		xerr := &Error{
			Err:      err,
			ExitCode: 1,
		}

		if x, ok := err.(*exec.ExitError); ok {
			if status, ok := x.Sys().(syscall.WaitStatus); ok {
				xerr.ExitCode = int32(status.ExitStatus())
			}
		}

		return xerr
	}

	return nil
}

// NewRoundTrip starts a Go function to implement a toolbox backed http.RoundTripper
func NewRoundTrip() *Process {
	return NewFunc(func(ctx context.Context, host string) error {
		p, _ := ctx.Value(FuncIO).(*IO)

		closers := []io.Closer{p.In.Closer}

		defer func() {
			for _, c := range closers {
				_ = c.Close()
			}
		}()

		c, err := new(net.Dialer).DialContext(ctx, "tcp", host)
		if err != nil {
			return err
		}

		closers = append(closers, c)

		go func() {
			<-ctx.Done()
			if ctx.Err() == context.DeadlineExceeded {
				_ = c.Close()
			}
		}()

		_, err = io.Copy(c, p.In.Reader)
		if err != nil {
			return err
		}

		_, err = io.Copy(p.Out, c)
		if err != nil {
			return err
		}

		return nil
	}).WithIO()
}
