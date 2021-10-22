// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package remote provides remote access to a debugproxy server.
package remote

import (
	"fmt"
	"io"
	"net/rpc"
	"os"
	"os/exec"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/server/protocol"
)

var _ debug.Program = (*Program)(nil)
var _ debug.File = (*File)(nil)

// Program implements the debug.Program interface.
// Through that interface it provides access to a program being
// debugged on a possibly remote machine by communicating
// with a debugproxy adjacent to the target program.
type Program struct {
	client *rpc.Client
}

// DebugproxyCmd is the path to the debugproxy command. It is a variable in case
// the default value, "debugproxy", is not in the $PATH.
var DebugproxyCmd = "debugproxy"

// New connects to the specified host using SSH, starts DebugproxyCmd
// there, and creates a new program from the specified file.
// The program can then be started by the Run method.
func New(host string, textFile string) (*Program, error) {
	// TODO: add args.
	cmdStrs := []string{"/usr/bin/ssh", host, DebugproxyCmd, "-text", textFile}
	if host == "localhost" {
		cmdStrs = cmdStrs[2:]
	}
	cmd := exec.Command(cmdStrs[0], cmdStrs[1:]...)
	stdin, toStdin, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	fromStdout, stdout, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = os.Stderr // Stderr from proxy appears on our stderr.
	err = cmd.Start()
	if err != nil {
		return nil, err
	}
	stdout.Close()
	if msg, err := readLine(fromStdout); err != nil {
		return nil, err
	} else if msg != "OK" {
		// Communication error.
		return nil, fmt.Errorf("unrecognized message %q", msg)
	}
	program := &Program{
		client: rpc.NewClient(&rwc{
			ssh: cmd,
			r:   fromStdout,
			w:   toStdin,
		}),
	}
	return program, nil
}

// readLine reads one line of text from the reader. It does no buffering.
// The trailing newline is read but not returned.
func readLine(r io.Reader) (string, error) {
	b := make([]byte, 0, 10)
	var c [1]byte
	for {
		_, err := io.ReadFull(r, c[:])
		if err != nil {
			return "", err
		}
		if c[0] == '\n' {
			break
		}
		b = append(b, c[0])
	}
	return string(b), nil
}

// rwc creates a single io.ReadWriteCloser from a read side and a write side.
// It also holds the command object so we can wait for SSH to complete.
// It allows us to do RPC over an SSH connection.
type rwc struct {
	ssh *exec.Cmd
	r   *os.File
	w   *os.File
}

func (rwc *rwc) Read(p []byte) (int, error) {
	return rwc.r.Read(p)
}

func (rwc *rwc) Write(p []byte) (int, error) {
	return rwc.w.Write(p)
}

func (rwc *rwc) Close() error {
	rerr := rwc.r.Close()
	werr := rwc.w.Close()
	cerr := rwc.ssh.Wait()
	if cerr != nil {
		// Wait exit status is most important.
		return cerr
	}
	if rerr != nil {
		return rerr
	}
	return werr
}

func (p *Program) Open(name string, mode string) (debug.File, error) {
	req := protocol.OpenRequest{
		Name: name,
		Mode: mode,
	}
	var resp protocol.OpenResponse
	err := p.client.Call("Server.Open", &req, &resp)
	if err != nil {
		return nil, err
	}
	f := &File{
		prog: p,
		fd:   resp.FD,
	}
	return f, nil
}

func (p *Program) Run(args ...string) (debug.Status, error) {
	req := protocol.RunRequest{args}
	var resp protocol.RunResponse
	err := p.client.Call("Server.Run", &req, &resp)
	if err != nil {
		return debug.Status{}, err
	}
	return resp.Status, nil
}

func (p *Program) Stop() (debug.Status, error) {
	panic("unimplemented")
}

func (p *Program) Resume() (debug.Status, error) {
	req := protocol.ResumeRequest{}
	var resp protocol.ResumeResponse
	err := p.client.Call("Server.Resume", &req, &resp)
	if err != nil {
		return debug.Status{}, err
	}
	return resp.Status, nil
}

func (p *Program) Kill() (debug.Status, error) {
	panic("unimplemented")
}

func (p *Program) Breakpoint(address uint64) ([]uint64, error) {
	req := protocol.BreakpointRequest{
		Address: address,
	}
	var resp protocol.BreakpointResponse
	err := p.client.Call("Server.Breakpoint", &req, &resp)
	return resp.PCs, err
}

func (p *Program) BreakpointAtFunction(name string) ([]uint64, error) {
	req := protocol.BreakpointAtFunctionRequest{
		Function: name,
	}
	var resp protocol.BreakpointResponse
	err := p.client.Call("Server.BreakpointAtFunction", &req, &resp)
	return resp.PCs, err
}

func (p *Program) BreakpointAtLine(file string, line uint64) ([]uint64, error) {
	req := protocol.BreakpointAtLineRequest{
		File: file,
		Line: line,
	}
	var resp protocol.BreakpointResponse
	err := p.client.Call("Server.BreakpointAtLine", &req, &resp)
	return resp.PCs, err
}

func (p *Program) DeleteBreakpoints(pcs []uint64) error {
	req := protocol.DeleteBreakpointsRequest{PCs: pcs}
	var resp protocol.DeleteBreakpointsResponse
	return p.client.Call("Server.DeleteBreakpoints", &req, &resp)
}

func (p *Program) Eval(expr string) ([]string, error) {
	req := protocol.EvalRequest{
		Expr: expr,
	}
	var resp protocol.EvalResponse
	err := p.client.Call("Server.Eval", &req, &resp)
	return resp.Result, err
}

func (p *Program) Evaluate(e string) (debug.Value, error) {
	req := protocol.EvaluateRequest{
		Expression: e,
	}
	var resp protocol.EvaluateResponse
	err := p.client.Call("Server.Evaluate", &req, &resp)
	return resp.Result, err
}

func (p *Program) Frames(count int) ([]debug.Frame, error) {
	req := protocol.FramesRequest{
		Count: count,
	}
	var resp protocol.FramesResponse
	err := p.client.Call("Server.Frames", &req, &resp)
	return resp.Frames, err
}

func (p *Program) Goroutines() ([]*debug.Goroutine, error) {
	req := protocol.GoroutinesRequest{}
	var resp protocol.GoroutinesResponse
	err := p.client.Call("Server.Goroutines", &req, &resp)
	return resp.Goroutines, err
}

func (p *Program) VarByName(name string) (debug.Var, error) {
	req := protocol.VarByNameRequest{Name: name}
	var resp protocol.VarByNameResponse
	err := p.client.Call("Server.VarByName", &req, &resp)
	return resp.Var, err
}

func (p *Program) Value(v debug.Var) (debug.Value, error) {
	req := protocol.ValueRequest{Var: v}
	var resp protocol.ValueResponse
	err := p.client.Call("Server.Value", &req, &resp)
	return resp.Value, err
}

func (p *Program) MapElement(m debug.Map, index uint64) (debug.Var, debug.Var, error) {
	req := protocol.MapElementRequest{Map: m, Index: index}
	var resp protocol.MapElementResponse
	err := p.client.Call("Server.MapElement", &req, &resp)
	return resp.Key, resp.Value, err
}

// File implements the debug.File interface, providing access
// to file-like resources associated with the target program.
type File struct {
	prog *Program // The Program associated with the file.
	fd   int      // File descriptor.
}

func (f *File) ReadAt(p []byte, offset int64) (int, error) {
	req := protocol.ReadAtRequest{
		FD:     f.fd,
		Len:    len(p),
		Offset: offset,
	}
	var resp protocol.ReadAtResponse
	err := f.prog.client.Call("Server.ReadAt", &req, &resp)
	return copy(p, resp.Data), err
}

func (f *File) WriteAt(p []byte, offset int64) (int, error) {
	req := protocol.WriteAtRequest{
		FD:     f.fd,
		Data:   p,
		Offset: offset,
	}
	var resp protocol.WriteAtResponse
	err := f.prog.client.Call("Server.WriteAt", &req, &resp)
	return resp.Len, err
}

func (f *File) Close() error {
	req := protocol.CloseRequest{
		FD: f.fd,
	}
	var resp protocol.CloseResponse
	err := f.prog.client.Call("Server.Close", &req, &resp)
	return err
}
