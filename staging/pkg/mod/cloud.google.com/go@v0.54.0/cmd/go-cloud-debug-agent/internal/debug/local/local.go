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

// +build linux

// Package local provides access to a local program.
package local

import (
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/server"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/server/protocol"
)

var _ debug.Program = (*Program)(nil)
var _ debug.File = (*File)(nil)

// Program implements the debug.Program interface.
// Through that interface it provides access to a program being debugged.
type Program struct {
	s *server.Server
}

// New creates a new program from the specified file.
// The program can then be started by the Run method.
func New(textFile string) (*Program, error) {
	s, err := server.New(textFile)
	return &Program{s: s}, err
}

func (p *Program) Open(name string, mode string) (debug.File, error) {
	req := protocol.OpenRequest{
		Name: name,
		Mode: mode,
	}
	var resp protocol.OpenResponse
	err := p.s.Open(&req, &resp)
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
	err := p.s.Run(&req, &resp)
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
	err := p.s.Resume(&req, &resp)
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
	err := p.s.Breakpoint(&req, &resp)
	return resp.PCs, err
}

func (p *Program) BreakpointAtFunction(name string) ([]uint64, error) {
	req := protocol.BreakpointAtFunctionRequest{
		Function: name,
	}
	var resp protocol.BreakpointResponse
	err := p.s.BreakpointAtFunction(&req, &resp)
	return resp.PCs, err
}

func (p *Program) BreakpointAtLine(file string, line uint64) ([]uint64, error) {
	req := protocol.BreakpointAtLineRequest{
		File: file,
		Line: line,
	}
	var resp protocol.BreakpointResponse
	err := p.s.BreakpointAtLine(&req, &resp)
	return resp.PCs, err
}

func (p *Program) DeleteBreakpoints(pcs []uint64) error {
	req := protocol.DeleteBreakpointsRequest{PCs: pcs}
	var resp protocol.DeleteBreakpointsResponse
	return p.s.DeleteBreakpoints(&req, &resp)
}

func (p *Program) Eval(expr string) ([]string, error) {
	req := protocol.EvalRequest{
		Expr: expr,
	}
	var resp protocol.EvalResponse
	err := p.s.Eval(&req, &resp)
	return resp.Result, err
}

func (p *Program) Evaluate(e string) (debug.Value, error) {
	req := protocol.EvaluateRequest{
		Expression: e,
	}
	var resp protocol.EvaluateResponse
	err := p.s.Evaluate(&req, &resp)
	return resp.Result, err
}

func (p *Program) Frames(count int) ([]debug.Frame, error) {
	req := protocol.FramesRequest{
		Count: count,
	}
	var resp protocol.FramesResponse
	err := p.s.Frames(&req, &resp)
	return resp.Frames, err
}

func (p *Program) Goroutines() ([]*debug.Goroutine, error) {
	req := protocol.GoroutinesRequest{}
	var resp protocol.GoroutinesResponse
	err := p.s.Goroutines(&req, &resp)
	return resp.Goroutines, err
}

func (p *Program) VarByName(name string) (debug.Var, error) {
	req := protocol.VarByNameRequest{Name: name}
	var resp protocol.VarByNameResponse
	err := p.s.VarByName(&req, &resp)
	return resp.Var, err
}

func (p *Program) Value(v debug.Var) (debug.Value, error) {
	req := protocol.ValueRequest{Var: v}
	var resp protocol.ValueResponse
	err := p.s.Value(&req, &resp)
	return resp.Value, err
}

func (p *Program) MapElement(m debug.Map, index uint64) (debug.Var, debug.Var, error) {
	req := protocol.MapElementRequest{Map: m, Index: index}
	var resp protocol.MapElementResponse
	err := p.s.MapElement(&req, &resp)
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
	err := f.prog.s.ReadAt(&req, &resp)
	return copy(p, resp.Data), err
}

func (f *File) WriteAt(p []byte, offset int64) (int, error) {
	panic("unimplemented")
}

func (f *File) Close() error {
	req := protocol.CloseRequest{
		FD: f.fd,
	}
	var resp protocol.CloseResponse
	err := f.prog.s.Close(&req, &resp)
	return err
}
