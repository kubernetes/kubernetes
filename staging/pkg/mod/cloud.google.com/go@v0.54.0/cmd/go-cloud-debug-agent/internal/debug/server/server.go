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

// Package server provides RPC access to a local program being debugged.
// It is the remote end of the client implementation of the Program interface.
package server

//go:generate sh -c "m4 -P eval.m4 > eval.go"

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/arch"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/elf"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/server/protocol"
)

type breakpoint struct {
	pc        uint64
	origInstr [arch.MaxBreakpointSize]byte
}

type call struct {
	req, resp interface{}
	errc      chan error
}

type Server struct {
	arch       arch.Architecture
	executable string // Name of executable.
	dwarfData  *dwarf.Data

	breakpointc chan call
	otherc      chan call

	fc chan func() error
	ec chan error

	proc            *os.Process
	procIsUp        bool
	stoppedPid      int
	stoppedRegs     syscall.PtraceRegs
	topOfStackAddrs []uint64
	breakpoints     map[uint64]breakpoint
	files           []*file // Index == file descriptor.
	printer         *Printer

	// goroutineStack reads the stack of a (non-running) goroutine.
	goroutineStack     func(uint64) ([]debug.Frame, error)
	goroutineStackOnce sync.Once
}

// peek implements the Peeker interface required by the printer.
func (s *Server) peek(offset uintptr, buf []byte) error {
	return s.ptracePeek(s.stoppedPid, offset, buf)
}

// New parses the executable and builds local data structures for answering requests.
// It returns a Server ready to serve requests about the executable.
func New(executable string) (*Server, error) {
	fd, err := os.Open(executable)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	architecture, dwarfData, err := loadExecutable(fd)
	if err != nil {
		return nil, err
	}
	srv := &Server{
		arch:        *architecture,
		executable:  executable,
		dwarfData:   dwarfData,
		breakpointc: make(chan call),
		otherc:      make(chan call),
		fc:          make(chan func() error),
		ec:          make(chan error),
		breakpoints: make(map[uint64]breakpoint),
	}
	srv.printer = NewPrinter(architecture, dwarfData, srv)
	go ptraceRun(srv.fc, srv.ec)
	go srv.loop()
	return srv, nil
}

func loadExecutable(f *os.File) (*arch.Architecture, *dwarf.Data, error) {
	// TODO: How do we detect NaCl?
	if obj, err := elf.NewFile(f); err == nil {
		dwarfData, err := obj.DWARF()
		if err != nil {
			return nil, nil, err
		}

		switch obj.Machine {
		case elf.EM_ARM:
			return &arch.ARM, dwarfData, nil
		case elf.EM_386:
			switch obj.Class {
			case elf.ELFCLASS32:
				return &arch.X86, dwarfData, nil
			case elf.ELFCLASS64:
				return &arch.AMD64, dwarfData, nil
			}
		case elf.EM_X86_64:
			return &arch.AMD64, dwarfData, nil
		}
		return nil, nil, fmt.Errorf("unrecognized ELF architecture")
	}
	return nil, nil, fmt.Errorf("unrecognized binary format")
}

func (s *Server) loop() {
	for {
		var c call
		select {
		case c = <-s.breakpointc:
		case c = <-s.otherc:
		}
		s.dispatch(c)
	}
}

func (s *Server) dispatch(c call) {
	switch req := c.req.(type) {
	case *protocol.BreakpointRequest:
		c.errc <- s.handleBreakpoint(req, c.resp.(*protocol.BreakpointResponse))
	case *protocol.BreakpointAtFunctionRequest:
		c.errc <- s.handleBreakpointAtFunction(req, c.resp.(*protocol.BreakpointResponse))
	case *protocol.BreakpointAtLineRequest:
		c.errc <- s.handleBreakpointAtLine(req, c.resp.(*protocol.BreakpointResponse))
	case *protocol.DeleteBreakpointsRequest:
		c.errc <- s.handleDeleteBreakpoints(req, c.resp.(*protocol.DeleteBreakpointsResponse))
	case *protocol.CloseRequest:
		c.errc <- s.handleClose(req, c.resp.(*protocol.CloseResponse))
	case *protocol.EvalRequest:
		c.errc <- s.handleEval(req, c.resp.(*protocol.EvalResponse))
	case *protocol.EvaluateRequest:
		c.errc <- s.handleEvaluate(req, c.resp.(*protocol.EvaluateResponse))
	case *protocol.FramesRequest:
		c.errc <- s.handleFrames(req, c.resp.(*protocol.FramesResponse))
	case *protocol.OpenRequest:
		c.errc <- s.handleOpen(req, c.resp.(*protocol.OpenResponse))
	case *protocol.ReadAtRequest:
		c.errc <- s.handleReadAt(req, c.resp.(*protocol.ReadAtResponse))
	case *protocol.ResumeRequest:
		c.errc <- s.handleResume(req, c.resp.(*protocol.ResumeResponse))
	case *protocol.RunRequest:
		c.errc <- s.handleRun(req, c.resp.(*protocol.RunResponse))
	case *protocol.VarByNameRequest:
		c.errc <- s.handleVarByName(req, c.resp.(*protocol.VarByNameResponse))
	case *protocol.ValueRequest:
		c.errc <- s.handleValue(req, c.resp.(*protocol.ValueResponse))
	case *protocol.MapElementRequest:
		c.errc <- s.handleMapElement(req, c.resp.(*protocol.MapElementResponse))
	case *protocol.GoroutinesRequest:
		c.errc <- s.handleGoroutines(req, c.resp.(*protocol.GoroutinesResponse))
	default:
		panic(fmt.Sprintf("unexpected call request type %T", c.req))
	}
}

func (s *Server) call(c chan call, req, resp interface{}) error {
	errc := make(chan error)
	c <- call{req, resp, errc}
	return <-errc
}

type file struct {
	mode  string
	index int
	f     debug.File
}

func (s *Server) Open(req *protocol.OpenRequest, resp *protocol.OpenResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleOpen(req *protocol.OpenRequest, resp *protocol.OpenResponse) error {
	// TODO: Better simulation. For now we just open the named OS file.
	var flag int
	switch req.Mode {
	case "r":
		flag = os.O_RDONLY
	case "w":
		flag = os.O_WRONLY
	case "rw":
		flag = os.O_RDWR
	default:
		return fmt.Errorf("Open: bad open mode %q", req.Mode)
	}
	osFile, err := os.OpenFile(req.Name, flag, 0)
	if err != nil {
		return err
	}
	// Find a file descriptor (index) slot.
	index := 0
	for ; index < len(s.files) && s.files[index] != nil; index++ {
	}
	f := &file{
		mode:  req.Mode,
		index: index,
		f:     osFile,
	}
	if index == len(s.files) {
		s.files = append(s.files, f)
	} else {
		s.files[index] = f
	}
	return nil
}

func (s *Server) ReadAt(req *protocol.ReadAtRequest, resp *protocol.ReadAtResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleReadAt(req *protocol.ReadAtRequest, resp *protocol.ReadAtResponse) error {
	fd := req.FD
	if fd < 0 || len(s.files) <= fd || s.files[fd] == nil {
		return fmt.Errorf("ReadAt: bad file descriptor %d", fd)
	}
	f := s.files[fd]
	buf := make([]byte, req.Len) // TODO: Don't allocate every time
	n, err := f.f.ReadAt(buf, req.Offset)
	resp.Data = buf[:n]
	return err
}

func (s *Server) Close(req *protocol.CloseRequest, resp *protocol.CloseResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleClose(req *protocol.CloseRequest, resp *protocol.CloseResponse) error {
	fd := req.FD
	if fd < 0 || fd >= len(s.files) || s.files[fd] == nil {
		return fmt.Errorf("Close: bad file descriptor %d", fd)
	}
	err := s.files[fd].f.Close()
	// Remove it regardless
	s.files[fd] = nil
	return err
}

func (s *Server) Run(req *protocol.RunRequest, resp *protocol.RunResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleRun(req *protocol.RunRequest, resp *protocol.RunResponse) error {
	if s.proc != nil {
		s.proc.Kill()
		s.proc = nil
		s.procIsUp = false
		s.stoppedPid = 0
		s.stoppedRegs = syscall.PtraceRegs{}
		s.topOfStackAddrs = nil
	}
	argv := append([]string{s.executable}, req.Args...)
	p, err := s.startProcess(s.executable, argv, &os.ProcAttr{
		Files: []*os.File{
			nil,       // TODO: be able to feed the target's stdin.
			os.Stderr, // TODO: be able to capture the target's stdout.
			os.Stderr,
		},
		Sys: &syscall.SysProcAttr{
			Pdeathsig: syscall.SIGKILL,
			Ptrace:    true,
		},
	})
	if err != nil {
		return err
	}
	s.proc = p
	s.stoppedPid = p.Pid
	return nil
}

func (s *Server) Resume(req *protocol.ResumeRequest, resp *protocol.ResumeResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleResume(req *protocol.ResumeRequest, resp *protocol.ResumeResponse) error {
	if s.proc == nil {
		return fmt.Errorf("Resume: Run did not successfully start a process")
	}

	if !s.procIsUp {
		s.procIsUp = true
		if _, err := s.waitForTrap(s.stoppedPid, false); err != nil {
			return err
		}
		if err := s.ptraceSetOptions(s.stoppedPid, syscall.PTRACE_O_TRACECLONE); err != nil {
			return fmt.Errorf("ptraceSetOptions: %v", err)
		}
	} else if _, ok := s.breakpoints[s.stoppedRegs.Rip]; ok {
		if err := s.ptraceSingleStep(s.stoppedPid); err != nil {
			return fmt.Errorf("ptraceSingleStep: %v", err)
		}
		if _, err := s.waitForTrap(s.stoppedPid, false); err != nil {
			return err
		}
	}

	for {
		if err := s.setBreakpoints(); err != nil {
			return err
		}
		if err := s.ptraceCont(s.stoppedPid, 0); err != nil {
			return fmt.Errorf("ptraceCont: %v", err)
		}

		wpid, err := s.waitForTrap(-1, true)
		if err == nil {
			s.stoppedPid = wpid
			break
		}
		bce, ok := err.(*breakpointsChangedError)
		if !ok {
			return err
		}

		if err := syscall.Kill(s.stoppedPid, syscall.SIGSTOP); err != nil {
			return fmt.Errorf("kill(SIGSTOP): %v", err)
		}
		_, status, err := s.wait(s.stoppedPid, false)
		if err != nil {
			return fmt.Errorf("wait (after SIGSTOP): %v", err)
		}
		if !status.Stopped() || status.StopSignal() != syscall.SIGSTOP {
			return fmt.Errorf("wait (after SIGSTOP): unexpected wait status 0x%x", status)
		}

		if err := s.liftBreakpoints(); err != nil {
			return err
		}

	loop:
		for c := bce.call; ; {
			s.dispatch(c)
			select {
			case c = <-s.breakpointc:
			default:
				break loop
			}
		}
	}
	if err := s.liftBreakpoints(); err != nil {
		return err
	}

	if err := s.ptraceGetRegs(s.stoppedPid, &s.stoppedRegs); err != nil {
		return fmt.Errorf("ptraceGetRegs: %v", err)
	}

	s.stoppedRegs.Rip -= uint64(s.arch.BreakpointSize)

	if err := s.ptraceSetRegs(s.stoppedPid, &s.stoppedRegs); err != nil {
		return fmt.Errorf("ptraceSetRegs: %v", err)
	}

	resp.Status.PC = s.stoppedRegs.Rip
	resp.Status.SP = s.stoppedRegs.Rsp
	return nil
}

func (s *Server) waitForTrap(pid int, allowBreakpointsChange bool) (wpid int, err error) {
	for {
		wpid, status, err := s.wait(pid, allowBreakpointsChange)
		if err != nil {
			if _, ok := err.(*breakpointsChangedError); !ok {
				err = fmt.Errorf("wait: %v", err)
			}
			return 0, err
		}
		if status.StopSignal() == syscall.SIGTRAP && status.TrapCause() != syscall.PTRACE_EVENT_CLONE {
			return wpid, nil
		}
		if status.StopSignal() == syscall.SIGPROF {
			err = s.ptraceCont(wpid, int(syscall.SIGPROF))
		} else {
			err = s.ptraceCont(wpid, 0) // TODO: non-zero when wait catches other signals?
		}
		if err != nil {
			return 0, fmt.Errorf("ptraceCont: %v", err)
		}
	}
}

func (s *Server) Breakpoint(req *protocol.BreakpointRequest, resp *protocol.BreakpointResponse) error {
	return s.call(s.breakpointc, req, resp)
}

func (s *Server) handleBreakpoint(req *protocol.BreakpointRequest, resp *protocol.BreakpointResponse) error {
	return s.addBreakpoints([]uint64{req.Address}, resp)
}

func (s *Server) BreakpointAtFunction(req *protocol.BreakpointAtFunctionRequest, resp *protocol.BreakpointResponse) error {
	return s.call(s.breakpointc, req, resp)
}

func (s *Server) handleBreakpointAtFunction(req *protocol.BreakpointAtFunctionRequest, resp *protocol.BreakpointResponse) error {
	pc, err := s.functionStartAddress(req.Function)
	if err != nil {
		return err
	}
	return s.addBreakpoints([]uint64{pc}, resp)
}

func (s *Server) BreakpointAtLine(req *protocol.BreakpointAtLineRequest, resp *protocol.BreakpointResponse) error {
	return s.call(s.breakpointc, req, resp)
}

func (s *Server) handleBreakpointAtLine(req *protocol.BreakpointAtLineRequest, resp *protocol.BreakpointResponse) error {
	if s.dwarfData == nil {
		return fmt.Errorf("no DWARF data")
	}
	if pcs, err := s.dwarfData.LineToBreakpointPCs(req.File, req.Line); err != nil {
		return err
	} else {
		return s.addBreakpoints(pcs, resp)
	}
}

// addBreakpoints adds breakpoints at the addresses in pcs, then stores pcs in the response.
func (s *Server) addBreakpoints(pcs []uint64, resp *protocol.BreakpointResponse) error {
	// Get the original code at each address with ptracePeek.
	bps := make([]breakpoint, 0, len(pcs))
	for _, pc := range pcs {
		if _, alreadySet := s.breakpoints[pc]; alreadySet {
			continue
		}
		var bp breakpoint
		if err := s.ptracePeek(s.stoppedPid, uintptr(pc), bp.origInstr[:s.arch.BreakpointSize]); err != nil {
			return fmt.Errorf("ptracePeek: %v", err)
		}
		bp.pc = pc
		bps = append(bps, bp)
	}
	// If all the peeks succeeded, update the list of breakpoints.
	for _, bp := range bps {
		s.breakpoints[bp.pc] = bp
	}
	resp.PCs = pcs
	return nil
}

func (s *Server) DeleteBreakpoints(req *protocol.DeleteBreakpointsRequest, resp *protocol.DeleteBreakpointsResponse) error {
	return s.call(s.breakpointc, req, resp)
}

func (s *Server) handleDeleteBreakpoints(req *protocol.DeleteBreakpointsRequest, resp *protocol.DeleteBreakpointsResponse) error {
	for _, pc := range req.PCs {
		delete(s.breakpoints, pc)
	}
	return nil
}

func (s *Server) setBreakpoints() error {
	for pc := range s.breakpoints {
		err := s.ptracePoke(s.stoppedPid, uintptr(pc), s.arch.BreakpointInstr[:s.arch.BreakpointSize])
		if err != nil {
			return fmt.Errorf("setBreakpoints: %v", err)
		}
	}
	return nil
}

func (s *Server) liftBreakpoints() error {
	for pc, breakpoint := range s.breakpoints {
		err := s.ptracePoke(s.stoppedPid, uintptr(pc), breakpoint.origInstr[:s.arch.BreakpointSize])
		if err != nil {
			return fmt.Errorf("liftBreakpoints: %v", err)
		}
	}
	return nil
}

func (s *Server) Eval(req *protocol.EvalRequest, resp *protocol.EvalResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleEval(req *protocol.EvalRequest, resp *protocol.EvalResponse) (err error) {
	resp.Result, err = s.eval(req.Expr)
	return err
}

// eval evaluates an expression.
// TODO: very weak.
func (s *Server) eval(expr string) ([]string, error) {
	switch {
	case strings.HasPrefix(expr, "re:"):
		// Regular expression. Return list of symbols.
		re, err := regexp.Compile(expr[3:])
		if err != nil {
			return nil, err
		}
		return s.dwarfData.LookupMatchingSymbols(re)

	case strings.HasPrefix(expr, "addr:"):
		// Symbol lookup. Return address.
		addr, err := s.functionStartAddress(expr[5:])
		if err != nil {
			return nil, err
		}
		return []string{fmt.Sprintf("%#x", addr)}, nil

	case strings.HasPrefix(expr, "val:"):
		// Symbol lookup. Return formatted value.
		value, err := s.printer.Sprint(expr[4:])
		if err != nil {
			return nil, err
		}
		return []string{value}, nil

	case strings.HasPrefix(expr, "src:"):
		// Numerical address. Return file.go:123.
		addr, err := strconv.ParseUint(expr[4:], 0, 0)
		if err != nil {
			return nil, err
		}
		file, line, err := s.lookupSource(addr)
		if err != nil {
			return nil, err
		}
		return []string{fmt.Sprintf("%s:%d", file, line)}, nil

	case len(expr) > 0 && '0' <= expr[0] && expr[0] <= '9':
		// Numerical address. Return symbol.
		addr, err := strconv.ParseUint(expr, 0, 0)
		if err != nil {
			return nil, err
		}
		entry, _, err := s.dwarfData.PCToFunction(addr)
		if err != nil {
			return nil, err
		}
		name, ok := entry.Val(dwarf.AttrName).(string)
		if !ok {
			return nil, fmt.Errorf("function at 0x%x has no name", addr)
		}
		return []string{name}, nil
	}

	return nil, fmt.Errorf("bad expression syntax: %q", expr)
}

func (s *Server) Evaluate(req *protocol.EvaluateRequest, resp *protocol.EvaluateResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleEvaluate(req *protocol.EvaluateRequest, resp *protocol.EvaluateResponse) (err error) {
	resp.Result, err = s.evalExpression(req.Expression, s.stoppedRegs.Rip, s.stoppedRegs.Rsp)
	return err
}

func (s *Server) lookupSource(pc uint64) (file string, line uint64, err error) {
	if s.dwarfData == nil {
		return
	}
	// TODO: The gosym equivalent also returns the relevant Func. Do that when
	// DWARF has the same facility.
	return s.dwarfData.PCToLine(pc)
}

func (s *Server) Frames(req *protocol.FramesRequest, resp *protocol.FramesResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleFrames(req *protocol.FramesRequest, resp *protocol.FramesResponse) error {
	// TODO: verify that we're stopped.
	if s.topOfStackAddrs == nil {
		if err := s.evaluateTopOfStackAddrs(); err != nil {
			return err
		}
	}

	regs := syscall.PtraceRegs{}
	err := s.ptraceGetRegs(s.stoppedPid, &regs)
	if err != nil {
		return err
	}
	resp.Frames, err = s.walkStack(regs.Rip, regs.Rsp, req.Count)
	return err
}

// walkStack returns up to the requested number of stack frames.
func (s *Server) walkStack(pc, sp uint64, count int) ([]debug.Frame, error) {
	var frames []debug.Frame

	var buf [8]byte
	b := new(bytes.Buffer)
	r := s.dwarfData.Reader()

	// TODO: handle walking over a split stack.
	for i := 0; i < count; i++ {
		b.Reset()
		file, line, err := s.dwarfData.PCToLine(pc)
		if err != nil {
			return frames, err
		}
		fpOffset, err := s.dwarfData.PCToSPOffset(pc)
		if err != nil {
			return frames, err
		}
		fp := sp + uint64(fpOffset)
		entry, funcEntry, err := s.dwarfData.PCToFunction(pc)
		if err != nil {
			return frames, err
		}
		frame := debug.Frame{
			PC:            pc,
			SP:            sp,
			File:          file,
			Line:          line,
			FunctionStart: funcEntry,
		}
		frame.Function, _ = entry.Val(dwarf.AttrName).(string)
		r.Seek(entry.Offset)
		for {
			entry, err := r.Next()
			if err != nil {
				return frames, err
			}
			if entry.Tag == 0 {
				break
			}
			// TODO: report variables we couldn't parse?
			if entry.Tag == dwarf.TagFormalParameter {
				if v, err := s.parseParameterOrLocal(entry, fp); err == nil {
					frame.Params = append(frame.Params, debug.Param(v))
				}
			}
			if entry.Tag == dwarf.TagVariable {
				if v, err := s.parseParameterOrLocal(entry, fp); err == nil {
					frame.Vars = append(frame.Vars, v)
				}
			}
		}
		frames = append(frames, frame)

		// Walk to the caller's PC and SP.
		if s.topOfStack(funcEntry) {
			break
		}
		err = s.ptracePeek(s.stoppedPid, uintptr(fp-uint64(s.arch.PointerSize)), buf[:s.arch.PointerSize])
		if err != nil {
			return frames, fmt.Errorf("ptracePeek: %v", err)
		}
		pc, sp = s.arch.Uintptr(buf[:s.arch.PointerSize]), fp
	}
	return frames, nil
}

// parseParameterOrLocal parses the entry for a function parameter or local
// variable, which are both specified the same way. fp contains the frame
// pointer, which is used to calculate the variable location.
func (s *Server) parseParameterOrLocal(entry *dwarf.Entry, fp uint64) (debug.LocalVar, error) {
	var v debug.LocalVar
	v.Name, _ = entry.Val(dwarf.AttrName).(string)
	if off, err := s.dwarfData.EntryTypeOffset(entry); err != nil {
		return v, err
	} else {
		v.Var.TypeID = uint64(off)
	}
	if i := entry.Val(dwarf.AttrLocation); i == nil {
		return v, fmt.Errorf("missing location description")
	} else if locationDescription, ok := i.([]uint8); !ok {
		return v, fmt.Errorf("unsupported location description")
	} else if offset, err := evalLocation(locationDescription); err != nil {
		return v, err
	} else {
		v.Var.Address = fp + uint64(offset)
	}
	return v, nil
}

func (s *Server) evaluateTopOfStackAddrs() error {
	var (
		lookup   func(name string) (uint64, error)
		indirect bool
		names    []string
	)
	if _, err := s.dwarfData.LookupVariable("runtime.rt0_goPC"); err != nil {
		// Look for a Go 1.3 binary (or earlier version).
		lookup, indirect, names = s.functionStartAddress, false, []string{
			"runtime.goexit",
			"runtime.mstart",
			"runtime.mcall",
			"runtime.morestack",
			"runtime.lessstack",
			"_rt0_go",
		}
	} else {
		// Look for a Go 1.4 binary (or later version).
		lookup = func(name string) (uint64, error) {
			entry, err := s.dwarfData.LookupVariable(name)
			if err != nil {
				return 0, err
			}
			return s.dwarfData.EntryLocation(entry)
		}
		indirect, names = true, []string{
			"runtime.goexitPC",
			"runtime.mstartPC",
			"runtime.mcallPC",
			"runtime.morestackPC",
			"runtime.rt0_goPC",
		}
	}
	// TODO: also look for runtime.externalthreadhandlerp, on Windows.

	addrs := make([]uint64, 0, len(names))
	for _, name := range names {
		addr, err := lookup(name)
		if err != nil {
			return err
		}
		addrs = append(addrs, addr)
	}

	if indirect {
		buf := make([]byte, s.arch.PointerSize)
		for i, addr := range addrs {
			if err := s.ptracePeek(s.stoppedPid, uintptr(addr), buf); err != nil {
				return fmt.Errorf("ptracePeek: %v", err)
			}
			addrs[i] = s.arch.Uintptr(buf)
		}
	}

	s.topOfStackAddrs = addrs
	return nil
}

// topOfStack is the out-of-process equivalent of runtime·topofstack.
func (s *Server) topOfStack(funcEntry uint64) bool {
	for _, addr := range s.topOfStackAddrs {
		if addr == funcEntry {
			return true
		}
	}
	return false
}

func (s *Server) VarByName(req *protocol.VarByNameRequest, resp *protocol.VarByNameResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleVarByName(req *protocol.VarByNameRequest, resp *protocol.VarByNameResponse) error {
	entry, err := s.dwarfData.LookupVariable(req.Name)
	if err != nil {
		return fmt.Errorf("variable %s: %s", req.Name, err)
	}

	loc, err := s.dwarfData.EntryLocation(entry)
	if err != nil {
		return fmt.Errorf("variable %s: %s", req.Name, err)
	}

	off, err := s.dwarfData.EntryTypeOffset(entry)
	if err != nil {
		return fmt.Errorf("variable %s: %s", req.Name, err)
	}

	resp.Var.TypeID = uint64(off)
	resp.Var.Address = loc
	return nil
}

func (s *Server) Value(req *protocol.ValueRequest, resp *protocol.ValueResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleValue(req *protocol.ValueRequest, resp *protocol.ValueResponse) error {
	t, err := s.dwarfData.Type(dwarf.Offset(req.Var.TypeID))
	if err != nil {
		return err
	}
	resp.Value, err = s.value(t, req.Var.Address)
	return err
}

func (s *Server) MapElement(req *protocol.MapElementRequest, resp *protocol.MapElementResponse) error {
	return s.call(s.otherc, req, resp)
}

func (s *Server) handleMapElement(req *protocol.MapElementRequest, resp *protocol.MapElementResponse) error {
	t, err := s.dwarfData.Type(dwarf.Offset(req.Map.TypeID))
	if err != nil {
		return err
	}
	m, ok := t.(*dwarf.MapType)
	if !ok {
		return fmt.Errorf("variable is not a map")
	}
	var count uint64
	// fn will be called for each element of the map.
	// When we reach the requested element, we fill in *resp and stop.
	// TODO: cache locations of elements.
	fn := func(keyAddr, valAddr uint64, keyType, valType dwarf.Type) bool {
		count++
		if count == req.Index+1 {
			resp.Key = debug.Var{TypeID: uint64(keyType.Common().Offset), Address: keyAddr}
			resp.Value = debug.Var{TypeID: uint64(valType.Common().Offset), Address: valAddr}
			return false
		}
		return true
	}
	if err := s.peekMapValues(m, req.Map.Address, fn); err != nil {
		return err
	}
	if count <= req.Index {
		// There weren't enough elements.
		return fmt.Errorf("map has no element %d", req.Index)
	}
	return nil
}

func (s *Server) Goroutines(req *protocol.GoroutinesRequest, resp *protocol.GoroutinesResponse) error {
	return s.call(s.otherc, req, resp)
}

const invalidStatus debug.GoroutineStatus = 99

var (
	gStatus = [...]debug.GoroutineStatus{
		0: debug.Queued,  // _Gidle
		1: debug.Queued,  // _Grunnable
		2: debug.Running, // _Grunning
		3: debug.Blocked, // _Gsyscall
		4: debug.Blocked, // _Gwaiting
		5: invalidStatus, // _Gmoribund_unused
		6: invalidStatus, // _Gdead
		7: invalidStatus, // _Genqueue
		8: debug.Running, // _Gcopystack
	}
	gScanStatus = [...]debug.GoroutineStatus{
		0: invalidStatus, // _Gscan + _Gidle
		1: debug.Queued,  // _Gscanrunnable
		2: debug.Running, // _Gscanrunning
		3: debug.Blocked, // _Gscansyscall
		4: debug.Blocked, // _Gscanwaiting
		5: invalidStatus, // _Gscan + _Gmoribund_unused
		6: invalidStatus, // _Gscan + _Gdead
		7: debug.Queued,  // _Gscanenqueue
	}
	gStatusString = [...]string{
		0: "idle",
		1: "runnable",
		2: "running",
		3: "syscall",
		4: "waiting",
		8: "copystack",
	}
	gScanStatusString = [...]string{
		1: "scanrunnable",
		2: "scanrunning",
		3: "scansyscall",
		4: "scanwaiting",
		7: "scanenqueue",
	}
)

func (s *Server) handleGoroutines(req *protocol.GoroutinesRequest, resp *protocol.GoroutinesResponse) error {
	// Get DWARF type information for runtime.g.
	ge, err := s.dwarfData.LookupEntry("runtime.g")
	if err != nil {
		return err
	}
	t, err := s.dwarfData.Type(ge.Offset)
	if err != nil {
		return err
	}
	gType, ok := followTypedefs(t).(*dwarf.StructType)
	if !ok {
		return errors.New("runtime.g is not a struct")
	}

	var (
		allgPtr, allgLen uint64
		allgPtrOk        bool
	)
	for {
		// Try to read the slice runtime.allgs.
		allgsEntry, err := s.dwarfData.LookupVariable("runtime.allgs")
		if err != nil {
			break
		}
		allgsAddr, err := s.dwarfData.EntryLocation(allgsEntry)
		if err != nil {
			break
		}
		off, err := s.dwarfData.EntryTypeOffset(allgsEntry)
		if err != nil {
			break
		}
		t, err := s.dwarfData.Type(off)
		if err != nil {
			break
		}
		allgsType, ok := followTypedefs(t).(*dwarf.SliceType)
		if !ok {
			break
		}
		allgs, err := s.peekSlice(allgsType, allgsAddr)
		if err != nil {
			break
		}

		allgPtr, allgLen, allgPtrOk = allgs.Address, allgs.Length, true
		break
	}
	if !allgPtrOk {
		// Read runtime.allg.
		allgEntry, err := s.dwarfData.LookupVariable("runtime.allg")
		if err != nil {
			return err
		}
		allgAddr, err := s.dwarfData.EntryLocation(allgEntry)
		if err != nil {
			return err
		}
		allgPtr, err = s.peekPtr(allgAddr)
		if err != nil {
			return fmt.Errorf("reading allg: %v", err)
		}

		// Read runtime.allglen.
		allglenEntry, err := s.dwarfData.LookupVariable("runtime.allglen")
		if err != nil {
			return err
		}
		off, err := s.dwarfData.EntryTypeOffset(allglenEntry)
		if err != nil {
			return err
		}
		allglenType, err := s.dwarfData.Type(off)
		if err != nil {
			return err
		}
		allglenAddr, err := s.dwarfData.EntryLocation(allglenEntry)
		if err != nil {
			return err
		}
		switch followTypedefs(allglenType).(type) {
		case *dwarf.UintType, *dwarf.IntType:
			allgLen, err = s.peekUint(allglenAddr, allglenType.Common().ByteSize)
			if err != nil {
				return fmt.Errorf("reading allglen: %v", err)
			}
		default:
			// Some runtimes don't specify the type for allglen.  Assume it's uint32.
			allgLen, err = s.peekUint(allglenAddr, 4)
			if err != nil {
				return fmt.Errorf("reading allglen: %v", err)
			}
			if allgLen != 0 {
				break
			}
			// Zero?  Let's try uint64.
			allgLen, err = s.peekUint(allglenAddr, 8)
			if err != nil {
				return fmt.Errorf("reading allglen: %v", err)
			}
		}
	}

	// Initialize s.goroutineStack.
	s.goroutineStackOnce.Do(func() { s.goroutineStackInit(gType) })

	for i := uint64(0); i < allgLen; i++ {
		// allg is an array of pointers to g structs.  Read allg[i].
		g, err := s.peekPtr(allgPtr + i*uint64(s.arch.PointerSize))
		if err != nil {
			return err
		}
		gr := debug.Goroutine{}

		// Read status from the field named "atomicstatus" or "status".
		status, err := s.peekUintStructField(gType, g, "atomicstatus")
		if err != nil {
			status, err = s.peekUintOrIntStructField(gType, g, "status")
		}
		if err != nil {
			return err
		}
		if status == 6 {
			// _Gdead.
			continue
		}
		gr.Status = invalidStatus
		if status < uint64(len(gStatus)) {
			gr.Status = gStatus[status]
			gr.StatusString = gStatusString[status]
		} else if status^0x1000 < uint64(len(gScanStatus)) {
			gr.Status = gScanStatus[status^0x1000]
			gr.StatusString = gScanStatusString[status^0x1000]
		}
		if gr.Status == invalidStatus {
			return fmt.Errorf("unexpected goroutine status 0x%x", status)
		}
		if status == 4 || status == 0x1004 {
			// _Gwaiting or _Gscanwaiting.
			// Try reading waitreason to get a better value for StatusString.
			// Depending on the runtime, waitreason may be a Go string or a C string.
			if waitreason, err := s.peekStringStructField(gType, g, "waitreason", 80); err == nil {
				if waitreason != "" {
					gr.StatusString = waitreason
				}
			} else if ptr, err := s.peekPtrStructField(gType, g, "waitreason"); err == nil {
				waitreason := s.peekCString(ptr, 80)
				if waitreason != "" {
					gr.StatusString = waitreason
				}
			}
		}

		gr.ID, err = s.peekIntStructField(gType, g, "goid")
		if err != nil {
			return err
		}

		// Best-effort attempt to get the names of the goroutine function and the
		// function that created the goroutine.  They aren't always available.
		functionName := func(pc uint64) string {
			entry, _, err := s.dwarfData.PCToFunction(pc)
			if err != nil {
				return ""
			}
			name, _ := entry.Val(dwarf.AttrName).(string)
			return name
		}
		if startpc, err := s.peekUintStructField(gType, g, "startpc"); err == nil {
			gr.Function = functionName(startpc)
		}
		if gopc, err := s.peekUintStructField(gType, g, "gopc"); err == nil {
			gr.Caller = functionName(gopc)
		}
		if gr.Status != debug.Running {
			// TODO: running goroutines too.
			gr.StackFrames, _ = s.goroutineStack(g)
		}

		resp.Goroutines = append(resp.Goroutines, &gr)
	}

	return nil
}

// TODO: let users specify how many frames they want.  10 will be enough to
// determine the reason a goroutine is blocked.
const goroutineStackFrameCount = 10

// goroutineStackInit initializes s.goroutineStack.
func (s *Server) goroutineStackInit(gType *dwarf.StructType) {
	// If we fail to read the DWARF data needed for s.goroutineStack, calling it
	// will always return the error that occurred during initialization.
	var err error // err is captured by the func below.
	s.goroutineStack = func(gAddr uint64) ([]debug.Frame, error) {
		return nil, err
	}

	// Get g field "sched", which contains fields pc and sp.
	schedField, err := getField(gType, "sched")
	if err != nil {
		return
	}
	schedOffset := uint64(schedField.ByteOffset)
	schedType, ok := followTypedefs(schedField.Type).(*dwarf.StructType)
	if !ok {
		err = errors.New(`g field "sched" has the wrong type`)
		return
	}

	// Get the size of the pc and sp fields and their offsets inside the g struct,
	// so we can quickly peek those values for each goroutine later.
	var (
		schedPCOffset, schedSPOffset     uint64
		schedPCByteSize, schedSPByteSize int64
	)
	for _, x := range []struct {
		field    string
		offset   *uint64
		bytesize *int64
	}{
		{"pc", &schedPCOffset, &schedPCByteSize},
		{"sp", &schedSPOffset, &schedSPByteSize},
	} {
		var f *dwarf.StructField
		f, err = getField(schedType, x.field)
		if err != nil {
			return
		}
		*x.offset = schedOffset + uint64(f.ByteOffset)
		switch t := followTypedefs(f.Type).(type) {
		case *dwarf.UintType, *dwarf.IntType:
			*x.bytesize = t.Common().ByteSize
		default:
			err = fmt.Errorf("gobuf field %q has the wrong type", x.field)
			return
		}
	}

	s.goroutineStack = func(gAddr uint64) ([]debug.Frame, error) {
		schedPC, err := s.peekUint(gAddr+schedPCOffset, schedPCByteSize)
		if err != nil {
			return nil, err
		}
		schedSP, err := s.peekUint(gAddr+schedSPOffset, schedSPByteSize)
		if err != nil {
			return nil, err
		}
		return s.walkStack(schedPC, schedSP, goroutineStackFrameCount)
	}
}
