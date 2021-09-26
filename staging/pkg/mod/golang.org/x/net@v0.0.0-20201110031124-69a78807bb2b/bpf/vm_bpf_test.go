// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf_test

import (
	"net"
	"runtime"
	"testing"
	"time"

	"golang.org/x/net/bpf"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
	"golang.org/x/net/nettest"
)

// A virtualMachine is a BPF virtual machine which can process an
// input packet against a BPF program and render a verdict.
type virtualMachine interface {
	Run(in []byte) (int, error)
}

// canUseOSVM indicates if the OS BPF VM is available on this platform.
func canUseOSVM() bool {
	// OS BPF VM can only be used on platforms where x/net/ipv4 supports
	// attaching a BPF program to a socket.
	switch runtime.GOOS {
	case "linux":
		return true
	}

	return false
}

// All BPF tests against both the Go VM and OS VM are assumed to
// be used with a UDP socket. As a result, the entire contents
// of a UDP datagram is sent through the BPF program, but only
// the body after the UDP header will ever be returned in output.

// testVM sets up a Go BPF VM, and if available, a native OS BPF VM
// for integration testing.
func testVM(t *testing.T, filter []bpf.Instruction) (virtualMachine, func(), error) {
	goVM, err := bpf.NewVM(filter)
	if err != nil {
		// Some tests expect an error, so this error must be returned
		// instead of fatally exiting the test
		return nil, nil, err
	}

	mvm := &multiVirtualMachine{
		goVM: goVM,

		t: t,
	}

	// If available, add the OS VM for tests which verify that both the Go
	// VM and OS VM have exactly the same output for the same input program
	// and packet.
	done := func() {}
	if canUseOSVM() {
		osVM, osVMDone := testOSVM(t, filter)
		done = func() { osVMDone() }
		mvm.osVM = osVM
	}

	return mvm, done, nil
}

// udpHeaderLen is the length of a UDP header.
const udpHeaderLen = 8

// A multiVirtualMachine is a virtualMachine which can call out to both the Go VM
// and the native OS VM, if the OS VM is available.
type multiVirtualMachine struct {
	goVM virtualMachine
	osVM virtualMachine

	t *testing.T
}

func (mvm *multiVirtualMachine) Run(in []byte) (int, error) {
	if len(in) < udpHeaderLen {
		mvm.t.Fatalf("input must be at least length of UDP header (%d), got: %d",
			udpHeaderLen, len(in))
	}

	// All tests have a UDP header as part of input, because the OS VM
	// packets always will. For the Go VM, this output is trimmed before
	// being sent back to tests.
	goOut, goErr := mvm.goVM.Run(in)
	if goOut >= udpHeaderLen {
		goOut -= udpHeaderLen
	}

	// If Go output is larger than the size of the packet, packet filtering
	// interop tests must trim the output bytes to the length of the packet.
	// The BPF VM should not do this on its own, as other uses of it do
	// not trim the output byte count.
	trim := len(in) - udpHeaderLen
	if goOut > trim {
		goOut = trim
	}

	// When the OS VM is not available, process using the Go VM alone
	if mvm.osVM == nil {
		return goOut, goErr
	}

	// The OS VM will apply its own UDP header, so remove the pseudo header
	// that the Go VM needs.
	osOut, err := mvm.osVM.Run(in[udpHeaderLen:])
	if err != nil {
		mvm.t.Fatalf("error while running OS VM: %v", err)
	}

	// Verify both VMs return same number of bytes
	var mismatch bool
	if goOut != osOut {
		mismatch = true
		mvm.t.Logf("output byte count does not match:\n- go: %v\n- os: %v", goOut, osOut)
	}

	if mismatch {
		mvm.t.Fatal("Go BPF and OS BPF packet outputs do not match")
	}

	return goOut, goErr
}

// An osVirtualMachine is a virtualMachine which uses the OS's BPF VM for
// processing BPF programs.
type osVirtualMachine struct {
	l net.PacketConn
	s net.Conn
}

// testOSVM creates a virtualMachine which uses the OS's BPF VM by injecting
// packets into a UDP listener with a BPF program attached to it.
func testOSVM(t *testing.T, filter []bpf.Instruction) (virtualMachine, func()) {
	l, err := nettest.NewLocalPacketListener("udp")
	if err != nil {
		t.Fatalf("failed to open OS VM UDP listener: %v", err)
	}

	prog, err := bpf.Assemble(filter)
	if err != nil {
		t.Fatalf("failed to compile BPF program: %v", err)
	}

	ip := l.LocalAddr().(*net.UDPAddr).IP
	if ip.To4() != nil && ip.To16() == nil {
		err = ipv4.NewPacketConn(l).SetBPF(prog)
	} else {
		err = ipv6.NewPacketConn(l).SetBPF(prog)
	}
	if err != nil {
		t.Fatalf("failed to attach BPF program to listener: %v", err)
	}

	s, err := net.Dial(l.LocalAddr().Network(), l.LocalAddr().String())
	if err != nil {
		t.Fatalf("failed to dial connection to listener: %v", err)
	}

	done := func() {
		_ = s.Close()
		_ = l.Close()
	}

	return &osVirtualMachine{
		l: l,
		s: s,
	}, done
}

// Run sends the input bytes into the OS's BPF VM and returns its verdict.
func (vm *osVirtualMachine) Run(in []byte) (int, error) {
	go func() {
		_, _ = vm.s.Write(in)
	}()

	vm.l.SetDeadline(time.Now().Add(50 * time.Millisecond))

	var b [512]byte
	n, _, err := vm.l.ReadFrom(b[:])
	if err != nil {
		// A timeout indicates that BPF filtered out the packet, and thus,
		// no input should be returned.
		if nerr, ok := err.(net.Error); ok && nerr.Timeout() {
			return n, nil
		}

		return n, err
	}

	return n, nil
}
