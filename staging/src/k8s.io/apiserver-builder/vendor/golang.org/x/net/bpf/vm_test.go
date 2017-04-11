// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf_test

import (
	"fmt"
	"testing"

	"golang.org/x/net/bpf"
)

var _ bpf.Instruction = unknown{}

type unknown struct{}

func (unknown) Assemble() (bpf.RawInstruction, error) {
	return bpf.RawInstruction{}, nil
}

func TestVMUnknownInstruction(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadConstant{
			Dst: bpf.RegA,
			Val: 100,
		},
		// Should terminate the program with an error immediately
		unknown{},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer done()

	_, err = vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0x00, 0x00,
	})
	if errStr(err) != "unknown Instruction at index 1: bpf_test.unknown" {
		t.Fatalf("unexpected error while running program: %v", err)
	}
}

func TestVMNoReturnInstruction(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.LoadConstant{
			Dst: bpf.RegA,
			Val: 1,
		},
	})
	if errStr(err) != "BPF program must end with RetA or RetConstant" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMNoInputInstructions(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{})
	if errStr(err) != "one or more Instructions must be specified" {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ExampleNewVM demonstrates usage of a VM, using an Ethernet frame
// as input and checking its EtherType to determine if it should be accepted.
func ExampleNewVM() {
	// Offset | Length | Comment
	// -------------------------
	//   00   |   06   | Ethernet destination MAC address
	//   06   |   06   | Ethernet source MAC address
	//   12   |   02   | Ethernet EtherType
	const (
		etOff = 12
		etLen = 2

		etARP = 0x0806
	)

	// Set up a VM to filter traffic based on if its EtherType
	// matches the ARP EtherType.
	vm, err := bpf.NewVM([]bpf.Instruction{
		// Load EtherType value from Ethernet header
		bpf.LoadAbsolute{
			Off:  etOff,
			Size: etLen,
		},
		// If EtherType is equal to the ARP EtherType, jump to allow
		// packet to be accepted
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      etARP,
			SkipTrue: 1,
		},
		// EtherType does not match the ARP EtherType
		bpf.RetConstant{
			Val: 0,
		},
		// EtherType matches the ARP EtherType, accept up to 1500
		// bytes of packet
		bpf.RetConstant{
			Val: 1500,
		},
	})
	if err != nil {
		panic(fmt.Sprintf("failed to load BPF program: %v", err))
	}

	// Create an Ethernet frame with the ARP EtherType for testing
	frame := []byte{
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
		0x08, 0x06,
		// Payload omitted for brevity
	}

	// Run our VM's BPF program using the Ethernet frame as input
	out, err := vm.Run(frame)
	if err != nil {
		panic(fmt.Sprintf("failed to accept Ethernet frame: %v", err))
	}

	// BPF VM can return a byte count greater than the number of input
	// bytes, so trim the output to match the input byte length
	if out > len(frame) {
		out = len(frame)
	}

	fmt.Printf("out: %d bytes", out)

	// Output:
	// out: 14 bytes
}

// errStr returns the string representation of an error, or
// "<nil>" if it is nil.
func errStr(err error) string {
	if err == nil {
		return "<nil>"
	}

	return err.Error()
}
