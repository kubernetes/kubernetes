// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package bpf implements marshaling and unmarshaling of programs for the
Berkeley Packet Filter virtual machine, and provides a Go implementation
of the virtual machine.

BPF's main use is to specify a packet filter for network taps, so that
the kernel doesn't have to expensively copy every packet it sees to
userspace. However, it's been repurposed to other areas where running
user code in-kernel is needed. For example, Linux's seccomp uses BPF
to apply security policies to system calls. For simplicity, this
documentation refers only to packets, but other uses of BPF have their
own data payloads.

BPF programs run in a restricted virtual machine. It has almost no
access to kernel functions, and while conditional branches are
allowed, they can only jump forwards, to guarantee that there are no
infinite loops.

# The virtual machine

The BPF VM is an accumulator machine. Its main register, called
register A, is an implicit source and destination in all arithmetic
and logic operations. The machine also has 16 scratch registers for
temporary storage, and an indirection register (register X) for
indirect memory access. All registers are 32 bits wide.

Each run of a BPF program is given one packet, which is placed in the
VM's read-only "main memory". LoadAbsolute and LoadIndirect
instructions can fetch up to 32 bits at a time into register A for
examination.

The goal of a BPF program is to produce and return a verdict (uint32),
which tells the kernel what to do with the packet. In the context of
packet filtering, the returned value is the number of bytes of the
packet to forward to userspace, or 0 to ignore the packet. Other
contexts like seccomp define their own return values.

In order to simplify programs, attempts to read past the end of the
packet terminate the program execution with a verdict of 0 (ignore
packet). This means that the vast majority of BPF programs don't need
to do any explicit bounds checking.

In addition to the bytes of the packet, some BPF programs have access
to extensions, which are essentially calls to kernel utility
functions. Currently, the only extensions supported by this package
are the Linux packet filter extensions.

# Examples

This packet filter selects all ARP packets.

	bpf.Assemble([]bpf.Instruction{
		// Load "EtherType" field from the ethernet header.
		bpf.LoadAbsolute{Off: 12, Size: 2},
		// Skip over the next instruction if EtherType is not ARP.
		bpf.JumpIf{Cond: bpf.JumpNotEqual, Val: 0x0806, SkipTrue: 1},
		// Verdict is "send up to 4k of the packet to userspace."
		bpf.RetConstant{Val: 4096},
		// Verdict is "ignore packet."
		bpf.RetConstant{Val: 0},
	})

This packet filter captures a random 1% sample of traffic.

	bpf.Assemble([]bpf.Instruction{
		// Get a 32-bit random number from the Linux kernel.
		bpf.LoadExtension{Num: bpf.ExtRand},
		// 1% dice roll?
		bpf.JumpIf{Cond: bpf.JumpLessThan, Val: 2^32/100, SkipFalse: 1},
		// Capture.
		bpf.RetConstant{Val: 4096},
		// Ignore.
		bpf.RetConstant{Val: 0},
	})
*/
package bpf // import "golang.org/x/net/bpf"
