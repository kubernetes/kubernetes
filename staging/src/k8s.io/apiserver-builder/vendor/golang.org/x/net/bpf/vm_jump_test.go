// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf_test

import (
	"testing"

	"golang.org/x/net/bpf"
)

func TestVMJumpOne(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.Jump{
			Skip: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 9,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		1,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpOutOfProgram(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.Jump{
			Skip: 1,
		},
		bpf.RetA{},
	})
	if errStr(err) != "cannot jump 1 instructions; jumping past program bounds" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMJumpIfTrueOutOfProgram(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			SkipTrue: 2,
		},
		bpf.RetA{},
	})
	if errStr(err) != "cannot jump 2 instructions in true case; jumping past program bounds" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMJumpIfFalseOutOfProgram(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.JumpIf{
			Cond:      bpf.JumpEqual,
			SkipFalse: 3,
		},
		bpf.RetA{},
	})
	if errStr(err) != "cannot jump 3 instructions in false case; jumping past program bounds" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMJumpIfEqual(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      1,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 9,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		1,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfNotEqual(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.JumpIf{
			Cond:      bpf.JumpNotEqual,
			Val:       1,
			SkipFalse: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 9,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		1,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfGreaterThan(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 4,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpGreaterThan,
			Val:      0x00010202,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 12,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 4, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfLessThan(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 4,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpLessThan,
			Val:      0xff010203,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 12,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 4, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfGreaterOrEqual(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 4,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpGreaterOrEqual,
			Val:      0x00010203,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 12,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 4, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfLessOrEqual(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 4,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpLessOrEqual,
			Val:      0xff010203,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 12,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 4, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfBitsSet(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 2,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpBitsSet,
			Val:      0x1122,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 10,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0x01, 0x02,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 2, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMJumpIfBitsNotSet(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 2,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpBitsNotSet,
			Val:      0x1221,
			SkipTrue: 1,
		},
		bpf.RetConstant{
			Val: 0,
		},
		bpf.RetConstant{
			Val: 10,
		},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0x01, 0x02,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 2, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}
