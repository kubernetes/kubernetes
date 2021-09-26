// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf_test

import (
	"testing"

	"golang.org/x/net/bpf"
)

func TestVMALUOpAdd(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpAdd,
			Val: 3,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		8, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 3, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpSub(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.TAX{},
		bpf.ALUOpX{
			Op: bpf.ALUOpSub,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		1, 2, 3,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 0, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpMul(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpMul,
			Val: 2,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		6, 2, 3, 4,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 4, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpDiv(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpDiv,
			Val: 2,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		20, 2, 3, 4,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 2, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpDivByZeroALUOpConstant(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpDiv,
			Val: 0,
		},
		bpf.RetA{},
	})
	if errStr(err) != "cannot divide by zero using ALUOpConstant" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMALUOpDivByZeroALUOpX(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		// Load byte 0 into X
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.TAX{},
		// Load byte 1 into A
		bpf.LoadAbsolute{
			Off:  9,
			Size: 1,
		},
		// Attempt to perform 1/0
		bpf.ALUOpX{
			Op: bpf.ALUOpDiv,
		},
		// Return 4 bytes if program does not terminate
		bpf.LoadConstant{
			Val: 12,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 3, 4,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 0, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpOr(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 2,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpOr,
			Val: 0x01,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0x00, 0x10, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x09, 0xff,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 9, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpAnd(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 2,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpAnd,
			Val: 0x0019,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0xaa, 0x09,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpShiftLeft(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpShiftLeft,
			Val: 0x01,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      0x02,
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
		0x01, 0xaa,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpShiftRight(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpShiftRight,
			Val: 0x01,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      0x04,
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
		0x08, 0xff, 0xff,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpMod(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpMod,
			Val: 20,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		30, 0, 0,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 2, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpModByZeroALUOpConstant(t *testing.T) {
	_, _, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpMod,
			Val: 0,
		},
		bpf.RetA{},
	})
	if errStr(err) != "cannot divide by zero using ALUOpConstant" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestVMALUOpModByZeroALUOpX(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		// Load byte 0 into X
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.TAX{},
		// Load byte 1 into A
		bpf.LoadAbsolute{
			Off:  9,
			Size: 1,
		},
		// Attempt to perform 1%0
		bpf.ALUOpX{
			Op: bpf.ALUOpMod,
		},
		// Return 4 bytes if program does not terminate
		bpf.LoadConstant{
			Val: 12,
		},
		bpf.RetA{},
	})
	if err != nil {
		t.Fatalf("failed to load BPF program: %v", err)
	}
	defer done()

	out, err := vm.Run([]byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0, 1, 3, 4,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 0, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpXor(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpXor,
			Val: 0x0a,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      0x01,
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
		0x0b, 0x00, 0x00, 0x00,
	})
	if err != nil {
		t.Fatalf("unexpected error while running program: %v", err)
	}
	if want, got := 1, out; want != got {
		t.Fatalf("unexpected number of output bytes:\n- want: %d\n-  got: %d",
			want, got)
	}
}

func TestVMALUOpUnknown(t *testing.T) {
	vm, done, err := testVM(t, []bpf.Instruction{
		bpf.LoadAbsolute{
			Off:  8,
			Size: 1,
		},
		bpf.ALUOpConstant{
			Op:  bpf.ALUOpAdd,
			Val: 1,
		},
		// Verify that an unknown operation is a no-op
		bpf.ALUOpConstant{
			Op: 100,
		},
		bpf.JumpIf{
			Cond:     bpf.JumpEqual,
			Val:      0x02,
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
