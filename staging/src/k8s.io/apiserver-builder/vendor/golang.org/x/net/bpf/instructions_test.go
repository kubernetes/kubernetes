// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

import (
	"io/ioutil"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// This is a direct translation of the program in
// testdata/all_instructions.txt.
var allInstructions = []Instruction{
	LoadConstant{Dst: RegA, Val: 42},
	LoadConstant{Dst: RegX, Val: 42},

	LoadScratch{Dst: RegA, N: 3},
	LoadScratch{Dst: RegX, N: 3},

	LoadAbsolute{Off: 42, Size: 1},
	LoadAbsolute{Off: 42, Size: 2},
	LoadAbsolute{Off: 42, Size: 4},

	LoadIndirect{Off: 42, Size: 1},
	LoadIndirect{Off: 42, Size: 2},
	LoadIndirect{Off: 42, Size: 4},

	LoadMemShift{Off: 42},

	LoadExtension{Num: ExtLen},
	LoadExtension{Num: ExtProto},
	LoadExtension{Num: ExtType},
	LoadExtension{Num: ExtRand},

	StoreScratch{Src: RegA, N: 3},
	StoreScratch{Src: RegX, N: 3},

	ALUOpConstant{Op: ALUOpAdd, Val: 42},
	ALUOpConstant{Op: ALUOpSub, Val: 42},
	ALUOpConstant{Op: ALUOpMul, Val: 42},
	ALUOpConstant{Op: ALUOpDiv, Val: 42},
	ALUOpConstant{Op: ALUOpOr, Val: 42},
	ALUOpConstant{Op: ALUOpAnd, Val: 42},
	ALUOpConstant{Op: ALUOpShiftLeft, Val: 42},
	ALUOpConstant{Op: ALUOpShiftRight, Val: 42},
	ALUOpConstant{Op: ALUOpMod, Val: 42},
	ALUOpConstant{Op: ALUOpXor, Val: 42},

	ALUOpX{Op: ALUOpAdd},
	ALUOpX{Op: ALUOpSub},
	ALUOpX{Op: ALUOpMul},
	ALUOpX{Op: ALUOpDiv},
	ALUOpX{Op: ALUOpOr},
	ALUOpX{Op: ALUOpAnd},
	ALUOpX{Op: ALUOpShiftLeft},
	ALUOpX{Op: ALUOpShiftRight},
	ALUOpX{Op: ALUOpMod},
	ALUOpX{Op: ALUOpXor},

	NegateA{},

	Jump{Skip: 10},
	JumpIf{Cond: JumpEqual, Val: 42, SkipTrue: 8, SkipFalse: 9},
	JumpIf{Cond: JumpNotEqual, Val: 42, SkipTrue: 8},
	JumpIf{Cond: JumpLessThan, Val: 42, SkipTrue: 7},
	JumpIf{Cond: JumpLessOrEqual, Val: 42, SkipTrue: 6},
	JumpIf{Cond: JumpGreaterThan, Val: 42, SkipTrue: 4, SkipFalse: 5},
	JumpIf{Cond: JumpGreaterOrEqual, Val: 42, SkipTrue: 3, SkipFalse: 4},
	JumpIf{Cond: JumpBitsSet, Val: 42, SkipTrue: 2, SkipFalse: 3},

	TAX{},
	TXA{},

	RetA{},
	RetConstant{Val: 42},
}
var allInstructionsExpected = "testdata/all_instructions.bpf"

// Check that we produce the same output as the canonical bpf_asm
// linux kernel tool.
func TestInterop(t *testing.T) {
	out, err := Assemble(allInstructions)
	if err != nil {
		t.Fatalf("assembly of allInstructions program failed: %s", err)
	}
	t.Logf("Assembled program is %d instructions long", len(out))

	bs, err := ioutil.ReadFile(allInstructionsExpected)
	if err != nil {
		t.Fatalf("reading %s: %s", allInstructionsExpected, err)
	}
	// First statement is the number of statements, last statement is
	// empty. We just ignore both and rely on slice length.
	stmts := strings.Split(string(bs), ",")
	if len(stmts)-2 != len(out) {
		t.Fatalf("test program lengths don't match: %s has %d, Go implementation has %d", allInstructionsExpected, len(stmts)-2, len(allInstructions))
	}

	for i, stmt := range stmts[1 : len(stmts)-2] {
		nums := strings.Split(stmt, " ")
		if len(nums) != 4 {
			t.Fatalf("malformed instruction %d in %s: %s", i+1, allInstructionsExpected, stmt)
		}

		actual := out[i]

		op, err := strconv.ParseUint(nums[0], 10, 16)
		if err != nil {
			t.Fatalf("malformed opcode %s in instruction %d of %s", nums[0], i+1, allInstructionsExpected)
		}
		if actual.Op != uint16(op) {
			t.Errorf("opcode mismatch on instruction %d (%#v): got 0x%02x, want 0x%02x", i+1, allInstructions[i], actual.Op, op)
		}

		jt, err := strconv.ParseUint(nums[1], 10, 8)
		if err != nil {
			t.Fatalf("malformed jt offset %s in instruction %d of %s", nums[1], i+1, allInstructionsExpected)
		}
		if actual.Jt != uint8(jt) {
			t.Errorf("jt mismatch on instruction %d (%#v): got %d, want %d", i+1, allInstructions[i], actual.Jt, jt)
		}

		jf, err := strconv.ParseUint(nums[2], 10, 8)
		if err != nil {
			t.Fatalf("malformed jf offset %s in instruction %d of %s", nums[2], i+1, allInstructionsExpected)
		}
		if actual.Jf != uint8(jf) {
			t.Errorf("jf mismatch on instruction %d (%#v): got %d, want %d", i+1, allInstructions[i], actual.Jf, jf)
		}

		k, err := strconv.ParseUint(nums[3], 10, 32)
		if err != nil {
			t.Fatalf("malformed constant %s in instruction %d of %s", nums[3], i+1, allInstructionsExpected)
		}
		if actual.K != uint32(k) {
			t.Errorf("constant mismatch on instruction %d (%#v): got %d, want %d", i+1, allInstructions[i], actual.K, k)
		}
	}
}

// Check that assembly and disassembly match each other.
//
// Because we offer "fake" jump conditions that don't appear in the
// machine code, disassembly won't be a 1:1 match with the original
// source, although the behavior will be identical. However,
// reassembling the disassembly should produce an identical program.
func TestAsmDisasm(t *testing.T) {
	prog1, err := Assemble(allInstructions)
	if err != nil {
		t.Fatalf("assembly of allInstructions program failed: %s", err)
	}
	t.Logf("Assembled program is %d instructions long", len(prog1))

	src, allDecoded := Disassemble(prog1)
	if !allDecoded {
		t.Errorf("Disassemble(Assemble(allInstructions)) produced unrecognized instructions:")
		for i, inst := range src {
			if r, ok := inst.(RawInstruction); ok {
				t.Logf("  insn %d, %#v --> %#v", i+1, allInstructions[i], r)
			}
		}
	}

	prog2, err := Assemble(src)
	if err != nil {
		t.Fatalf("assembly of Disassemble(Assemble(allInstructions)) failed: %s", err)
	}

	if len(prog2) != len(prog1) {
		t.Fatalf("disassembly changed program size: %d insns before, %d insns after", len(prog1), len(prog2))
	}
	if !reflect.DeepEqual(prog1, prog2) {
		t.Errorf("program mutated by disassembly:")
		for i := range prog2 {
			if !reflect.DeepEqual(prog1[i], prog2[i]) {
				t.Logf("  insn %d, s: %#v, p1: %#v, p2: %#v", i+1, allInstructions[i], prog1[i], prog2[i])
			}
		}
	}
}
