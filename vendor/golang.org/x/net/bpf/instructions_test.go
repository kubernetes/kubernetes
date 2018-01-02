// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

import (
	"fmt"
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
func TestAsmDisasm(t *testing.T) {
	prog1, err := Assemble(allInstructions)
	if err != nil {
		t.Fatalf("assembly of allInstructions program failed: %s", err)
	}
	t.Logf("Assembled program is %d instructions long", len(prog1))

	got, allDecoded := Disassemble(prog1)
	if !allDecoded {
		t.Errorf("Disassemble(Assemble(allInstructions)) produced unrecognized instructions:")
		for i, inst := range got {
			if r, ok := inst.(RawInstruction); ok {
				t.Logf("  insn %d, %#v --> %#v", i+1, allInstructions[i], r)
			}
		}
	}

	if len(allInstructions) != len(got) {
		t.Fatalf("disassembly changed program size: %d insns before, %d insns after", len(allInstructions), len(got))
	}
	if !reflect.DeepEqual(allInstructions, got) {
		t.Errorf("program mutated by disassembly:")
		for i := range got {
			if !reflect.DeepEqual(allInstructions[i], got[i]) {
				t.Logf("  insn %d, s: %#v, p1: %#v, got: %#v", i+1, allInstructions[i], prog1[i], got[i])
			}
		}
	}
}

type InvalidInstruction struct{}

func (a InvalidInstruction) Assemble() (RawInstruction, error) {
	return RawInstruction{}, fmt.Errorf("Invalid Instruction")
}

func (a InvalidInstruction) String() string {
	return fmt.Sprintf("unknown instruction: %#v", a)
}

func TestString(t *testing.T) {
	testCases := []struct {
		instruction Instruction
		assembler   string
	}{
		{
			instruction: LoadConstant{Dst: RegA, Val: 42},
			assembler:   "ld #42",
		},
		{
			instruction: LoadConstant{Dst: RegX, Val: 42},
			assembler:   "ldx #42",
		},
		{
			instruction: LoadConstant{Dst: 0xffff, Val: 42},
			assembler:   "unknown instruction: bpf.LoadConstant{Dst:0xffff, Val:0x2a}",
		},
		{
			instruction: LoadScratch{Dst: RegA, N: 3},
			assembler:   "ld M[3]",
		},
		{
			instruction: LoadScratch{Dst: RegX, N: 3},
			assembler:   "ldx M[3]",
		},
		{
			instruction: LoadScratch{Dst: 0xffff, N: 3},
			assembler:   "unknown instruction: bpf.LoadScratch{Dst:0xffff, N:3}",
		},
		{
			instruction: LoadAbsolute{Off: 42, Size: 1},
			assembler:   "ldb [42]",
		},
		{
			instruction: LoadAbsolute{Off: 42, Size: 2},
			assembler:   "ldh [42]",
		},
		{
			instruction: LoadAbsolute{Off: 42, Size: 4},
			assembler:   "ld [42]",
		},
		{
			instruction: LoadAbsolute{Off: 42, Size: -1},
			assembler:   "unknown instruction: bpf.LoadAbsolute{Off:0x2a, Size:-1}",
		},
		{
			instruction: LoadIndirect{Off: 42, Size: 1},
			assembler:   "ldb [x + 42]",
		},
		{
			instruction: LoadIndirect{Off: 42, Size: 2},
			assembler:   "ldh [x + 42]",
		},
		{
			instruction: LoadIndirect{Off: 42, Size: 4},
			assembler:   "ld [x + 42]",
		},
		{
			instruction: LoadIndirect{Off: 42, Size: -1},
			assembler:   "unknown instruction: bpf.LoadIndirect{Off:0x2a, Size:-1}",
		},
		{
			instruction: LoadMemShift{Off: 42},
			assembler:   "ldx 4*([42]&0xf)",
		},
		{
			instruction: LoadExtension{Num: ExtLen},
			assembler:   "ld #len",
		},
		{
			instruction: LoadExtension{Num: ExtProto},
			assembler:   "ld #proto",
		},
		{
			instruction: LoadExtension{Num: ExtType},
			assembler:   "ld #type",
		},
		{
			instruction: LoadExtension{Num: ExtPayloadOffset},
			assembler:   "ld #poff",
		},
		{
			instruction: LoadExtension{Num: ExtInterfaceIndex},
			assembler:   "ld #ifidx",
		},
		{
			instruction: LoadExtension{Num: ExtNetlinkAttr},
			assembler:   "ld #nla",
		},
		{
			instruction: LoadExtension{Num: ExtNetlinkAttrNested},
			assembler:   "ld #nlan",
		},
		{
			instruction: LoadExtension{Num: ExtMark},
			assembler:   "ld #mark",
		},
		{
			instruction: LoadExtension{Num: ExtQueue},
			assembler:   "ld #queue",
		},
		{
			instruction: LoadExtension{Num: ExtLinkLayerType},
			assembler:   "ld #hatype",
		},
		{
			instruction: LoadExtension{Num: ExtRXHash},
			assembler:   "ld #rxhash",
		},
		{
			instruction: LoadExtension{Num: ExtCPUID},
			assembler:   "ld #cpu",
		},
		{
			instruction: LoadExtension{Num: ExtVLANTag},
			assembler:   "ld #vlan_tci",
		},
		{
			instruction: LoadExtension{Num: ExtVLANTagPresent},
			assembler:   "ld #vlan_avail",
		},
		{
			instruction: LoadExtension{Num: ExtVLANProto},
			assembler:   "ld #vlan_tpid",
		},
		{
			instruction: LoadExtension{Num: ExtRand},
			assembler:   "ld #rand",
		},
		{
			instruction: LoadAbsolute{Off: 0xfffff038, Size: 4},
			assembler:   "ld #rand",
		},
		{
			instruction: LoadExtension{Num: 0xfff},
			assembler:   "unknown instruction: bpf.LoadExtension{Num:4095}",
		},
		{
			instruction: StoreScratch{Src: RegA, N: 3},
			assembler:   "st M[3]",
		},
		{
			instruction: StoreScratch{Src: RegX, N: 3},
			assembler:   "stx M[3]",
		},
		{
			instruction: StoreScratch{Src: 0xffff, N: 3},
			assembler:   "unknown instruction: bpf.StoreScratch{Src:0xffff, N:3}",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpAdd, Val: 42},
			assembler:   "add #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpSub, Val: 42},
			assembler:   "sub #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpMul, Val: 42},
			assembler:   "mul #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpDiv, Val: 42},
			assembler:   "div #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpOr, Val: 42},
			assembler:   "or #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpAnd, Val: 42},
			assembler:   "and #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpShiftLeft, Val: 42},
			assembler:   "lsh #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpShiftRight, Val: 42},
			assembler:   "rsh #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpMod, Val: 42},
			assembler:   "mod #42",
		},
		{
			instruction: ALUOpConstant{Op: ALUOpXor, Val: 42},
			assembler:   "xor #42",
		},
		{
			instruction: ALUOpConstant{Op: 0xffff, Val: 42},
			assembler:   "unknown instruction: bpf.ALUOpConstant{Op:0xffff, Val:0x2a}",
		},
		{
			instruction: ALUOpX{Op: ALUOpAdd},
			assembler:   "add x",
		},
		{
			instruction: ALUOpX{Op: ALUOpSub},
			assembler:   "sub x",
		},
		{
			instruction: ALUOpX{Op: ALUOpMul},
			assembler:   "mul x",
		},
		{
			instruction: ALUOpX{Op: ALUOpDiv},
			assembler:   "div x",
		},
		{
			instruction: ALUOpX{Op: ALUOpOr},
			assembler:   "or x",
		},
		{
			instruction: ALUOpX{Op: ALUOpAnd},
			assembler:   "and x",
		},
		{
			instruction: ALUOpX{Op: ALUOpShiftLeft},
			assembler:   "lsh x",
		},
		{
			instruction: ALUOpX{Op: ALUOpShiftRight},
			assembler:   "rsh x",
		},
		{
			instruction: ALUOpX{Op: ALUOpMod},
			assembler:   "mod x",
		},
		{
			instruction: ALUOpX{Op: ALUOpXor},
			assembler:   "xor x",
		},
		{
			instruction: ALUOpX{Op: 0xffff},
			assembler:   "unknown instruction: bpf.ALUOpX{Op:0xffff}",
		},
		{
			instruction: NegateA{},
			assembler:   "neg",
		},
		{
			instruction: Jump{Skip: 10},
			assembler:   "ja 10",
		},
		{
			instruction: JumpIf{Cond: JumpEqual, Val: 42, SkipTrue: 8, SkipFalse: 9},
			assembler:   "jeq #42,8,9",
		},
		{
			instruction: JumpIf{Cond: JumpEqual, Val: 42, SkipTrue: 8},
			assembler:   "jeq #42,8",
		},
		{
			instruction: JumpIf{Cond: JumpEqual, Val: 42, SkipFalse: 8},
			assembler:   "jneq #42,8",
		},
		{
			instruction: JumpIf{Cond: JumpNotEqual, Val: 42, SkipTrue: 8},
			assembler:   "jneq #42,8",
		},
		{
			instruction: JumpIf{Cond: JumpLessThan, Val: 42, SkipTrue: 7},
			assembler:   "jlt #42,7",
		},
		{
			instruction: JumpIf{Cond: JumpLessOrEqual, Val: 42, SkipTrue: 6},
			assembler:   "jle #42,6",
		},
		{
			instruction: JumpIf{Cond: JumpGreaterThan, Val: 42, SkipTrue: 4, SkipFalse: 5},
			assembler:   "jgt #42,4,5",
		},
		{
			instruction: JumpIf{Cond: JumpGreaterThan, Val: 42, SkipTrue: 4},
			assembler:   "jgt #42,4",
		},
		{
			instruction: JumpIf{Cond: JumpGreaterOrEqual, Val: 42, SkipTrue: 3, SkipFalse: 4},
			assembler:   "jge #42,3,4",
		},
		{
			instruction: JumpIf{Cond: JumpGreaterOrEqual, Val: 42, SkipTrue: 3},
			assembler:   "jge #42,3",
		},
		{
			instruction: JumpIf{Cond: JumpBitsSet, Val: 42, SkipTrue: 2, SkipFalse: 3},
			assembler:   "jset #42,2,3",
		},
		{
			instruction: JumpIf{Cond: JumpBitsSet, Val: 42, SkipTrue: 2},
			assembler:   "jset #42,2",
		},
		{
			instruction: JumpIf{Cond: JumpBitsNotSet, Val: 42, SkipTrue: 2, SkipFalse: 3},
			assembler:   "jset #42,3,2",
		},
		{
			instruction: JumpIf{Cond: JumpBitsNotSet, Val: 42, SkipTrue: 2},
			assembler:   "jset #42,0,2",
		},
		{
			instruction: JumpIf{Cond: 0xffff, Val: 42, SkipTrue: 1, SkipFalse: 2},
			assembler:   "unknown instruction: bpf.JumpIf{Cond:0xffff, Val:0x2a, SkipTrue:0x1, SkipFalse:0x2}",
		},
		{
			instruction: TAX{},
			assembler:   "tax",
		},
		{
			instruction: TXA{},
			assembler:   "txa",
		},
		{
			instruction: RetA{},
			assembler:   "ret a",
		},
		{
			instruction: RetConstant{Val: 42},
			assembler:   "ret #42",
		},
		// Invalid instruction
		{
			instruction: InvalidInstruction{},
			assembler:   "unknown instruction: bpf.InvalidInstruction{}",
		},
	}

	for _, testCase := range testCases {
		if input, ok := testCase.instruction.(fmt.Stringer); ok {
			got := input.String()
			if got != testCase.assembler {
				t.Errorf("String did not return expected assembler notation, expected: %s, got: %s", testCase.assembler, got)
			}
		} else {
			t.Errorf("Instruction %#v is not a fmt.Stringer", testCase.instruction)
		}
	}
}
