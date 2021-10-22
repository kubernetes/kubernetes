// Copyright 2018 Google LLC
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

package elf

import (
	"fmt"
	"testing"
)

type nameTest struct {
	val interface{}
	str string
}

var nameTests = []nameTest{
	{ELFOSABI_LINUX, "ELFOSABI_LINUX"},
	{ET_EXEC, "ET_EXEC"},
	{EM_860, "EM_860"},
	{SHN_LOPROC, "SHN_LOPROC"},
	{SHT_PROGBITS, "SHT_PROGBITS"},
	{SHF_MERGE + SHF_TLS, "SHF_MERGE+SHF_TLS"},
	{PT_LOAD, "PT_LOAD"},
	{PF_W + PF_R + 0x50, "PF_W+PF_R+0x50"},
	{DT_SYMBOLIC, "DT_SYMBOLIC"},
	{DF_BIND_NOW, "DF_BIND_NOW"},
	{NT_FPREGSET, "NT_FPREGSET"},
	{STB_GLOBAL, "STB_GLOBAL"},
	{STT_COMMON, "STT_COMMON"},
	{STV_HIDDEN, "STV_HIDDEN"},
	{R_X86_64_PC32, "R_X86_64_PC32"},
	{R_ALPHA_OP_PUSH, "R_ALPHA_OP_PUSH"},
	{R_ARM_THM_ABS5, "R_ARM_THM_ABS5"},
	{R_386_GOT32, "R_386_GOT32"},
	{R_PPC_GOT16_HI, "R_PPC_GOT16_HI"},
	{R_SPARC_GOT22, "R_SPARC_GOT22"},
	{ET_LOOS + 5, "ET_LOOS+5"},
	{ProgFlag(0x50), "0x50"},
}

func TestNames(t *testing.T) {
	for i, tt := range nameTests {
		s := fmt.Sprint(tt.val)
		if s != tt.str {
			t.Errorf("#%d: Sprint(%d) = %q, want %q", i, tt.val, s, tt.str)
		}
	}
}
