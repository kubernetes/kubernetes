// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	"runtime"
	"testing"
	"unsafe"

	"golang.org/x/sys/cpu"
)

var s390xTests = []struct {
	name      string
	feature   bool
	facility  uint
	mandatory bool
}{
	{"ZARCH", cpu.S390X.HasZARCH, 1, true},
	{"STFLE", cpu.S390X.HasSTFLE, 7, true},
	{"LDISP", cpu.S390X.HasLDISP, 18, true},
	{"EIMM", cpu.S390X.HasEIMM, 21, true},
	{"DFP", cpu.S390X.HasDFP, 42, false},
	{"MSA", cpu.S390X.HasMSA, 17, false},
	{"VX", cpu.S390X.HasVX, 129, false},
	{"VXE", cpu.S390X.HasVXE, 135, false},
}

// bitIsSet reports whether the bit at index is set. The bit index
// is in big endian order, so bit index 0 is the leftmost bit.
func bitIsSet(bits [4]uint64, i uint) bool {
	return bits[i/64]&((1<<63)>>(i%64)) != 0
}

// facilityList contains the contents of location 200 on zos.
// Bits are numbered in big endian order so the
// leftmost bit (the MSB) is at index 0.
type facilityList struct {
	bits [4]uint64
}

func TestS390XVectorFacilityFeatures(t *testing.T) {
	// vector-enhancements require vector facility to be enabled
	if cpu.S390X.HasVXE && !cpu.S390X.HasVX {
		t.Error("HasVX expected true, got false (VXE is true)")
	}
}

func TestS390XMandatoryFeatures(t *testing.T) {
	for _, tc := range s390xTests {
		if tc.mandatory && !tc.feature {
			t.Errorf("Feature %s is mandatory but is not present", tc.name)
		}
	}
}

func TestS390XFeatures(t *testing.T) {
	if runtime.GOOS != "zos" {
		return
	}
	// Read available facilities from address 200.
	facilitiesAddress := uintptr(200)
	var facilities facilityList
	for i := 0; i < 4; i++ {
		facilities.bits[i] = *(*uint64)(unsafe.Pointer(facilitiesAddress + uintptr(8*i)))
	}

	for _, tc := range s390xTests {
		if want := bitIsSet(facilities.bits, tc.facility); want != tc.feature {
			t.Errorf("Feature %s expected %v, got %v", tc.name, want, tc.feature)
		}
	}
}
