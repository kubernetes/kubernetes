// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

func TestAMD64minimalFeatures(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		if !cpu.Initialized {
			t.Fatal("Initialized expected true, got false")
		}
		if !cpu.X86.HasSSE2 {
			t.Fatal("HasSSE2 expected true, got false")
		}
	}
}

func TestAVX2hasAVX(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		if cpu.X86.HasAVX2 && !cpu.X86.HasAVX {
			t.Fatal("HasAVX expected true, got false")
		}
	}
}

func TestAVX512HasAVX2AndAVX(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		if cpu.X86.HasAVX512 && !cpu.X86.HasAVX {
			t.Fatal("HasAVX expected true, got false")
		}
		if cpu.X86.HasAVX512 && !cpu.X86.HasAVX2 {
			t.Fatal("HasAVX2 expected true, got false")
		}
	}
}

func TestARM64minimalFeatures(t *testing.T) {
	if runtime.GOARCH != "arm64" || (runtime.GOOS == "darwin" || runtime.GOOS == "ios") {
		return
	}
	if !cpu.ARM64.HasASIMD {
		t.Fatal("HasASIMD expected true, got false")
	}
	if !cpu.ARM64.HasFP {
		t.Fatal("HasFP expected true, got false")
	}
}

func TestMIPS64Initialized(t *testing.T) {
	if runtime.GOARCH == "mips64" || runtime.GOARCH == "mips64le" {
		if !cpu.Initialized {
			t.Fatal("Initialized expected true, got false")
		}
	}
}

// On ppc64x, the ISA bit for POWER8 should always be set on POWER8 and beyond.
func TestPPC64minimalFeatures(t *testing.T) {
	// Do not run this with gccgo on ppc64, as it doesn't have POWER8 as a minimum
	// requirement.
	if runtime.Compiler == "gccgo" && runtime.GOARCH == "ppc64" {
		t.Skip("gccgo does not require POWER8 on ppc64; skipping")
	}
	if runtime.GOARCH == "ppc64" || runtime.GOARCH == "ppc64le" {
		if !cpu.PPC64.IsPOWER8 {
			t.Fatal("IsPOWER8 expected true, got false")
		}
	}
}
