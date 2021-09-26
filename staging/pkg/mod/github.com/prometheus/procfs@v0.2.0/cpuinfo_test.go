// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux

package procfs

import "testing"

const (
	cpuinfoArm7Legacy = `
Processor : ARMv7 Processor rev 5 (v7l)
processor : 0
BogoMIPS : 2400.00

processor : 1
BogoMIPS : 2400.00

Features : swp half thumb fastmult vfp edsp thumbee neon vfpv3 tls vfpv4 idiva idivt
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part : 0xc07
CPU revision : 5

Hardware : sun8i
Revision : 0000
Serial : 5400503583203c3c040e`

	cpuinfoArm7LegacyV1 = `
Processor       : ARMv6-compatible processor rev 5 (v6l)
BogoMIPS        : 791.34
Features        : swp half thumb fastmult vfp edsp java 
CPU implementer : 0x41
CPU architecture: 6TEJ
CPU variant     : 0x1
CPU part        : 0xb36
CPU revision    : 5

Hardware        : IMAPX200
Revision        : 0000
Serial          : 0000000000000000`

	cpuinfoArm7 = `
processor : 0
model name : ARMv7 Processor rev 3 (v7l)
BogoMIPS : 108.00
Features : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part : 0xd08
CPU revision : 3

processor : 1
model name : ARMv7 Processor rev 3 (v7l)
BogoMIPS : 108.00
Features : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part : 0xd08
CPU revision : 3

processor : 2
model name : ARMv7 Processor rev 3 (v7l)
BogoMIPS : 108.00
Features : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part : 0xd08
CPU revision : 3

processor : 3
model name : ARMv7 Processor rev 3 (v7l)
BogoMIPS : 108.00
Features : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part : 0xd08
CPU revision : 3

Hardware : BCM2835
Revision : c03111
`

	cpuinfoS390x = `
vendor_id       : IBM/S390
# processors    : 4
bogomips per cpu: 3033.00
max thread id   : 0
features	: esan3 zarch stfle msa ldisp eimm dfp edat etf3eh highgprs te vx sie
facilities      : 0 1 2 3 4 6 7 8 9 10 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30 31 32 33 34 35 36 37 40 41 42 43 44 45 46 47 48 49 50 51 52 53 55 57 73 74 75 76 77 80 81 82 128 129 131
cache0          : level=1 type=Data scope=Private size=128K line_size=256 associativity=8
cache1          : level=1 type=Instruction scope=Private size=96K line_size=256 associativity=6
cache2          : level=2 type=Data scope=Private size=2048K line_size=256 associativity=8
cache3          : level=2 type=Instruction scope=Private size=2048K line_size=256 associativity=8
cache4          : level=3 type=Unified scope=Shared size=65536K line_size=256 associativity=16
cache5          : level=4 type=Unified scope=Shared size=491520K line_size=256 associativity=30
processor 0: version = FF,  identification = 2733E8,  machine = 2964
processor 1: version = FF,  identification = 2733E8,  machine = 2964
processor 2: version = FF,  identification = 2733E8,  machine = 2964
processor 3: version = FF,  identification = 2733E8,  machine = 2964

cpu number      : 0
cpu MHz dynamic : 5000
cpu MHz static  : 5000

cpu number      : 1
cpu MHz dynamic : 5000
cpu MHz static  : 5000

cpu number      : 2
cpu MHz dynamic : 5000
cpu MHz static  : 5000

cpu number      : 3
cpu MHz dynamic : 5000
cpu MHz static  : 5000
`

	cpuinfoMips = `
system type		: UBNT_E100
machine			: Unknown
processor		: 0
cpu model		: Cavium Octeon+ V0.1
BogoMIPS		: 1000.00
wait instruction	: yes
microsecond timers	: yes
tlb_entries		: 64
extra interrupt vector	: yes
hardware watchpoint	: yes, count: 2, address/irw mask: [0x0ffc, 0x0ffb]
isa			: mips1 mips2 mips3 mips4 mips5 mips64r2
ASEs implemented	:
shadow register sets	: 1
kscratch registers	: 0
core			: 0
VCED exceptions		: not available
VCEI exceptions		: not available

processor		: 1
cpu model		: Cavium Octeon+ V0.1
BogoMIPS		: 1000.00
wait instruction	: yes
microsecond timers	: yes
tlb_entries		: 64
extra interrupt vector	: yes
hardware watchpoint	: yes, count: 2, address/irw mask: [0x0ffc, 0x0ffb]
isa			: mips1 mips2 mips3 mips4 mips5 mips64r2
ASEs implemented	:
shadow register sets	: 1
kscratch registers	: 0
core			: 1
VCED exceptions		: not available
VCEI exceptions		: not available

`

	cpuinfoPpc64 = `
processor	: 0
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

processor	: 1
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

processor	: 2
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

processor	: 3
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

processor	: 4
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

processor	: 5
cpu		: POWER7 (architected), altivec supported
clock		: 3000.000000MHz
revision	: 2.1 (pvr 003f 0201)

timebase	: 512000000
platform	: pSeries
model		: IBM,8233-E8B
machine		: CHRP IBM,8233-E8B
`

	cpuinfoRiscv64 = `
processor	: 0
hart		: 0
isa		: rv64imafdcsu
mmu		: sv48

processor	: 1
hart		: 1
isa		: rv64imafdcsu
mmu		: sv48
`
)

func TestCPUInfoX86(t *testing.T) {
	parseCPUInfo = parseCPUInfoX86
	cpuinfo, err := getProcFixtures(t).CPUInfo()
	if err != nil {
		t.Fatal(err)
	}

	if cpuinfo == nil {
		t.Fatal("cpuinfo is nil")
	}

	if want, have := 8, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}

	if want, have := uint(7), cpuinfo[7].Processor; want != have {
		t.Errorf("want processor %v, have %v", want, have)
	}
	if want, have := "GenuineIntel", cpuinfo[0].VendorID; want != have {
		t.Errorf("want vendor %v, have %v", want, have)
	}
	if want, have := "6", cpuinfo[1].CPUFamily; want != have {
		t.Errorf("want family %v, have %v", want, have)
	}
	if want, have := "142", cpuinfo[2].Model; want != have {
		t.Errorf("want model %v, have %v", want, have)
	}
	if want, have := "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz", cpuinfo[3].ModelName; want != have {
		t.Errorf("want model %v, have %v", want, have)
	}
	if want, have := uint(8), cpuinfo[4].Siblings; want != have {
		t.Errorf("want siblings %v, have %v", want, have)
	}
	if want, have := "1", cpuinfo[5].CoreID; want != have {
		t.Errorf("want core id %v, have %v", want, have)
	}
	if want, have := uint(4), cpuinfo[6].CPUCores; want != have {
		t.Errorf("want cpu cores %v, have %v", want, have)
	}
	if want, have := "vme", cpuinfo[7].Flags[1]; want != have {
		t.Errorf("want flag %v, have %v", want, have)
	}
}

func TestCPUInfoParseARMLegacy(t *testing.T) {
	cpuinfo, err := parseCPUInfoARM([]byte(cpuinfoArm7Legacy))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse arm cpu info: %v", err)
	}
	if want, have := 2, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := "ARMv7 Processor rev 5 (v7l)", cpuinfo[0].ModelName; want != have {
		t.Errorf("want vendor %v, have %v", want, have)
	}
	if want, have := "thumb", cpuinfo[1].Flags[2]; want != have {
		t.Errorf("want flag %v, have %v", want, have)
	}
}

func TestCPUInfoParseARMLegacyV1(t *testing.T) {
	cpuinfo, err := parseCPUInfoARM([]byte(cpuinfoArm7LegacyV1))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse arm cpu info: %v", err)
	}
	if want, have := 1, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := "ARMv6-compatible processor rev 5 (v6l)", cpuinfo[0].ModelName; want != have {
		t.Errorf("want vendor %v, have %v", want, have)
	}
	if want, have := "thumb", cpuinfo[0].Flags[2]; want != have {
		t.Errorf("want flag %v, have %v", want, have)
	}
}

func TestCPUInfoParseARM(t *testing.T) {
	cpuinfo, err := parseCPUInfoARM([]byte(cpuinfoArm7))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse arm cpu info: %v", err)
	}
	if want, have := 4, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := "ARMv7 Processor rev 3 (v7l)", cpuinfo[0].ModelName; want != have {
		t.Errorf("want vendor %v, have %v", want, have)
	}
	if want, have := "thumb", cpuinfo[1].Flags[1]; want != have {
		t.Errorf("want flag %v, have %v", want, have)
	}
}

func TestCPUInfoParseS390X(t *testing.T) {
	cpuinfo, err := parseCPUInfoS390X([]byte(cpuinfoS390x))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse s390x cpu info: %v", err)
	}
	if want, have := 4, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := "IBM/S390", cpuinfo[0].VendorID; want != have {
		t.Errorf("want vendor %v, have %v", want, have)
	}
	if want, have := "ldisp", cpuinfo[1].Flags[4]; want != have {
		t.Errorf("want flag %v, have %v", want, have)
	}
	if want, have := 5000.0, cpuinfo[2].CPUMHz; want != have {
		t.Errorf("want cpu MHz %v, have %v", want, have)
	}
}

func TestCPUInfoParseMips(t *testing.T) {
	cpuinfo, err := parseCPUInfoMips([]byte(cpuinfoMips))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse mips cpu info: %v", err)
	}
	if want, have := 2, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := 1000.00, cpuinfo[0].BogoMips; want != have {
		t.Errorf("want BogoMIPS %v, have %v", want, have)
	}
	if want, have := "Cavium Octeon+ V0.1", cpuinfo[1].ModelName; want != have {
		t.Errorf("want ModelName '%v', have '%v'", want, have)
	}
}

func TestCPUInfoParsePPC(t *testing.T) {
	cpuinfo, err := parseCPUInfoPPC([]byte(cpuinfoPpc64))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse ppc cpu info: %v", err)
	}
	if want, have := 6, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := 3000.00, cpuinfo[2].CPUMHz; want != have {
		t.Errorf("want cpu mhz %v, have %v", want, have)
	}
}

func TestCPUInfoParseRISCV64(t *testing.T) {
	cpuinfo, err := parseCPUInfoRISCV([]byte(cpuinfoRiscv64))
	if err != nil || cpuinfo == nil {
		t.Fatalf("unable to parse ppc cpu info: %v", err)
	}
	if want, have := 2, len(cpuinfo); want != have {
		t.Errorf("want number of processors %v, have %v", want, have)
	}
	if want, have := "1", cpuinfo[1].CoreID; want != have {
		t.Errorf("want CoreId %v, have %v", want, have)
	}
	if want, have := "rv64imafdcsu", cpuinfo[1].ModelName; want != have {
		t.Errorf("want ModelName %v, have %v", want, have)
	}
}
