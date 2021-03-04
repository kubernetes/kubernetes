package types // import "github.com/docker/docker/api/types"

// Seccomp represents the config for a seccomp profile for syscall restriction.
type Seccomp struct {
	DefaultAction Action `json:"defaultAction"`
	// Architectures is kept to maintain backward compatibility with the old
	// seccomp profile.
	Architectures []Arch         `json:"architectures,omitempty"`
	ArchMap       []Architecture `json:"archMap,omitempty"`
	Syscalls      []*Syscall     `json:"syscalls"`
}

// Architecture is used to represent a specific architecture
// and its sub-architectures
type Architecture struct {
	Arch      Arch   `json:"architecture"`
	SubArches []Arch `json:"subArchitectures"`
}

// Arch used for architectures
type Arch string

// Additional architectures permitted to be used for system calls
// By default only the native architecture of the kernel is permitted
const (
	ArchX86         Arch = "SCMP_ARCH_X86"
	ArchX86_64      Arch = "SCMP_ARCH_X86_64"
	ArchX32         Arch = "SCMP_ARCH_X32"
	ArchARM         Arch = "SCMP_ARCH_ARM"
	ArchAARCH64     Arch = "SCMP_ARCH_AARCH64"
	ArchMIPS        Arch = "SCMP_ARCH_MIPS"
	ArchMIPS64      Arch = "SCMP_ARCH_MIPS64"
	ArchMIPS64N32   Arch = "SCMP_ARCH_MIPS64N32"
	ArchMIPSEL      Arch = "SCMP_ARCH_MIPSEL"
	ArchMIPSEL64    Arch = "SCMP_ARCH_MIPSEL64"
	ArchMIPSEL64N32 Arch = "SCMP_ARCH_MIPSEL64N32"
	ArchPPC         Arch = "SCMP_ARCH_PPC"
	ArchPPC64       Arch = "SCMP_ARCH_PPC64"
	ArchPPC64LE     Arch = "SCMP_ARCH_PPC64LE"
	ArchS390        Arch = "SCMP_ARCH_S390"
	ArchS390X       Arch = "SCMP_ARCH_S390X"
)

// Action taken upon Seccomp rule match
type Action string

// Define actions for Seccomp rules
const (
	ActKill  Action = "SCMP_ACT_KILL"
	ActTrap  Action = "SCMP_ACT_TRAP"
	ActErrno Action = "SCMP_ACT_ERRNO"
	ActTrace Action = "SCMP_ACT_TRACE"
	ActAllow Action = "SCMP_ACT_ALLOW"
)

// Operator used to match syscall arguments in Seccomp
type Operator string

// Define operators for syscall arguments in Seccomp
const (
	OpNotEqual     Operator = "SCMP_CMP_NE"
	OpLessThan     Operator = "SCMP_CMP_LT"
	OpLessEqual    Operator = "SCMP_CMP_LE"
	OpEqualTo      Operator = "SCMP_CMP_EQ"
	OpGreaterEqual Operator = "SCMP_CMP_GE"
	OpGreaterThan  Operator = "SCMP_CMP_GT"
	OpMaskedEqual  Operator = "SCMP_CMP_MASKED_EQ"
)

// Arg used for matching specific syscall arguments in Seccomp
type Arg struct {
	Index    uint     `json:"index"`
	Value    uint64   `json:"value"`
	ValueTwo uint64   `json:"valueTwo"`
	Op       Operator `json:"op"`
}

// Filter is used to conditionally apply Seccomp rules
type Filter struct {
	Caps      []string `json:"caps,omitempty"`
	Arches    []string `json:"arches,omitempty"`
	MinKernel string   `json:"minKernel,omitempty"`
}

// Syscall is used to match a group of syscalls in Seccomp
type Syscall struct {
	Name     string   `json:"name,omitempty"`
	Names    []string `json:"names,omitempty"`
	Action   Action   `json:"action"`
	Args     []*Arg   `json:"args"`
	Comment  string   `json:"comment"`
	Includes Filter   `json:"includes"`
	Excludes Filter   `json:"excludes"`
}
