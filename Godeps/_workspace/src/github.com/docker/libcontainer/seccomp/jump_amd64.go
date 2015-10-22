// +build linux,amd64

package seccomp

// Using BPF filters
//
// ref: http://www.gsp.com/cgi-bin/man.cgi?topic=bpf
import "syscall"

func jumpGreaterThan(f *filter, v uint, jt sockFilter) {
	lo := uint32(uint64(v) % 0x100000000)
	hi := uint32(uint64(v) / 0x100000000)
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JGT+syscall.BPF_K, (hi), 4, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, (hi), 0, 5))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JGE+syscall.BPF_K, (lo), 0, 2))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
	*f = append(*f, jt)
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
}

func jumpEqualTo(f *filter, v uint, jt sockFilter) {
	lo := uint32(uint64(v) % 0x100000000)
	hi := uint32(uint64(v) / 0x100000000)
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, (hi), 0, 5))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, (lo), 0, 2))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
	*f = append(*f, jt)
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
}

func jumpLessThan(f *filter, v uint, jt sockFilter) {
	lo := uint32(uint64(v) % 0x100000000)
	hi := uint32(uint64(v) / 0x100000000)
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JGT+syscall.BPF_K, (hi), 6, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, (hi), 0, 3))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JGT+syscall.BPF_K, (lo), 2, 0))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
	*f = append(*f, jt)
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
}

func jumpNotEqualTo(f *filter, v uint, jt sockFilter) {
	lo := uint32(uint64(v) % 0x100000000)
	hi := uint32(uint64(v) / 0x100000000)
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, hi, 5, 0))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 0))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, lo, 2, 0))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
	*f = append(*f, jt)
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
}

// this checks for a value inside a mask. The evalusation is equal to doing
// CLONE_NEWUSER & syscallMask == CLONE_NEWUSER
func jumpMaskEqualTo(f *filter, v uint, jt sockFilter) {
	lo := uint32(uint64(v) % 0x100000000)
	hi := uint32(uint64(v) / 0x100000000)
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, hi, 0, 6))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 0))
	*f = append(*f, scmpBpfStmt(syscall.BPF_ALU+syscall.BPF_AND, uint32(v)))
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, lo, 0, 2))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
	*f = append(*f, jt)
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_MEM, 1))
}
