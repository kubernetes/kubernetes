package seccomp

import (
	"fmt"
	"syscall"
	"unsafe"
)

type sockFilter struct {
	code uint16
	jt   uint8
	jf   uint8
	k    uint32
}

func newFilter() *filter {
	var f filter
	f = append(f, sockFilter{
		pfLD + syscall.BPF_W + syscall.BPF_ABS,
		0,
		0,
		uint32(unsafe.Offsetof(secData.nr)),
	})
	return &f
}

type filter []sockFilter

func (f *filter) addSyscall(s *Syscall, labels *bpfLabels) {
	if len(s.Args) == 0 {
		f.call(s.Value, scmpBpfStmt(syscall.BPF_RET+syscall.BPF_K, s.scmpAction()))
	} else {
		if len(s.Args[0]) > 0 {
			lb := fmt.Sprintf(labelTemplate, s.Value, s.Args[0][0].Index)
			f.call(s.Value,
				scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JA, labelIndex(labels, lb),
					jumpJT, jumpJF))
		}
	}
}

func (f *filter) addArguments(s *Syscall, labels *bpfLabels) error {
	for i := 0; len(s.Args) > i; i++ {
		if len(s.Args[i]) > 0 {
			lb := fmt.Sprintf(labelTemplate, s.Value, s.Args[i][0].Index)
			f.label(labels, lb)
			f.arg(s.Args[i][0].Index)
		}
		for j := 0; j < len(s.Args[i]); j++ {
			var jf sockFilter
			if len(s.Args)-1 > i && len(s.Args[i+1]) > 0 {
				lbj := fmt.Sprintf(labelTemplate, s.Value, s.Args[i+1][0].Index)
				jf = scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JA,
					labelIndex(labels, lbj), jumpJT, jumpJF)
			} else {
				jf = scmpBpfStmt(syscall.BPF_RET+syscall.BPF_K, s.scmpAction())
			}
			if err := f.op(s.Args[i][j].Op, s.Args[i][j].Value, jf); err != nil {
				return err
			}
		}
		f.allow()
	}
	return nil
}

func (f *filter) label(labels *bpfLabels, lb string) {
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JA, labelIndex(labels, lb), labelJT, labelJF))
}

func (f *filter) call(nr uint32, jt sockFilter) {
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JEQ+syscall.BPF_K, nr, 0, 1))
	*f = append(*f, jt)
}

func (f *filter) allow() {
	*f = append(*f, scmpBpfStmt(syscall.BPF_RET+syscall.BPF_K, retAllow))
}

func (f *filter) deny() {
	*f = append(*f, scmpBpfStmt(syscall.BPF_RET+syscall.BPF_K, retTrap))
}

func (f *filter) arg(index uint32) {
	arg(f, index)
}

func (f *filter) op(operation Operator, v uint, jf sockFilter) error {
	switch operation {
	case EqualTo:
		jumpEqualTo(f, v, jf)
	case NotEqualTo:
		jumpNotEqualTo(f, v, jf)
	case GreatherThan:
		jumpGreaterThan(f, v, jf)
	case LessThan:
		jumpLessThan(f, v, jf)
	case MaskEqualTo:
		jumpMaskEqualTo(f, v, jf)
	default:
		return ErrUnsupportedOperation
	}
	return nil
}

func arg(f *filter, idx uint32) {
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_W+syscall.BPF_ABS, endian.low(idx)))
	*f = append(*f, scmpBpfStmt(syscall.BPF_ST, 0))
	*f = append(*f, scmpBpfStmt(syscall.BPF_LD+syscall.BPF_W+syscall.BPF_ABS, endian.hi(idx)))
	*f = append(*f, scmpBpfStmt(syscall.BPF_ST, 1))
}

func jump(f *filter, labels *bpfLabels, lb string) {
	*f = append(*f, scmpBpfJump(syscall.BPF_JMP+syscall.BPF_JA, labelIndex(labels, lb),
		jumpJT, jumpJF))
}
