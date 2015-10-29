package seccomp

import "strings"

type bpfLabel struct {
	label    string
	location uint32
}

type bpfLabels []bpfLabel

// labelIndex returns the index for the label if it exists in the slice.
// if it does not exist in the slice it appends the label lb to the end
// of the slice and returns the index.
func labelIndex(labels *bpfLabels, lb string) uint32 {
	var id uint32
	for id = 0; id < uint32(len(*labels)); id++ {
		if strings.EqualFold(lb, (*labels)[id].label) {
			return id
		}
	}
	*labels = append(*labels, bpfLabel{lb, 0xffffffff})
	return id
}

func scmpBpfStmt(code uint16, k uint32) sockFilter {
	return sockFilter{code, 0, 0, k}
}

func scmpBpfJump(code uint16, k uint32, jt, jf uint8) sockFilter {
	return sockFilter{code, jt, jf, k}
}
