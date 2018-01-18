package nl

import "encoding/binary"

const (
	MPLS_LS_LABEL_SHIFT = 12
	MPLS_LS_S_SHIFT     = 8
)

func EncodeMPLSStack(labels ...int) []byte {
	b := make([]byte, 4*len(labels))
	for idx, label := range labels {
		l := label << MPLS_LS_LABEL_SHIFT
		if idx == len(labels)-1 {
			l |= 1 << MPLS_LS_S_SHIFT
		}
		binary.BigEndian.PutUint32(b[idx*4:], uint32(l))
	}
	return b
}

func DecodeMPLSStack(buf []byte) []int {
	if len(buf)%4 != 0 {
		return nil
	}
	stack := make([]int, 0, len(buf)/4)
	for len(buf) > 0 {
		l := binary.BigEndian.Uint32(buf[:4])
		buf = buf[4:]
		stack = append(stack, int(l)>>MPLS_LS_LABEL_SHIFT)
		if (l>>MPLS_LS_S_SHIFT)&1 > 0 {
			break
		}
	}
	return stack
}
