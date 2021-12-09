package ber

func encodeUnsignedInteger(i uint64) []byte {
	n := uint64Length(i)
	out := make([]byte, n)

	var j int
	for ; n > 0; n-- {
		out[j] = (byte(i >> uint((n-1)*8)))
		j++
	}

	return out
}

func uint64Length(i uint64) (numBytes int) {
	numBytes = 1

	for i > 255 {
		numBytes++
		i >>= 8
	}

	return
}
