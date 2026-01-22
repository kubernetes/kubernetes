package ansiterm

import (
	"strconv"
)

func sliceContains(bytes []byte, b byte) bool {
	for _, v := range bytes {
		if v == b {
			return true
		}
	}

	return false
}

func convertBytesToInteger(bytes []byte) int {
	s := string(bytes)
	i, _ := strconv.Atoi(s)
	return i
}
