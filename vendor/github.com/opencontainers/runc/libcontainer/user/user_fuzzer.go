// +build gofuzz

package user

import (
	"io"
	"strings"
)

func IsDivisbleBy(n int, divisibleby int) bool {
	return (n % divisibleby) == 0
}

func FuzzUser(data []byte) int {
	if len(data) == 0 {
		return -1
	}
	if !IsDivisbleBy(len(data), 5) {
		return -1
	}

	var divided [][]byte

	chunkSize := len(data) / 5

	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize

		divided = append(divided, data[i:end])
	}

	_, _ = ParsePasswdFilter(strings.NewReader(string(divided[0])), nil)

	var passwd, group io.Reader

	group = strings.NewReader(string(divided[1]))
	_, _ = GetAdditionalGroups([]string{string(divided[2])}, group)

	passwd = strings.NewReader(string(divided[3]))
	_, _ = GetExecUser(string(divided[4]), nil, passwd, group)
	return 1
}
