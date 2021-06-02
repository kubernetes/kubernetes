// +build fuzz

package dns

import "strings"

func Fuzz(data []byte) int {
	msg := new(Msg)

	if err := msg.Unpack(data); err != nil {
		return 0
	}
	if _, err := msg.Pack(); err != nil {
		return 0
	}

	return 1
}

func FuzzNewRR(data []byte) int {
	str := string(data)
	// Do not fuzz lines that include the $INCLUDE keyword and hint the fuzzer
	// at avoiding them.
	// See GH#1025 for context.
	if strings.Contains(strings.ToUpper(str), "$INCLUDE") {
		return -1
	}
	if _, err := NewRR(str); err != nil {
		return 0
	}
	return 1
}
