// +build fuzz

package dns

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
	if _, err := NewRR(string(data)); err != nil {
		return 0
	}
	return 1
}
