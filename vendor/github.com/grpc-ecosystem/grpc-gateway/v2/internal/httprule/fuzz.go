//go:build gofuzz
// +build gofuzz

package httprule

func Fuzz(data []byte) int {
	if _, err := Parse(string(data)); err != nil {
		return 0
	}
	return 0
}
