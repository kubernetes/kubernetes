// +build gofuzz

package printf

func Fuzz(data []byte) int {
	_, err := Parse(string(data))
	if err == nil {
		return 1
	}
	return 0
}
