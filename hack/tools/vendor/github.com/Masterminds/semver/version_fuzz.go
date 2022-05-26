// +build gofuzz

package semver

func Fuzz(data []byte) int {
	if _, err := NewVersion(string(data)); err != nil {
		return 0
	}
	return 1
}
