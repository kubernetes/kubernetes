//go:build gofuzz
// +build gofuzz

package userns

func FuzzUIDMap(uidmap []byte) int {
	_ = uidMapInUserNS(string(uidmap))
	return 1
}
