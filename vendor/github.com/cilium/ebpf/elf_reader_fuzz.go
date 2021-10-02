// +build gofuzz

// Use with https://github.com/dvyukov/go-fuzz

package ebpf

import "bytes"

func FuzzLoadCollectionSpec(data []byte) int {
	spec, err := LoadCollectionSpecFromReader(bytes.NewReader(data))
	if err != nil {
		if spec != nil {
			panic("spec is not nil")
		}
		return 0
	}
	if spec == nil {
		panic("spec is nil")
	}
	return 1
}
