// +build gofuzz

package roaring

import "bytes"

func FuzzSerializationStream(data []byte) int {
	newrb := NewBitmap()
	if _, err := newrb.ReadFrom(bytes.NewReader(data)); err != nil {
		return 0
	}
	return 1
}

func FuzzSerializationBuffer(data []byte) int {
	newrb := NewBitmap()
	if _, err := newrb.FromBuffer(data); err != nil {
		return 0
	}
	return 1
}
