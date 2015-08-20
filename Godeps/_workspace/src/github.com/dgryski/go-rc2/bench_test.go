package rc2

import (
	"testing"

	ebfe "github.com/ebfe/rc2"
)

func BenchmarkEncrypt(b *testing.B) {
	r, _ := New([]byte{0, 0, 0, 0, 0, 0, 0, 0}, 64)
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Encrypt(src[:], src[:])
	}
}
func BenchmarkEbfeEncrypt(b *testing.B) {
	r, _ := ebfe.NewCipher([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Encrypt(src[:], src[:])
	}
}
func BenchmarkDecrypt(b *testing.B) {
	r, _ := New([]byte{0, 0, 0, 0, 0, 0, 0, 0}, 64)
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Decrypt(src[:], src[:])
	}
}
func BenchmarkEbfeDecrypt(b *testing.B) {
	r, _ := ebfe.NewCipher([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	b.ResetTimer()
	var src [8]byte
	for i := 0; i < b.N; i++ {
		r.Decrypt(src[:], src[:])
	}
}
