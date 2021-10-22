package xxhash

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"strings"
	"testing"

	OneOfOne "github.com/OneOfOne/xxhash"
	"github.com/spaolacci/murmur3"
)

var result uint64

func BenchmarkStringHash(b *testing.B) {
	const s = "abcdefghijklmnop"
	var r uint64
	b.ReportAllocs()
	for n := 0; n < b.N; n++ {
		r = Sum64([]byte(s))
	}
	result = r
}

func TestSum(t *testing.T) {
	for i, tt := range []struct {
		input string
		want  uint64
	}{
		{"", 0xef46db3751d8e999},
		{"a", 0xd24ec4f1a98c6e5b},
		{"as", 0x1c330fb2d66be179},
		{"asd", 0x631c37ce72a97393},
		{"asdf", 0x415872f599cea71e},
		{
			// Exactly 63 characters, which exercises all code paths.
			"Call me Ishmael. Some years ago--never mind how long precisely-",
			0x02a2e85470d6fd96,
		},
	} {
		for chunkSize := 1; chunkSize <= len(tt.input); chunkSize++ {
			x := New()
			for j := 0; j < len(tt.input); j += chunkSize {
				end := j + chunkSize
				if end > len(tt.input) {
					end = len(tt.input)
				}
				chunk := []byte(tt.input[j:end])
				n, err := x.Write(chunk)
				if err != nil || n != len(chunk) {
					t.Fatalf("[i=%d,chunkSize=%d] Write: got (%d, %v); want (%d, nil)",
						i, chunkSize, n, err, len(chunk))
				}
			}
			if got := x.Sum64(); got != tt.want {
				t.Fatalf("[i=%d,chunkSize=%d] got 0x%x; want 0x%x",
					i, chunkSize, got, tt.want)
			}
			var b [8]byte
			binary.BigEndian.PutUint64(b[:], tt.want)
			if got := x.Sum(nil); !bytes.Equal(got, b[:]) {
				t.Fatalf("[i=%d,chunkSize=%d] Sum: got %v; want %v",
					i, chunkSize, got, b[:])
			}
		}
		if got := Sum64([]byte(tt.input)); got != tt.want {
			t.Fatalf("[i=%d] Sum64: got 0x%x; want 0x%x", i, got, tt.want)
		}
		if got := Sum64String(tt.input); got != tt.want {
			t.Fatalf("[i=%d] Sum64String: got 0x%x; want 0x%x", i, got, tt.want)
		}
	}
}

func TestReset(t *testing.T) {
	parts := []string{"The quic", "k br", "o", "wn fox jumps", " ov", "er the lazy ", "dog."}
	x := New()
	for _, part := range parts {
		x.Write([]byte(part))
	}
	h0 := x.Sum64()

	x.Reset()
	x.Write([]byte(strings.Join(parts, "")))
	h1 := x.Sum64()

	if h0 != h1 {
		t.Errorf("0x%x != 0x%x", h0, h1)
	}
}

var (
	sink  uint64
	sinkb []byte
)

func sumFunc(h hash.Hash) func(b []byte) uint64 {
	return func(b []byte) uint64 {
		h.Reset()
		h.Write(b)
		sinkb = h.Sum(nil)
		return 0 // value doesn't matter
	}
}

func BenchmarkHashes(b *testing.B) {
	for _, ht := range []struct {
		name string
		f    interface{}
	}{
		{"xxhash", Sum64},
		{"xxhash-string", Sum64String},
		{"OneOfOne", OneOfOne.Checksum64},
		{"murmur3", murmur3.Sum64},
		{"CRC-32", sumFunc(crc32.NewIEEE())},
	} {
		for _, nt := range []struct {
			name string
			n    int
		}{
			{"5 B", 5},
			{"100 B", 100},
			{"4 KB", 4e3},
			{"10 MB", 10e6},
		} {
			input := make([]byte, nt.n)
			for i := range input {
				input[i] = byte(i)
			}
			benchName := fmt.Sprintf("%s,n=%s", ht.name, nt.name)
			if ht.name == "xxhash-string" {
				f := ht.f.(func(string) uint64)
				s := string(input)
				b.Run(benchName, func(b *testing.B) {
					b.SetBytes(int64(len(input)))
					for i := 0; i < b.N; i++ {
						sink = f(s)
					}
				})
			} else {
				f := ht.f.(func([]byte) uint64)
				b.Run(benchName, func(b *testing.B) {
					b.SetBytes(int64(len(input)))
					for i := 0; i < b.N; i++ {
						sink = f(input)
					}
				})
			}
		}
	}
}
