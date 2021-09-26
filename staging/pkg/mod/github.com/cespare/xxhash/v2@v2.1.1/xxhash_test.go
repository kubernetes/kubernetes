package xxhash

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"
)

func TestAll(t *testing.T) {
	for _, tt := range []struct {
		name  string
		input string
		want  uint64
	}{
		{"empty", "", 0xef46db3751d8e999},
		{"a", "a", 0xd24ec4f1a98c6e5b},
		{"as", "as", 0x1c330fb2d66be179},
		{"asd", "asd", 0x631c37ce72a97393},
		{"asdf", "asdf", 0x415872f599cea71e},
		{
			"len=63",
			// Exactly 63 characters, which exercises all code paths.
			"Call me Ishmael. Some years ago--never mind how long precisely-",
			0x02a2e85470d6fd96,
		},
	} {
		for chunkSize := 1; chunkSize <= len(tt.input); chunkSize++ {
			name := fmt.Sprintf("%s,chunkSize=%d", tt.name, chunkSize)
			t.Run(name, func(t *testing.T) {
				testDigest(t, tt.input, chunkSize, tt.want)
			})
		}
		t.Run(tt.name, func(t *testing.T) { testSum(t, tt.input, tt.want) })
	}
}

func testDigest(t *testing.T, input string, chunkSize int, want uint64) {
	d := New()
	ds := New() // uses WriteString
	for i := 0; i < len(input); i += chunkSize {
		chunk := input[i:]
		if len(chunk) > chunkSize {
			chunk = chunk[:chunkSize]
		}
		n, err := d.Write([]byte(chunk))
		if err != nil || n != len(chunk) {
			t.Fatalf("Digest.Write: got (%d, %v); want (%d, nil)", n, err, len(chunk))
		}
		n, err = ds.WriteString(chunk)
		if err != nil || n != len(chunk) {
			t.Fatalf("Digest.WriteString: got (%d, %v); want (%d, nil)", n, err, len(chunk))
		}
	}
	if got := d.Sum64(); got != want {
		t.Fatalf("Digest.Sum64: got 0x%x; want 0x%x", got, want)
	}
	if got := ds.Sum64(); got != want {
		t.Fatalf("Digest.Sum64 (WriteString): got 0x%x; want 0x%x", got, want)
	}
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], want)
	if got := d.Sum(nil); !bytes.Equal(got, b[:]) {
		t.Fatalf("Sum: got %v; want %v", got, b[:])
	}
}

func testSum(t *testing.T, input string, want uint64) {
	if got := Sum64([]byte(input)); got != want {
		t.Fatalf("Sum64: got 0x%x; want 0x%x", got, want)
	}
	if got := Sum64String(input); got != want {
		t.Fatalf("Sum64String: got 0x%x; want 0x%x", got, want)
	}
}

func TestReset(t *testing.T) {
	parts := []string{"The quic", "k br", "o", "wn fox jumps", " ov", "er the lazy ", "dog."}
	d := New()
	for _, part := range parts {
		d.Write([]byte(part))
	}
	h0 := d.Sum64()

	d.Reset()
	d.Write([]byte(strings.Join(parts, "")))
	h1 := d.Sum64()

	if h0 != h1 {
		t.Errorf("0x%x != 0x%x", h0, h1)
	}
}

func TestBinaryMarshaling(t *testing.T) {
	d := New()
	d.WriteString("abc")
	b, err := d.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	d = New()
	d.WriteString("junk")
	if err := d.UnmarshalBinary(b); err != nil {
		t.Fatal(err)
	}
	d.WriteString("def")
	if got, want := d.Sum64(), Sum64String("abcdef"); got != want {
		t.Fatalf("after MarshalBinary+UnmarshalBinary, got 0x%x; want 0x%x", got, want)
	}

	d0 := New()
	d1 := New()
	for i := 0; i < 64; i++ {
		b, err := d0.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		d0 = new(Digest)
		if err := d0.UnmarshalBinary(b); err != nil {
			t.Fatal(err)
		}
		if got, want := d0.Sum64(), d1.Sum64(); got != want {
			t.Fatalf("after %d Writes, unmarshaled Digest gave sum 0x%x; want 0x%x", i, got, want)
		}

		d0.Write([]byte{'a'})
		d1.Write([]byte{'a'})
	}
}

var sink uint64

func TestAllocs(t *testing.T) {
	const shortStr = "abcdefghijklmnop"
	// Sum64([]byte(shortString)) shouldn't allocate because the
	// intermediate []byte ought not to escape.
	// (See https://github.com/cespare/xxhash/pull/2.)
	t.Run("Sum64", func(t *testing.T) {
		testAllocs(t, func() {
			sink = Sum64([]byte(shortStr))
		})
	})
	// Creating and using a Digest shouldn't allocate because its methods
	// shouldn't make it escape. (A previous version of New returned a
	// hash.Hash64 which forces an allocation.)
	t.Run("Digest", func(t *testing.T) {
		b := []byte("asdf")
		testAllocs(t, func() {
			d := New()
			d.Write(b)
			sink = d.Sum64()
		})
	})
}

func testAllocs(t *testing.T, fn func()) {
	t.Helper()
	if allocs := int(testing.AllocsPerRun(10, fn)); allocs > 0 {
		t.Fatalf("got %d allocation(s) (want zero)", allocs)
	}
}
