package tsm1_test

import (
	"reflect"
	"testing"
	"testing/quick"

	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

func Test_BooleanEncoder_NoValues(t *testing.T) {
	enc := tsm1.NewBooleanEncoder(0)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec tsm1.BooleanDecoder
	dec.SetBytes(b)
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_BooleanEncoder_Single(t *testing.T) {
	enc := tsm1.NewBooleanEncoder(1)
	v1 := true
	enc.Write(v1)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec tsm1.BooleanDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got false, exp true")
	}

	if v1 != dec.Read() {
		t.Fatalf("unexpected value: got %v, exp %v", dec.Read(), v1)
	}
}

func Test_BooleanEncoder_Multi_Compressed(t *testing.T) {
	enc := tsm1.NewBooleanEncoder(10)

	values := make([]bool, 10)
	for i := range values {
		values[i] = i%2 == 0
		enc.Write(values[i])
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if exp := 4; len(b) != exp {
		t.Fatalf("unexpected length: got %v, exp %v", len(b), exp)
	}

	var dec tsm1.BooleanDecoder
	dec.SetBytes(b)

	for i, v := range values {
		if !dec.Next() {
			t.Fatalf("unexpected next value: got false, exp true")
		}
		if v != dec.Read() {
			t.Fatalf("unexpected value at pos %d: got %v, exp %v", i, dec.Read(), v)
		}
	}

	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_BooleanEncoder_Quick(t *testing.T) {
	if err := quick.Check(func(values []bool) bool {
		expected := values
		if values == nil {
			expected = []bool{}
		}
		// Write values to encoder.
		enc := tsm1.NewBooleanEncoder(1024)
		for _, v := range values {
			enc.Write(v)
		}

		// Retrieve compressed bytes.
		buf, err := enc.Bytes()
		if err != nil {
			t.Fatal(err)
		}

		// Read values out of decoder.
		got := make([]bool, 0, len(values))
		var dec tsm1.BooleanDecoder
		dec.SetBytes(buf)
		for dec.Next() {
			got = append(got, dec.Read())
		}

		// Verify that input and output values match.
		if !reflect.DeepEqual(expected, got) {
			t.Fatalf("mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", expected, got)
		}

		return true
	}, nil); err != nil {
		t.Fatal(err)
	}
}

func Test_BooleanDecoder_Corrupt(t *testing.T) {
	cases := []string{
		"",         // Empty
		"\x10\x90", // Packed: invalid count
		"\x10\x7f", // Packed: count greater than remaining bits, multiple bytes expected
		"\x10\x01", // Packed: count greater than remaining bits, one byte expected
	}

	for _, c := range cases {
		var dec tsm1.BooleanDecoder
		dec.SetBytes([]byte(c))
		if dec.Next() {
			t.Fatalf("exp next == false, got true for case %q", c)
		}
	}
}

func BenchmarkBooleanDecoder_2048(b *testing.B) { benchmarkBooleanDecoder(b, 2048) }

func benchmarkBooleanDecoder(b *testing.B, size int) {
	e := tsm1.NewBooleanEncoder(size)
	for i := 0; i < size; i++ {
		e.Write(i&1 == 1)
	}
	bytes, err := e.Bytes()
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var d tsm1.BooleanDecoder
		d.SetBytes(bytes)

		var n int
		for d.Next() {
			_ = d.Read()
			n++
		}
		if n != size {
			b.Fatalf("expected to read %d booleans, but read %d", size, n)
		}
	}
}
