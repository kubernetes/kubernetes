package tsm1

import (
	"fmt"
	"reflect"
	"testing"
	"testing/quick"
)

func Test_StringEncoder_NoValues(t *testing.T) {
	enc := NewStringEncoder(1024)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec StringDecoder
	if err := dec.SetBytes(b); err != nil {
		t.Fatalf("unexpected error creating string decoder: %v", err)
	}
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_StringEncoder_Single(t *testing.T) {
	enc := NewStringEncoder(1024)
	v1 := "v1"
	enc.Write(v1)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec StringDecoder
	if dec.SetBytes(b); err != nil {
		t.Fatalf("unexpected error creating string decoder: %v", err)
	}
	if !dec.Next() {
		t.Fatalf("unexpected next value: got false, exp true")
	}

	if v1 != dec.Read() {
		t.Fatalf("unexpected value: got %v, exp %v", dec.Read(), v1)
	}
}

func Test_StringEncoder_Multi_Compressed(t *testing.T) {
	enc := NewStringEncoder(1024)

	values := make([]string, 10)
	for i := range values {
		values[i] = fmt.Sprintf("value %d", i)
		enc.Write(values[i])
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != stringCompressedSnappy {
		t.Fatalf("unexpected encoding: got %v, exp %v", b[0], stringCompressedSnappy)
	}

	if exp := 51; len(b) != exp {
		t.Fatalf("unexpected length: got %v, exp %v", len(b), exp)
	}

	var dec StringDecoder
	if err := dec.SetBytes(b); err != nil {
		t.Fatalf("unexpected erorr creating string decoder: %v", err)
	}

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

func Test_StringEncoder_Quick(t *testing.T) {
	quick.Check(func(values []string) bool {
		expected := values
		if values == nil {
			expected = []string{}
		}
		// Write values to encoder.
		enc := NewStringEncoder(1024)
		for _, v := range values {
			enc.Write(v)
		}

		// Retrieve encoded bytes from encoder.
		buf, err := enc.Bytes()
		if err != nil {
			t.Fatal(err)
		}

		// Read values out of decoder.
		got := make([]string, 0, len(values))
		var dec StringDecoder
		if err := dec.SetBytes(buf); err != nil {
			t.Fatal(err)
		}
		for dec.Next() {
			if err := dec.Error(); err != nil {
				t.Fatal(err)
			}
			got = append(got, dec.Read())
		}

		// Verify that input and output values match.
		if !reflect.DeepEqual(expected, got) {
			t.Fatalf("mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", expected, got)
		}

		return true
	}, nil)
}

func Test_StringDecoder_Empty(t *testing.T) {
	var dec StringDecoder
	if err := dec.SetBytes([]byte{}); err != nil {
		t.Fatal(err)
	}

	if dec.Next() {
		t.Fatalf("exp Next() == false, got true")
	}
}

func Test_StringDecoder_CorruptRead(t *testing.T) {
	cases := []string{
		"\x10\x03\b\x03Hi", // Higher length than actual data
		"\x10\x1dp\x9c\x90\x90\x90\x90\x90\x90\x90\x90\x90length overflow----",
	}

	for _, c := range cases {
		var dec StringDecoder
		if err := dec.SetBytes([]byte(c)); err != nil {
			t.Fatal(err)
		}

		if !dec.Next() {
			t.Fatalf("exp Next() to return true, got false")
		}

		_ = dec.Read()
		if dec.Error() == nil {
			t.Fatalf("exp an err, got nil: %q", c)
		}
	}
}

func Test_StringDecoder_CorruptSetBytes(t *testing.T) {
	cases := []string{
		"0t\x00\x01\x000\x00\x01\x000\x00\x01\x000\x00\x01\x000\x00\x01" +
			"\x000\x00\x01\x000\x00\x01\x000\x00\x00\x00\xff:\x01\x00\x01\x00\x01" +
			"\x00\x01\x00\x01\x00\x01\x00\x010\x010\x000\x010\x010\x010\x01" +
			"0\x010\x010\x010\x010\x010\x010\x010\x010\x010\x010", // Upper slice bounds overflows negative
	}

	for _, c := range cases {
		var dec StringDecoder
		if err := dec.SetBytes([]byte(c)); err == nil {
			t.Fatalf("exp an err, got nil: %q", c)
		}
	}
}
