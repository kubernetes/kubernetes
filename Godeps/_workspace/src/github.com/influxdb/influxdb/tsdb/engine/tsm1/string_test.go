package tsm1

import (
	"fmt"
	"reflect"
	"testing"
	"testing/quick"
)

func Test_StringEncoder_NoValues(t *testing.T) {
	enc := NewStringEncoder()
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dec, err := NewStringDecoder(b)
	if err != nil {
		t.Fatalf("unexpected erorr creating string decoder: %v", err)
	}
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_StringEncoder_Single(t *testing.T) {
	enc := NewStringEncoder()
	v1 := "v1"
	enc.Write(v1)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dec, err := NewStringDecoder(b)
	if err != nil {
		t.Fatalf("unexpected erorr creating string decoder: %v", err)
	}
	if !dec.Next() {
		t.Fatalf("unexpected next value: got false, exp true")
	}

	if v1 != dec.Read() {
		t.Fatalf("unexpected value: got %v, exp %v", dec.Read(), v1)
	}
}

func Test_StringEncoder_Multi_Compressed(t *testing.T) {
	enc := NewStringEncoder()

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

	if exp := 47; len(b) != exp {
		t.Fatalf("unexpected length: got %v, exp %v", len(b), exp)
	}

	dec, err := NewStringDecoder(b)
	if err != nil {
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
		// Write values to encoder.
		enc := NewStringEncoder()
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
		dec, err := NewStringDecoder(buf)
		if err != nil {
			t.Fatal(err)
		}
		for dec.Next() {
			if err := dec.Error(); err != nil {
				t.Fatal(err)
			}
			got = append(got, dec.Read())
		}

		// Verify that input and output values match.
		if !reflect.DeepEqual(values, got) {
			t.Fatalf("mismatch:\n\nexp=%+v\n\ngot=%+v\n\n", values, got)
		}

		return true
	}, nil)
}
