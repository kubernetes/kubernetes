package tsm1_test

import (
	"reflect"
	"testing"
	"testing/quick"

	"github.com/influxdb/influxdb/tsdb/engine/tsm1"
)

func Test_BoolEncoder_NoValues(t *testing.T) {
	enc := tsm1.NewBoolEncoder()
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dec := tsm1.NewBoolDecoder(b)
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_BoolEncoder_Single(t *testing.T) {
	enc := tsm1.NewBoolEncoder()
	v1 := true
	enc.Write(v1)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dec := tsm1.NewBoolDecoder(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got false, exp true")
	}

	if v1 != dec.Read() {
		t.Fatalf("unexpected value: got %v, exp %v", dec.Read(), v1)
	}
}

func Test_BoolEncoder_Multi_Compressed(t *testing.T) {
	enc := tsm1.NewBoolEncoder()

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

	dec := tsm1.NewBoolDecoder(b)

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

func Test_BoolEncoder_Quick(t *testing.T) {
	if err := quick.Check(func(values []bool) bool {
		// Write values to encoder.
		enc := tsm1.NewBoolEncoder()
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
		dec := tsm1.NewBoolDecoder(buf)
		for dec.Next() {
			got = append(got, dec.Read())
		}

		// Verify that input and output values match.
		if !reflect.DeepEqual(values, got) {
			t.Fatalf("mismatch:\n\nexp=%+v\n\ngot=%+v\n\n", values, got)
		}

		return true
	}, nil); err != nil {
		t.Fatal(err)
	}
}
