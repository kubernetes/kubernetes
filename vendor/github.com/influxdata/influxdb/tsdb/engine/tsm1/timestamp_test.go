package tsm1

import (
	"reflect"
	"testing"
	"testing/quick"
	"time"
)

func Test_TimeEncoder(t *testing.T) {
	enc := NewTimeEncoder(1)

	x := []int64{}
	now := time.Unix(0, 0)
	x = append(x, now.UnixNano())
	enc.Write(now.UnixNano())
	for i := 1; i < 4; i++ {
		x = append(x, now.Add(time.Duration(i)*time.Second).UnixNano())
		enc.Write(x[i])
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	for i, v := range x {
		if !dec.Next() {
			t.Fatalf("Next == false, expected true")
		}

		if v != dec.Read() {
			t.Fatalf("Item %d mismatch, got %v, exp %v", i, dec.Read(), v)
		}
	}
}

func Test_TimeEncoder_NoValues(t *testing.T) {
	enc := NewTimeEncoder(0)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec TimeDecoder
	dec.Init(b)
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_TimeEncoder_One(t *testing.T) {
	enc := NewTimeEncoder(1)
	var tm int64

	enc.Write(tm)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedPackedSimple {
		t.Fatalf("Wrong encoding used: expected uncompressed, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if tm != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), tm)
	}
}

func Test_TimeEncoder_Two(t *testing.T) {
	enc := NewTimeEncoder(2)
	t1 := int64(0)
	t2 := int64(1)
	enc.Write(t1)
	enc.Write(t2)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t2)
	}
}

func Test_TimeEncoder_Three(t *testing.T) {
	enc := NewTimeEncoder(3)
	t1 := int64(0)
	t2 := int64(1)
	t3 := int64(3)

	enc.Write(t1)
	enc.Write(t2)
	enc.Write(t3)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedPackedSimple {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t2)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t3 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t3)
	}
}

func Test_TimeEncoder_Large_Range(t *testing.T) {
	enc := NewTimeEncoder(2)
	t1 := int64(1442369134000000000)
	t2 := int64(1442369135000000000)
	enc.Write(t1)
	enc.Write(t2)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t2)
	}
}

func Test_TimeEncoder_Uncompressed(t *testing.T) {
	enc := NewTimeEncoder(3)
	t1 := time.Unix(0, 0).UnixNano()
	t2 := time.Unix(1, 0).UnixNano()

	// about 36.5yrs in NS resolution is max range for compressed format
	// This should cause the encoding to fallback to raw points
	t3 := time.Unix(2, (2 << 59)).UnixNano()
	enc.Write(t1)
	enc.Write(t2)
	enc.Write(t3)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("expected error: %v", err)
	}

	if exp := 25; len(b) != exp {
		t.Fatalf("length mismatch: got %v, exp %v", len(b), exp)
	}

	if got := b[0] >> 4; got != timeUncompressed {
		t.Fatalf("Wrong encoding used: expected uncompressed, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t2)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if t3 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), t3)
	}
}

func Test_TimeEncoder_RLE(t *testing.T) {
	enc := NewTimeEncoder(512)
	var ts []int64
	for i := 0; i < 500; i++ {
		ts = append(ts, int64(i))
	}

	for _, v := range ts {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if exp := 12; len(b) != exp {
		t.Fatalf("length mismatch: got %v, exp %v", len(b), exp)
	}

	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected uncompressed, got %v", got)
	}

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec TimeDecoder
	dec.Init(b)
	for i, v := range ts {
		if !dec.Next() {
			t.Fatalf("Next == false, expected true")
		}

		if v != dec.Read() {
			t.Fatalf("Item %d mismatch, got %v, exp %v", i, dec.Read(), v)
		}
	}

	if dec.Next() {
		t.Fatalf("unexpected extra values")
	}
}

func Test_TimeEncoder_Reverse(t *testing.T) {
	enc := NewTimeEncoder(3)
	ts := []int64{
		int64(3),
		int64(2),
		int64(0),
	}

	for _, v := range ts {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeUncompressed {
		t.Fatalf("Wrong encoding used: expected uncompressed, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	i := 0
	for dec.Next() {
		if ts[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), ts[i])
		}
		i++
	}
}

func Test_TimeEncoder_220SecondDelta(t *testing.T) {
	enc := NewTimeEncoder(256)
	var ts []int64
	now := time.Now()
	for i := 0; i < 220; i++ {
		ts = append(ts, now.Add(time.Duration(i*60)*time.Second).UnixNano())
	}

	for _, v := range ts {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Using RLE, should get 12 bytes
	if exp := 12; len(b) != exp {
		t.Fatalf("unexpected length: got %v, exp %v", len(b), exp)
	}

	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected uncompressed, got %v", got)
	}

	var dec TimeDecoder
	dec.Init(b)
	i := 0
	for dec.Next() {
		if ts[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), ts[i])
		}
		i++
	}

	if i != len(ts) {
		t.Fatalf("Read too few values: exp %d, got %d", len(ts), i)
	}

	if dec.Next() {
		t.Fatalf("expecte Next() = false, got true")
	}
}

func Test_TimeEncoder_Quick(t *testing.T) {
	quick.Check(func(values []int64) bool {
		// Write values to encoder.
		enc := NewTimeEncoder(1024)
		exp := make([]int64, len(values))
		for i, v := range values {
			exp[i] = int64(v)
			enc.Write(exp[i])
		}

		// Retrieve encoded bytes from encoder.
		buf, err := enc.Bytes()
		if err != nil {
			t.Fatal(err)
		}

		// Read values out of decoder.
		got := make([]int64, 0, len(values))
		var dec TimeDecoder
		dec.Init(buf)
		for dec.Next() {
			if err := dec.Error(); err != nil {
				t.Fatal(err)
			}
			got = append(got, dec.Read())
		}

		// Verify that input and output values match.
		if !reflect.DeepEqual(exp, got) {
			t.Fatalf("mismatch:\n\nexp=%+v\n\ngot=%+v\n\n", exp, got)
		}

		return true
	}, nil)
}

func Test_TimeEncoder_RLESeconds(t *testing.T) {
	enc := NewTimeEncoder(6)
	ts := make([]int64, 6)

	ts[0] = int64(1444448158000000000)
	ts[1] = int64(1444448168000000000)
	ts[2] = int64(1444448178000000000)
	ts[3] = int64(1444448188000000000)
	ts[4] = int64(1444448198000000000)
	ts[5] = int64(1444448208000000000)

	for _, v := range ts {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var dec TimeDecoder
	dec.Init(b)
	for i, v := range ts {
		if !dec.Next() {
			t.Fatalf("Next == false, expected true")
		}

		if v != dec.Read() {
			t.Fatalf("Item %d mismatch, got %v, exp %v", i, dec.Read(), v)
		}
	}

	if dec.Next() {
		t.Fatalf("unexpected extra values")
	}
}

func TestTimeEncoder_Count_Uncompressed(t *testing.T) {
	enc := NewTimeEncoder(2)
	t1 := time.Unix(0, 0).UnixNano()
	t2 := time.Unix(1, 0).UnixNano()

	// about 36.5yrs in NS resolution is max range for compressed format
	// This should cause the encoding to fallback to raw points
	t3 := time.Unix(2, (2 << 59)).UnixNano()
	enc.Write(t1)
	enc.Write(t2)
	enc.Write(t3)

	b, err := enc.Bytes()
	if got := b[0] >> 4; got != timeUncompressed {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got, exp := CountTimestamps(b), 3; got != exp {
		t.Fatalf("count mismatch: got %v, exp %v", got, exp)
	}
}

func TestTimeEncoder_Count_RLE(t *testing.T) {
	enc := NewTimeEncoder(5)
	ts := make([]int64, 6)

	ts[0] = int64(1444448158000000000)
	ts[1] = int64(1444448168000000000)
	ts[2] = int64(1444448178000000000)
	ts[3] = int64(1444448188000000000)
	ts[4] = int64(1444448198000000000)
	ts[5] = int64(1444448208000000000)

	for _, v := range ts {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if got := b[0] >> 4; got != timeCompressedRLE {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got, exp := CountTimestamps(b), len(ts); got != exp {
		t.Fatalf("count mismatch: got %v, exp %v", got, exp)
	}
}

func TestTimeEncoder_Count_Simple8(t *testing.T) {
	enc := NewTimeEncoder(3)
	t1 := int64(0)
	t2 := int64(1)
	t3 := int64(3)

	enc.Write(t1)
	enc.Write(t2)
	enc.Write(t3)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; got != timeCompressedPackedSimple {
		t.Fatalf("Wrong encoding used: expected rle, got %v", got)
	}

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got, exp := CountTimestamps(b), 3; got != exp {
		t.Fatalf("count mismatch: got %v, exp %v", got, exp)
	}
}

func TestTimeDecoder_Corrupt(t *testing.T) {
	cases := []string{
		"",                 // Empty
		"\x10\x14",         // Packed: not enough data
		"\x20\x00",         // RLE: not enough data for starting timestamp
		"\x2012345678\x90", // RLE: initial timestamp but invalid uvarint encoding
		"\x2012345678\x7f", // RLE: timestamp, RLE but invalid repeat
		"\x00123",          // Raw: data length not multiple of 8
	}

	for _, c := range cases {
		var dec TimeDecoder
		dec.Init([]byte(c))
		if dec.Next() {
			t.Fatalf("exp next == false, got true")
		}
	}
}

func BenchmarkTimeEncoder(b *testing.B) {
	enc := NewTimeEncoder(1024)
	x := make([]int64, 1024)
	for i := 0; i < len(x); i++ {
		x[i] = time.Now().UnixNano()
		enc.Write(x[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Bytes()
	}
}

func BenchmarkTimeDecoder_Packed(b *testing.B) {
	x := make([]int64, 1024)
	enc := NewTimeEncoder(1024)
	for i := 0; i < len(x); i++ {
		x[i] = time.Now().UnixNano()
		enc.Write(x[i])
	}
	bytes, _ := enc.Bytes()

	b.ResetTimer()

	b.StopTimer()
	var dec TimeDecoder
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		dec.Init(bytes)
		for dec.Next() {
		}
	}
}

func BenchmarkTimeDecoder_RLE(b *testing.B) {
	x := make([]int64, 1024)
	enc := NewTimeEncoder(1024)
	for i := 0; i < len(x); i++ {
		x[i] = int64(i * 10)
		enc.Write(x[i])
	}
	bytes, _ := enc.Bytes()

	b.ResetTimer()

	b.StopTimer()
	var dec TimeDecoder
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		dec.Init(bytes)
		for dec.Next() {
		}
	}
}
