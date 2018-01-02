package tsm1

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

func Test_IntegerEncoder_NoValues(t *testing.T) {
	enc := NewIntegerEncoder(0)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(b) > 0 {
		t.Fatalf("unexpected lenght: exp 0, got %v", len(b))
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}
}

func Test_IntegerEncoder_One(t *testing.T) {
	enc := NewIntegerEncoder(1)
	v1 := int64(1)

	enc.Write(1)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; intCompressedSimple != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v1)
	}
}

func Test_IntegerEncoder_Two(t *testing.T) {
	enc := NewIntegerEncoder(2)
	var v1, v2 int64 = 1, 2

	enc.Write(v1)
	enc.Write(v2)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; intCompressedSimple != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v2)
	}
}

func Test_IntegerEncoder_Negative(t *testing.T) {
	enc := NewIntegerEncoder(3)
	var v1, v2, v3 int64 = -2, 0, 1

	enc.Write(v1)
	enc.Write(v2)
	enc.Write(v3)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; intCompressedSimple != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v2)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v3 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v3)
	}
}

func Test_IntegerEncoder_Large_Range(t *testing.T) {
	enc := NewIntegerEncoder(2)
	var v1, v2 int64 = math.MinInt64, math.MaxInt64
	enc.Write(v1)
	enc.Write(v2)
	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; intUncompressed != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v2)
	}
}

func Test_IntegerEncoder_Uncompressed(t *testing.T) {
	enc := NewIntegerEncoder(3)
	var v1, v2, v3 int64 = 0, 1, 1 << 60

	enc.Write(v1)
	enc.Write(v2)
	enc.Write(v3)

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("expected error: %v", err)
	}

	// 1 byte header + 3 * 8 byte values
	if exp := 25; len(b) != exp {
		t.Fatalf("length mismatch: got %v, exp %v", len(b), exp)
	}

	if got := b[0] >> 4; intUncompressed != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v1 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v1)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v2 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v2)
	}

	if !dec.Next() {
		t.Fatalf("unexpected next value: got true, exp false")
	}

	if v3 != dec.Read() {
		t.Fatalf("read value mismatch: got %v, exp %v", dec.Read(), v3)
	}
}

func Test_IntegerEncoder_NegativeUncompressed(t *testing.T) {
	values := []int64{
		-2352281900722994752, 1438442655375607923, -4110452567888190110,
		-1221292455668011702, -1941700286034261841, -2836753127140407751,
		1432686216250034552, 3663244026151507025, -3068113732684750258,
		-1949953187327444488, 3713374280993588804, 3226153669854871355,
		-2093273755080502606, 1006087192578600616, -2272122301622271655,
		2533238229511593671, -4450454445568858273, 2647789901083530435,
		2761419461769776844, -1324397441074946198, -680758138988210958,
		94468846694902125, -2394093124890745254, -2682139311758778198,
	}
	enc := NewIntegerEncoder(256)
	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("expected error: %v", err)
	}

	if got := b[0] >> 4; intUncompressed != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)

	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_AllNegative(t *testing.T) {
	enc := NewIntegerEncoder(3)
	values := []int64{
		-10, -5, -1,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := b[0] >> 4; intCompressedSimple != got {
		t.Fatalf("encoding type mismatch: exp uncompressed, got %v", got)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_CounterPacked(t *testing.T) {
	enc := NewIntegerEncoder(16)
	values := []int64{
		1e15, 1e15 + 1, 1e15 + 2, 1e15 + 3, 1e15 + 4, 1e15 + 6,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != intCompressedSimple {
		t.Fatalf("unexpected encoding format: expected simple, got %v", b[0]>>4)
	}

	// Should use 1 header byte + 2, 8 byte words if delta-encoding is used based on
	// values sizes.  Without delta-encoding, we'd get 49 bytes.
	if exp := 17; len(b) != exp {
		t.Fatalf("encoded length mismatch: got %v, exp %v", len(b), exp)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_CounterRLE(t *testing.T) {
	enc := NewIntegerEncoder(16)
	values := []int64{
		1e15, 1e15 + 1, 1e15 + 2, 1e15 + 3, 1e15 + 4, 1e15 + 5,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != intCompressedRLE {
		t.Fatalf("unexpected encoding format: expected RLE, got %v", b[0]>>4)
	}

	// Should use 1 header byte, 8 byte first value, 1 var-byte for delta and 1 var-byte for
	// count of deltas in this particular RLE.
	if exp := 11; len(b) != exp {
		t.Fatalf("encoded length mismatch: got %v, exp %v", len(b), exp)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_Descending(t *testing.T) {
	enc := NewIntegerEncoder(16)
	values := []int64{
		7094, 4472, 1850,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != intCompressedRLE {
		t.Fatalf("unexpected encoding format: expected simple, got %v", b[0]>>4)
	}

	// Should use 1 header byte, 8 byte first value, 1 var-byte for delta and 1 var-byte for
	// count of deltas in this particular RLE.
	if exp := 12; len(b) != exp {
		t.Fatalf("encoded length mismatch: got %v, exp %v", len(b), exp)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_Flat(t *testing.T) {
	enc := NewIntegerEncoder(16)
	values := []int64{
		1, 1, 1, 1,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != intCompressedRLE {
		t.Fatalf("unexpected encoding format: expected simple, got %v", b[0]>>4)
	}

	// Should use 1 header byte, 8 byte first value, 1 var-byte for delta and 1 var-byte for
	// count of deltas in this particular RLE.
	if exp := 11; len(b) != exp {
		t.Fatalf("encoded length mismatch: got %v, exp %v", len(b), exp)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_MinMax(t *testing.T) {
	enc := NewIntegerEncoder(2)
	values := []int64{
		math.MinInt64, math.MaxInt64,
	}

	for _, v := range values {
		enc.Write(v)
	}

	b, err := enc.Bytes()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if b[0]>>4 != intUncompressed {
		t.Fatalf("unexpected encoding format: expected simple, got %v", b[0]>>4)
	}

	if exp := 17; len(b) != exp {
		t.Fatalf("encoded length mismatch: got %v, exp %v", len(b), exp)
	}

	var dec IntegerDecoder
	dec.SetBytes(b)
	i := 0
	for dec.Next() {
		if i > len(values) {
			t.Fatalf("read too many values: got %v, exp %v", i, len(values))
		}

		if values[i] != dec.Read() {
			t.Fatalf("read value %d mismatch: got %v, exp %v", i, dec.Read(), values[i])
		}
		i += 1
	}

	if i != len(values) {
		t.Fatalf("failed to read enough values: got %v, exp %v", i, len(values))
	}
}

func Test_IntegerEncoder_Quick(t *testing.T) {
	quick.Check(func(values []int64) bool {
		expected := values
		if values == nil {
			expected = []int64{} // is this really expected?
		}

		// Write values to encoder.
		enc := NewIntegerEncoder(1024)
		for _, v := range values {
			enc.Write(v)
		}

		// Retrieve encoded bytes from encoder.
		buf, err := enc.Bytes()
		if err != nil {
			t.Fatal(err)
		}

		// Read values out of decoder.
		got := make([]int64, 0, len(values))
		var dec IntegerDecoder
		dec.SetBytes(buf)
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

func Test_IntegerDecoder_Corrupt(t *testing.T) {
	cases := []string{
		"",                     // Empty
		"\x00abc",              // Uncompressed: less than 8 bytes
		"\x10abc",              // Packed: less than 8 bytes
		"\x20abc",              // RLE: less than 8 bytes
		"\x2012345678\x90",     // RLE: valid starting value but invalid delta value
		"\x2012345678\x01\x90", // RLE: valid starting, valid delta value, invalid repeat value
	}

	for _, c := range cases {
		var dec IntegerDecoder
		dec.SetBytes([]byte(c))
		if dec.Next() {
			t.Fatalf("exp next == false, got true")
		}
	}
}

func BenchmarkIntegerEncoderRLE(b *testing.B) {
	enc := NewIntegerEncoder(1024)
	x := make([]int64, 1024)
	for i := 0; i < len(x); i++ {
		x[i] = int64(i)
		enc.Write(x[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Bytes()
	}
}

func BenchmarkIntegerEncoderPackedSimple(b *testing.B) {
	enc := NewIntegerEncoder(1024)
	x := make([]int64, 1024)
	for i := 0; i < len(x); i++ {
		// Small amount of randomness prevents RLE from being used
		x[i] = int64(i) + int64(rand.Intn(10))
		enc.Write(x[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Bytes()
	}
}

type byteSetter interface {
	SetBytes(b []byte)
}

func BenchmarkIntegerDecoderPackedSimple(b *testing.B) {
	x := make([]int64, 1024)
	enc := NewIntegerEncoder(1024)
	for i := 0; i < len(x); i++ {
		// Small amount of randomness prevents RLE from being used
		x[i] = int64(i) + int64(rand.Intn(10))
		enc.Write(x[i])
	}
	bytes, _ := enc.Bytes()

	b.ResetTimer()

	var dec IntegerDecoder
	for i := 0; i < b.N; i++ {
		dec.SetBytes(bytes)
		for dec.Next() {
		}
	}
}

func BenchmarkIntegerDecoderRLE(b *testing.B) {
	x := make([]int64, 1024)
	enc := NewIntegerEncoder(1024)
	for i := 0; i < len(x); i++ {
		x[i] = int64(i)
		enc.Write(x[i])
	}
	bytes, _ := enc.Bytes()

	b.ResetTimer()

	var dec IntegerDecoder
	dec.SetBytes(bytes)

	for i := 0; i < b.N; i++ {
		dec.SetBytes(bytes)
		for dec.Next() {
		}
	}
}
