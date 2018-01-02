package tsm1_test

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

func TestEncoding_FloatBlock(t *testing.T) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, float64(i))
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %s\n\texp: %s\n", spew.Sdump(decodedValues), spew.Sdump(values))
	}
}

func TestEncoding_FloatBlock_ZeroTime(t *testing.T) {
	values := make([]tsm1.Value, 3)
	for i := 0; i < 3; i++ {
		values[i] = tsm1.NewValue(0, float64(i))
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues, values)
	}
}

func TestEncoding_FloatBlock_SimilarFloats(t *testing.T) {
	values := make([]tsm1.Value, 5)
	values[0] = tsm1.NewValue(1444238178437870000, 6.00065e+06)
	values[1] = tsm1.NewValue(1444238185286830000, 6.000656e+06)
	values[2] = tsm1.NewValue(1444238188441501000, 6.000657e+06)
	values[3] = tsm1.NewValue(1444238195286811000, 6.000659e+06)
	values[4] = tsm1.NewValue(1444238198439917000, 6.000661e+06)

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues, values)
	}
}

func TestEncoding_IntBlock_Basic(t *testing.T) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, int64(i))
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if len(decodedValues) != len(values) {
		t.Fatalf("unexpected results length:\n\tgot: %v\n\texp: %v\n", len(decodedValues), len(values))
	}

	for i := 0; i < len(decodedValues); i++ {
		if decodedValues[i].UnixNano() != values[i].UnixNano() {
			t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues[i].UnixNano(), values[i].UnixNano())
		}

		if decodedValues[i].Value() != values[i].Value() {
			t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues[i].Value(), values[i].Value())
		}
	}
}

func TestEncoding_IntBlock_Negatives(t *testing.T) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		v := int64(i)
		if i%2 == 0 {
			v = -v
		}
		values[i] = tsm1.NewValue(t, int64(v))
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues, values)
	}
}

func TestEncoding_BooleanBlock_Basic(t *testing.T) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		v := true
		if i%2 == 0 {
			v = false
		}
		values[i] = tsm1.NewValue(t, v)
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues, values)
	}
}

func TestEncoding_StringBlock_Basic(t *testing.T) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, fmt.Sprintf("value %d", i))
	}

	b, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	decodedValues, err = tsm1.DecodeBlock(b, decodedValues)
	if err != nil {
		t.Fatalf("unexpected error decoding block: %v", err)
	}

	if !reflect.DeepEqual(decodedValues, values) {
		t.Fatalf("unexpected results:\n\tgot: %v\n\texp: %v\n", decodedValues, values)
	}
}

func TestEncoding_BlockType(t *testing.T) {
	tests := []struct {
		value     interface{}
		blockType byte
	}{
		{value: float64(1.0), blockType: tsm1.BlockFloat64},
		{value: int64(1), blockType: tsm1.BlockInteger},
		{value: true, blockType: tsm1.BlockBoolean},
		{value: "string", blockType: tsm1.BlockString},
	}

	for _, test := range tests {
		var values []tsm1.Value
		values = append(values, tsm1.NewValue(0, test.value))

		b, err := tsm1.Values(values).Encode(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		bt, err := tsm1.BlockType(b)
		if err != nil {
			t.Fatalf("unexpected error decoding block type: %v", err)
		}

		if got, exp := bt, test.blockType; got != exp {
			t.Fatalf("block type mismatch: got %v, exp %v", got, exp)
		}
	}

	_, err := tsm1.BlockType([]byte{10})
	if err == nil {
		t.Fatalf("expected error decoding block type, got nil")
	}
}

func TestEncoding_Count(t *testing.T) {
	tests := []struct {
		value     interface{}
		blockType byte
	}{
		{value: float64(1.0), blockType: tsm1.BlockFloat64},
		{value: int64(1), blockType: tsm1.BlockInteger},
		{value: true, blockType: tsm1.BlockBoolean},
		{value: "string", blockType: tsm1.BlockString},
	}

	for _, test := range tests {
		var values []tsm1.Value
		values = append(values, tsm1.NewValue(0, test.value))

		b, err := tsm1.Values(values).Encode(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if got, exp := tsm1.BlockCount(b), 1; got != exp {
			t.Fatalf("block count mismatch: got %v, exp %v", got, exp)
		}
	}
}

func TestValues_MergeFloat(t *testing.T) {
	tests := []struct {
		a, b, exp []tsm1.Value
	}{

		{ // empty a
			a: []tsm1.Value{},

			b: []tsm1.Value{
				tsm1.NewValue(1, 1.2),
				tsm1.NewValue(2, 2.2),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.2),
				tsm1.NewValue(2, 2.2),
			},
		},
		{ // empty b
			a: []tsm1.Value{
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
			},

			b: []tsm1.Value{},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
			},
			b: []tsm1.Value{
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(2, 2.2), // duplicate data
			},
			exp: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.2),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(1, 1.1), // duplicate data
				tsm1.NewValue(2, 2.1),
			},
			b: []tsm1.Value{
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(2, 2.2), // duplicate data
			},
			exp: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.2),
			},
		},

		{
			a: []tsm1.Value{
				tsm1.NewValue(1, 1.1),
			},
			b: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.2), // overwrites a
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(3, 3.2),
				tsm1.NewValue(4, 4.2),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.2),
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(3, 3.2),
				tsm1.NewValue(4, 4.2),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
				tsm1.NewValue(3, 3.1),
				tsm1.NewValue(4, 4.1),
			},

			b: []tsm1.Value{
				tsm1.NewValue(1, 1.2), // overwrites a
				tsm1.NewValue(2, 2.2), // overwrites a
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.2),
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(3, 3.1),
				tsm1.NewValue(4, 4.1),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
				tsm1.NewValue(3, 3.1),
				tsm1.NewValue(4, 4.1),
			},

			b: []tsm1.Value{
				tsm1.NewValue(1, 1.2), // overwrites a
				tsm1.NewValue(2, 2.2), // overwrites a
				tsm1.NewValue(3, 3.2),
				tsm1.NewValue(4, 4.2),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.2),
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(3, 3.2),
				tsm1.NewValue(4, 4.2),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.1),
				tsm1.NewValue(3, 3.1),
				tsm1.NewValue(4, 4.1),
			},
			b: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(4, 4.2),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(0, 0.0),
				tsm1.NewValue(1, 1.1),
				tsm1.NewValue(2, 2.2),
				tsm1.NewValue(3, 3.1),
				tsm1.NewValue(4, 4.2),
			},
		},

		{
			a: []tsm1.Value{
				tsm1.NewValue(1462498658242869207, 0.0),
				tsm1.NewValue(1462498658288956853, 1.1),
			},
			b: []tsm1.Value{
				tsm1.NewValue(1462498658242870810, 0.0),
				tsm1.NewValue(1462498658262911238, 2.2),
				tsm1.NewValue(1462498658282415038, 4.2),
				tsm1.NewValue(1462498658282417760, 4.2),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1462498658242869207, 0.0),
				tsm1.NewValue(1462498658242870810, 0.0),
				tsm1.NewValue(1462498658262911238, 2.2),
				tsm1.NewValue(1462498658282415038, 4.2),
				tsm1.NewValue(1462498658282417760, 4.2),
				tsm1.NewValue(1462498658288956853, 1.1),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
			},
			b: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
			},
			b: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(7, 7.0),
				tsm1.NewValue(8, 8.0),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
				tsm1.NewValue(7, 7.0),
				tsm1.NewValue(8, 8.0),
			},
		},
		{
			a: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
			},
			b: []tsm1.Value{
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
			},
			exp: []tsm1.Value{
				tsm1.NewValue(1, 1.0),
				tsm1.NewValue(2, 2.0),
				tsm1.NewValue(3, 3.0),
				tsm1.NewValue(4, 4.0),
				tsm1.NewValue(5, 5.0),
				tsm1.NewValue(6, 6.0),
			},
		},
	}

	for i, test := range tests {
		got := tsm1.Values(test.a).Merge(test.b)

		if exp, got := len(test.exp), len(got); exp != got {
			t.Fatalf("test(%d): value length mismatch: exp %v, got %v", i, exp, got)
		}

		for i := range test.exp {
			if exp, got := test.exp[i].String(), got[i].String(); exp != got {
				t.Fatalf("value mismatch:\n exp %v\n got %v", exp, got)
			}
		}
	}
}

func TestIntegerValues_Merge(t *testing.T) {
	integerValue := func(t int64, f int64) tsm1.IntegerValue {
		return *(tsm1.NewValue(t, f).(*tsm1.IntegerValue))
	}

	tests := []struct {
		a, b, exp []tsm1.IntegerValue
	}{

		{ // empty a
			a: []tsm1.IntegerValue{},

			b: []tsm1.IntegerValue{
				integerValue(1, 10),
				integerValue(2, 20),
			},
			exp: []tsm1.IntegerValue{
				integerValue(1, 10),
				integerValue(2, 20),
			},
		},
		{ // empty b
			a: []tsm1.IntegerValue{
				integerValue(1, 1),
				integerValue(2, 2),
			},

			b: []tsm1.IntegerValue{},
			exp: []tsm1.IntegerValue{
				integerValue(1, 1),
				integerValue(2, 2),
			},
		},
		{
			a: []tsm1.IntegerValue{
				integerValue(1, 1),
			},
			b: []tsm1.IntegerValue{
				integerValue(0, 0),
				integerValue(1, 10), // overwrites a
				integerValue(2, 20),
				integerValue(3, 30),
				integerValue(4, 40),
			},
			exp: []tsm1.IntegerValue{
				integerValue(0, 0),
				integerValue(1, 10),
				integerValue(2, 20),
				integerValue(3, 30),
				integerValue(4, 40),
			},
		},
		{
			a: []tsm1.IntegerValue{
				integerValue(1, 1),
				integerValue(2, 2),
				integerValue(3, 3),
				integerValue(4, 4),
			},

			b: []tsm1.IntegerValue{
				integerValue(1, 10), // overwrites a
				integerValue(2, 20), // overwrites a
			},
			exp: []tsm1.IntegerValue{
				integerValue(1, 10),
				integerValue(2, 20),
				integerValue(3, 3),
				integerValue(4, 4),
			},
		},
		{
			a: []tsm1.IntegerValue{
				integerValue(1, 1),
				integerValue(2, 2),
				integerValue(3, 3),
				integerValue(4, 4),
			},

			b: []tsm1.IntegerValue{
				integerValue(1, 10), // overwrites a
				integerValue(2, 20), // overwrites a
				integerValue(3, 30),
				integerValue(4, 40),
			},
			exp: []tsm1.IntegerValue{
				integerValue(1, 10),
				integerValue(2, 20),
				integerValue(3, 30),
				integerValue(4, 40),
			},
		},
		{
			a: []tsm1.IntegerValue{
				integerValue(0, 0),
				integerValue(1, 1),
				integerValue(2, 2),
				integerValue(3, 3),
				integerValue(4, 4),
			},
			b: []tsm1.IntegerValue{
				integerValue(0, 0),
				integerValue(2, 20),
				integerValue(4, 40),
			},
			exp: []tsm1.IntegerValue{
				integerValue(0, 0.0),
				integerValue(1, 1),
				integerValue(2, 20),
				integerValue(3, 3),
				integerValue(4, 40),
			},
		},
	}

	for i, test := range tests {
		if i != 2 {
			continue
		}

		got := tsm1.IntegerValues(test.a).Merge(test.b)
		if exp, got := len(test.exp), len(got); exp != got {
			t.Fatalf("test(%d): value length mismatch: exp %v, got %v", i, exp, got)
		}

		for i := range test.exp {
			if exp, got := test.exp[i].String(), got[i].String(); exp != got {
				t.Fatalf("value mismatch:\n exp %v\n got %v", exp, got)
			}
		}
	}
}

func TestFloatValues_Merge(t *testing.T) {
	floatValue := func(t int64, f float64) tsm1.FloatValue {
		return *(tsm1.NewValue(t, f).(*tsm1.FloatValue))
	}

	tests := []struct {
		a, b, exp []tsm1.FloatValue
	}{

		{ // empty a
			a: []tsm1.FloatValue{},

			b: []tsm1.FloatValue{
				floatValue(1, 1.2),
				floatValue(2, 2.2),
			},
			exp: []tsm1.FloatValue{
				floatValue(1, 1.2),
				floatValue(2, 2.2),
			},
		},
		{ // empty b
			a: []tsm1.FloatValue{
				floatValue(1, 1.1),
				floatValue(2, 2.1),
			},

			b: []tsm1.FloatValue{},
			exp: []tsm1.FloatValue{
				floatValue(1, 1.1),
				floatValue(2, 2.1),
			},
		},
		{
			a: []tsm1.FloatValue{
				floatValue(1, 1.1),
			},
			b: []tsm1.FloatValue{
				floatValue(0, 0.0),
				floatValue(1, 1.2), // overwrites a
				floatValue(2, 2.2),
				floatValue(3, 3.2),
				floatValue(4, 4.2),
			},
			exp: []tsm1.FloatValue{
				floatValue(0, 0.0),
				floatValue(1, 1.2),
				floatValue(2, 2.2),
				floatValue(3, 3.2),
				floatValue(4, 4.2),
			},
		},
		{
			a: []tsm1.FloatValue{
				floatValue(1, 1.1),
				floatValue(2, 2.1),
				floatValue(3, 3.1),
				floatValue(4, 4.1),
			},

			b: []tsm1.FloatValue{
				floatValue(1, 1.2), // overwrites a
				floatValue(2, 2.2), // overwrites a
			},
			exp: []tsm1.FloatValue{
				floatValue(1, 1.2),
				floatValue(2, 2.2),
				floatValue(3, 3.1),
				floatValue(4, 4.1),
			},
		},
		{
			a: []tsm1.FloatValue{
				floatValue(1, 1.1),
				floatValue(2, 2.1),
				floatValue(3, 3.1),
				floatValue(4, 4.1),
			},

			b: []tsm1.FloatValue{
				floatValue(1, 1.2), // overwrites a
				floatValue(2, 2.2), // overwrites a
				floatValue(3, 3.2),
				floatValue(4, 4.2),
			},
			exp: []tsm1.FloatValue{
				floatValue(1, 1.2),
				floatValue(2, 2.2),
				floatValue(3, 3.2),
				floatValue(4, 4.2),
			},
		},
		{
			a: []tsm1.FloatValue{
				floatValue(0, 0.0),
				floatValue(1, 1.1),
				floatValue(2, 2.1),
				floatValue(3, 3.1),
				floatValue(4, 4.1),
			},
			b: []tsm1.FloatValue{
				floatValue(0, 0.0),
				floatValue(2, 2.2),
				floatValue(4, 4.2),
			},
			exp: []tsm1.FloatValue{
				floatValue(0, 0.0),
				floatValue(1, 1.1),
				floatValue(2, 2.2),
				floatValue(3, 3.1),
				floatValue(4, 4.2),
			},
		},
	}

	for i, test := range tests {
		got := tsm1.FloatValues(test.a).Merge(test.b)
		if exp, got := len(test.exp), len(got); exp != got {
			t.Fatalf("test(%d): value length mismatch: exp %v, got %v", i, exp, got)
		}

		for i := range test.exp {
			if exp, got := test.exp[i].String(), got[i].String(); exp != got {
				t.Fatalf("value mismatch:\n exp %v\n got %v", exp, got)
			}
		}
	}
}

func TestBooleanValues_Merge(t *testing.T) {
	booleanValue := func(t int64, f bool) tsm1.BooleanValue {
		return *(tsm1.NewValue(t, f).(*tsm1.BooleanValue))
	}

	tests := []struct {
		a, b, exp []tsm1.BooleanValue
	}{

		{ // empty a
			a: []tsm1.BooleanValue{},

			b: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
			},
			exp: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
			},
		},
		{ // empty b
			a: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
			},

			b: []tsm1.BooleanValue{},
			exp: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
			},
		},
		{
			a: []tsm1.BooleanValue{
				booleanValue(1, true),
			},
			b: []tsm1.BooleanValue{
				booleanValue(0, false),
				booleanValue(1, false), // overwrites a
				booleanValue(2, false),
				booleanValue(3, false),
				booleanValue(4, false),
			},
			exp: []tsm1.BooleanValue{
				booleanValue(0, false),
				booleanValue(1, false),
				booleanValue(2, false),
				booleanValue(3, false),
				booleanValue(4, false),
			},
		},
		{
			a: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
				booleanValue(3, true),
				booleanValue(4, true),
			},

			b: []tsm1.BooleanValue{
				booleanValue(1, false), // overwrites a
				booleanValue(2, false), // overwrites a
			},
			exp: []tsm1.BooleanValue{
				booleanValue(1, false), // overwrites a
				booleanValue(2, false), // overwrites a
				booleanValue(3, true),
				booleanValue(4, true),
			},
		},
		{
			a: []tsm1.BooleanValue{
				booleanValue(1, true),
				booleanValue(2, true),
				booleanValue(3, true),
				booleanValue(4, true),
			},

			b: []tsm1.BooleanValue{
				booleanValue(1, false), // overwrites a
				booleanValue(2, false), // overwrites a
				booleanValue(3, false),
				booleanValue(4, false),
			},
			exp: []tsm1.BooleanValue{
				booleanValue(1, false),
				booleanValue(2, false),
				booleanValue(3, false),
				booleanValue(4, false),
			},
		},
		{
			a: []tsm1.BooleanValue{
				booleanValue(0, true),
				booleanValue(1, true),
				booleanValue(2, true),
				booleanValue(3, true),
				booleanValue(4, true),
			},
			b: []tsm1.BooleanValue{
				booleanValue(0, false),
				booleanValue(2, false),
				booleanValue(4, false),
			},
			exp: []tsm1.BooleanValue{
				booleanValue(0, false),
				booleanValue(1, true),
				booleanValue(2, false),
				booleanValue(3, true),
				booleanValue(4, false),
			},
		},
	}

	for i, test := range tests {
		got := tsm1.BooleanValues(test.a).Merge(test.b)
		if exp, got := len(test.exp), len(got); exp != got {
			t.Fatalf("test(%d): value length mismatch: exp %v, got %v", i, exp, got)
		}

		for i := range test.exp {
			if exp, got := test.exp[i].String(), got[i].String(); exp != got {
				t.Fatalf("value mismatch:\n exp %v\n got %v", exp, got)
			}
		}
	}
}

func TestStringValues_Merge(t *testing.T) {
	stringValue := func(t int64, f string) tsm1.StringValue {
		return *(tsm1.NewValue(t, f).(*tsm1.StringValue))
	}

	tests := []struct {
		a, b, exp []tsm1.StringValue
	}{

		{ // empty a
			a: []tsm1.StringValue{},

			b: []tsm1.StringValue{
				stringValue(1, "10"),
				stringValue(2, "20"),
			},
			exp: []tsm1.StringValue{
				stringValue(1, "10"),
				stringValue(2, "20"),
			},
		},
		{ // empty b
			a: []tsm1.StringValue{
				stringValue(1, "1"),
				stringValue(2, "2"),
			},

			b: []tsm1.StringValue{},
			exp: []tsm1.StringValue{
				stringValue(1, "1"),
				stringValue(2, "2"),
			},
		},
		{
			a: []tsm1.StringValue{
				stringValue(1, "1"),
			},
			b: []tsm1.StringValue{
				stringValue(0, "0"),
				stringValue(1, "10"), // overwrites a
				stringValue(2, "20"),
				stringValue(3, "30"),
				stringValue(4, "40"),
			},
			exp: []tsm1.StringValue{
				stringValue(0, "0"),
				stringValue(1, "10"),
				stringValue(2, "20"),
				stringValue(3, "30"),
				stringValue(4, "40"),
			},
		},
		{
			a: []tsm1.StringValue{
				stringValue(1, "1"),
				stringValue(2, "2"),
				stringValue(3, "3"),
				stringValue(4, "4"),
			},

			b: []tsm1.StringValue{
				stringValue(1, "10"), // overwrites a
				stringValue(2, "20"), // overwrites a
			},
			exp: []tsm1.StringValue{
				stringValue(1, "10"),
				stringValue(2, "20"),
				stringValue(3, "3"),
				stringValue(4, "4"),
			},
		},
		{
			a: []tsm1.StringValue{
				stringValue(1, "1"),
				stringValue(2, "2"),
				stringValue(3, "3"),
				stringValue(4, "4"),
			},

			b: []tsm1.StringValue{
				stringValue(1, "10"), // overwrites a
				stringValue(2, "20"), // overwrites a
				stringValue(3, "30"),
				stringValue(4, "40"),
			},
			exp: []tsm1.StringValue{
				stringValue(1, "10"),
				stringValue(2, "20"),
				stringValue(3, "30"),
				stringValue(4, "40"),
			},
		},
		{
			a: []tsm1.StringValue{
				stringValue(0, "0"),
				stringValue(1, "1"),
				stringValue(2, "2"),
				stringValue(3, "3"),
				stringValue(4, "4"),
			},
			b: []tsm1.StringValue{
				stringValue(0, "0"),
				stringValue(2, "20"),
				stringValue(4, "40"),
			},
			exp: []tsm1.StringValue{
				stringValue(0, "0.0"),
				stringValue(1, "1"),
				stringValue(2, "20"),
				stringValue(3, "3"),
				stringValue(4, "40"),
			},
		},
	}

	for i, test := range tests {
		if i != 2 {
			continue
		}

		got := tsm1.StringValues(test.a).Merge(test.b)
		if exp, got := len(test.exp), len(got); exp != got {
			t.Fatalf("test(%d): value length mismatch: exp %v, got %v", i, exp, got)
		}

		for i := range test.exp {
			if exp, got := test.exp[i].String(), got[i].String(); exp != got {
				t.Fatalf("value mismatch:\n exp %v\n got %v", exp, got)
			}
		}
	}
}
func getTimes(n, step int, precision time.Duration) []int64 {
	t := time.Now().Round(precision).UnixNano()
	a := make([]int64, n)
	for i := 0; i < n; i++ {
		a[i] = t + (time.Duration(i*60) * precision).Nanoseconds()
	}
	return a
}

func BenchmarkDecodeBlock_Float_Empty(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, float64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Float_EqualSize(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, float64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.Value, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Float_TypeSpecific(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, float64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.FloatValue, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeFloatBlock(bytes, &decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Integer_Empty(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, int64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Integer_EqualSize(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, int64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.Value, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Integer_TypeSpecific(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, int64(i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.IntegerValue, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeIntegerBlock(bytes, &decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Boolean_Empty(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, true)
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Boolean_EqualSize(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, true)
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.Value, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_Boolean_TypeSpecific(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, true)
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.BooleanValue, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBooleanBlock(bytes, &decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_String_Empty(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, fmt.Sprintf("value %d", i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	var decodedValues []tsm1.Value
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_String_EqualSize(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, fmt.Sprintf("value %d", i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.Value, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeBlock(bytes, decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkDecodeBlock_String_TypeSpecific(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, fmt.Sprintf("value %d", i))
	}

	bytes, err := tsm1.Values(values).Encode(nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	decodedValues := make([]tsm1.StringValue, len(values))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tsm1.DecodeStringBlock(bytes, &decodedValues)
		if err != nil {
			b.Fatalf("unexpected error decoding block: %v", err)
		}
	}
}

func BenchmarkValues_Deduplicate(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	values := make([]tsm1.Value, len(times))
	for i, t := range times {
		values[i] = tsm1.NewValue(t, fmt.Sprintf("value %d", i))
	}
	values = append(values, values...)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(values).Deduplicate()
	}
}

func BenchmarkValues_Merge(b *testing.B) {
	valueCount := 1000
	times := getTimes(valueCount, 60, time.Second)
	a := make([]tsm1.Value, len(times))
	c := make([]tsm1.Value, len(times))

	for i, t := range times {
		a[i] = tsm1.NewValue(t, float64(i))
		c[i] = tsm1.NewValue(t+1, float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(a).Merge(c)
	}
}

func BenchmarkValues_EncodeInteger(b *testing.B) {
	valueCount := 1024
	times := getTimes(valueCount, 60, time.Second)
	a := make([]tsm1.Value, len(times))

	for i, t := range times {
		a[i] = tsm1.NewValue(t, int64(i))
	}

	buf := make([]byte, 1024*8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(a).Encode(buf)
	}
}

func BenchmarkValues_EncodeFloat(b *testing.B) {
	valueCount := 1024
	times := getTimes(valueCount, 60, time.Second)
	a := make([]tsm1.Value, len(times))

	for i, t := range times {
		a[i] = tsm1.NewValue(t, float64(i))
	}

	buf := make([]byte, 1024*8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(a).Encode(buf)
	}
}
func BenchmarkValues_EncodeString(b *testing.B) {
	valueCount := 1024
	times := getTimes(valueCount, 60, time.Second)
	a := make([]tsm1.Value, len(times))

	for i, t := range times {
		a[i] = tsm1.NewValue(t, fmt.Sprintf("%d", i))
	}

	buf := make([]byte, 1024*8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(a).Encode(buf)
	}
}
func BenchmarkValues_EncodeBool(b *testing.B) {
	valueCount := 1024
	times := getTimes(valueCount, 60, time.Second)
	a := make([]tsm1.Value, len(times))

	for i, t := range times {
		if i%2 == 0 {
			a[i] = tsm1.NewValue(t, true)
		} else {
			a[i] = tsm1.NewValue(t, false)
		}
	}

	buf := make([]byte, 1024*8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tsm1.Values(a).Encode(buf)
	}
}
