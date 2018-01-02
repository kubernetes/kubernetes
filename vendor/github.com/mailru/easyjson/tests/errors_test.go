package tests

import (
	"testing"

	"github.com/mailru/easyjson/jlexer"
)

func TestMultipleErrorsInt(t *testing.T) {
	for i, test := range []struct {
		Data    []byte
		Offsets []int
	}{
		{
			Data:    []byte(`[1, 2, 3, "4", "5"]`),
			Offsets: []int{10, 15},
		},
		{
			Data:    []byte(`[1, {"2":"3"}, 3, "4"]`),
			Offsets: []int{4, 18},
		},
		{
			Data:    []byte(`[1, "2", "3", "4", "5", "6"]`),
			Offsets: []int{4, 9, 14, 19, 24},
		},
		{
			Data:    []byte(`[1, 2, 3, 4, "5"]`),
			Offsets: []int{13},
		},
		{
			Data:    []byte(`[{"1": "2"}]`),
			Offsets: []int{1},
		},
	} {
		l := jlexer.Lexer{
			Data:              test.Data,
			UseMultipleErrors: true,
		}

		var v ErrorIntSlice

		v.UnmarshalEasyJSON(&l)

		errors := l.GetNonFatalErrors()

		if len(errors) != len(test.Offsets) {
			t.Errorf("[%d] TestMultipleErrorsInt(): errornum: want: %d, got %d", i, len(test.Offsets), len(errors))
			return
		}

		for ii, e := range errors {
			if e.Offset != test.Offsets[ii] {
				t.Errorf("[%d] TestMultipleErrorsInt(): offset[%d]: want %d, got %d", i, ii, test.Offsets[ii], e.Offset)
			}
		}
	}
}

func TestMultipleErrorsBool(t *testing.T) {
	for i, test := range []struct {
		Data    []byte
		Offsets []int
	}{
		{
			Data: []byte(`[true, false, true, false]`),
		},
		{
			Data:    []byte(`["test", "value", "lol", "1"]`),
			Offsets: []int{1, 9, 18, 25},
		},
		{
			Data:    []byte(`[true, 42, {"a":"b", "c":"d"}, false]`),
			Offsets: []int{7, 11},
		},
	} {
		l := jlexer.Lexer{
			Data:              test.Data,
			UseMultipleErrors: true,
		}

		var v ErrorBoolSlice
		v.UnmarshalEasyJSON(&l)

		errors := l.GetNonFatalErrors()

		if len(errors) != len(test.Offsets) {
			t.Errorf("[%d] TestMultipleErrorsBool(): errornum: want: %d, got %d", i, len(test.Offsets), len(errors))
			return
		}
		for ii, e := range errors {
			if e.Offset != test.Offsets[ii] {
				t.Errorf("[%d] TestMultipleErrorsBool(): offset[%d]: want %d, got %d", i, ii, test.Offsets[ii], e.Offset)
			}
		}
	}
}

func TestMultipleErrorsUint(t *testing.T) {
	for i, test := range []struct {
		Data    []byte
		Offsets []int
	}{
		{
			Data: []byte(`[42, 42, 42]`),
		},
		{
			Data:    []byte(`[17, "42", 32]`),
			Offsets: []int{5},
		},
		{
			Data:    []byte(`["zz", "zz"]`),
			Offsets: []int{1, 7},
		},
		{
			Data:    []byte(`[{}, 42]`),
			Offsets: []int{1},
		},
	} {
		l := jlexer.Lexer{
			Data:              test.Data,
			UseMultipleErrors: true,
		}

		var v ErrorUintSlice
		v.UnmarshalEasyJSON(&l)

		errors := l.GetNonFatalErrors()

		if len(errors) != len(test.Offsets) {
			t.Errorf("[%d] TestMultipleErrorsUint(): errornum: want: %d, got %d", i, len(test.Offsets), len(errors))
			return
		}
		for ii, e := range errors {
			if e.Offset != test.Offsets[ii] {
				t.Errorf("[%d] TestMultipleErrorsUint(): offset[%d]: want %d, got %d", i, ii, test.Offsets[ii], e.Offset)
			}
		}
	}
}

func TestMultipleErrorsStruct(t *testing.T) {
	for i, test := range []struct {
		Data    []byte
		Offsets []int
	}{
		{
			Data: []byte(`{"string": "test", "slice":[42, 42, 42], "int_slice":[1, 2, 3]}`),
		},
		{
			Data:    []byte(`{"string": {"test": "test"}, "slice":[42, 42, 42], "int_slice":["1", 2, 3]}`),
			Offsets: []int{11, 64},
		},
		{
			Data:    []byte(`{"slice": [42, 42], "string": {"test": "test"}, "int_slice":["1", "2", 3]}`),
			Offsets: []int{30, 61, 66},
		},
		{
			Data:    []byte(`{"string": "test", "slice": {}}`),
			Offsets: []int{28},
		},
		{
			Data:    []byte(`{"slice":5, "string" : "test"}`),
			Offsets: []int{9},
		},
		{
			Data:    []byte(`{"slice" : "test", "string" : "test"}`),
			Offsets: []int{11},
		},
		{
			Data:    []byte(`{"slice": "", "string" : {}, "int":{}}`),
			Offsets: []int{10, 25, 35},
		},
	} {
		l := jlexer.Lexer{
			Data:              test.Data,
			UseMultipleErrors: true,
		}
		var v ErrorStruct
		v.UnmarshalEasyJSON(&l)

		errors := l.GetNonFatalErrors()

		if len(errors) != len(test.Offsets) {
			t.Errorf("[%d] TestMultipleErrorsStruct(): errornum: want: %d, got %d", i, len(test.Offsets), len(errors))
			return
		}
		for ii, e := range errors {
			if e.Offset != test.Offsets[ii] {
				t.Errorf("[%d] TestMultipleErrorsStruct(): offset[%d]: want %d, got %d", i, ii, test.Offsets[ii], e.Offset)
			}
		}
	}
}

func TestMultipleErrorsNestedStruct(t *testing.T) {
	for i, test := range []struct {
		Data    []byte
		Offsets []int
	}{
		{
			Data: []byte(`{"error_struct":{}}`),
		},
		{
			Data:    []byte(`{"error_struct":5}`),
			Offsets: []int{16},
		},
		{
			Data:    []byte(`{"error_struct":[]}`),
			Offsets: []int{16},
		},
		{
			Data:    []byte(`{"error_struct":{"int":{}}}`),
			Offsets: []int{23},
		},
		{
			Data:    []byte(`{"error_struct":{"int_slice":{}}, "int":4}`),
			Offsets: []int{29},
		},
		{
			Data:    []byte(`{"error_struct":{"int_slice":["1", 2, "3"]}, "int":[]}`),
			Offsets: []int{30, 38, 51},
		},
	} {
		l := jlexer.Lexer{
			Data:              test.Data,
			UseMultipleErrors: true,
		}
		var v ErrorNestedStruct
		v.UnmarshalEasyJSON(&l)

		errors := l.GetNonFatalErrors()

		if len(errors) != len(test.Offsets) {
			t.Errorf("[%d] TestMultipleErrorsNestedStruct(): errornum: want: %d, got %d", i, len(test.Offsets), len(errors))
			return
		}
		for ii, e := range errors {
			if e.Offset != test.Offsets[ii] {
				t.Errorf("[%d] TestMultipleErrorsNestedStruct(): offset[%d]: want %d, got %d", i, ii, test.Offsets[ii], e.Offset)
			}
		}
	}
}
