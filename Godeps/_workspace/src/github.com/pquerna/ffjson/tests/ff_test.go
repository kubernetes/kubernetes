/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package tff

import (
	"github.com/pquerna/ffjson/ffjson"
	fflib "github.com/pquerna/ffjson/fflib/v1"
	"github.com/stretchr/testify/require"

	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"
)

// If this is enabled testSameMarshal and testCycle will output failures to files
// for easy debugging.
var outputFileOnError = false

func newLogRecord() *Record {
	return &Record{
		OriginId: 11,
		Method:   "POST",
	}
}

func newLogFFRecord() *FFRecord {
	return &FFRecord{
		OriginId: 11,
		Method:   "POST",
	}
}

func BenchmarkMarshalJSON(b *testing.B) {
	record := newLogRecord()

	buf, err := json.Marshal(&record)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(&record)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkMarshalJSONNative(b *testing.B) {
	record := newLogFFRecord()

	buf, err := json.Marshal(record)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ffjson.MarshalFast(record)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkMarshalJSONNativePool(b *testing.B) {
	record := newLogFFRecord()

	buf, err := json.Marshal(&record)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bytes, err := ffjson.MarshalFast(record)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
		ffjson.Pool(bytes)
	}
}

type NopWriter struct{}

func (*NopWriter) Write(buf []byte) (int, error) {
	return len(buf), nil
}

func BenchmarkMarshalJSONNativeReuse(b *testing.B) {
	record := newLogFFRecord()

	buf, err := json.Marshal(&record)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	enc := ffjson.NewEncoder(&NopWriter{})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := enc.Encode(record)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkSimpleUnmarshal(b *testing.B) {
	record := newLogFFRecord()
	buf := []byte(`{"id": 123213, "OriginId": 22, "meth": "GET"}`)
	err := record.UnmarshalJSON(buf)
	if err != nil {
		b.Fatalf("UnmarshalJSON: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := record.UnmarshalJSON(buf)
		if err != nil {
			b.Fatalf("UnmarshalJSON: %v", err)
		}
	}
}

func BenchmarkSXimpleUnmarshalNative(b *testing.B) {
	record := newLogRecord()
	buf := []byte(`{"id": 123213, "OriginId": 22, "meth": "GET"}`)
	err := json.Unmarshal(buf, record)
	if err != nil {
		b.Fatalf("json.Unmarshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := ffjson.UnmarshalFast(buf, record)
		if err != nil {
			b.Fatalf("json.Unmarshal: %v", err)
		}
	}
}

func TestMarshalFaster(t *testing.T) {
	record := newLogFFRecord()
	_, err := ffjson.MarshalFast(record)
	require.NoError(t, err)

	r2 := newLogRecord()
	_, err = ffjson.MarshalFast(r2)
	require.Error(t, err, "Record should not support MarshalFast")
	_, err = ffjson.Marshal(r2)
	require.NoError(t, err)
}

func TestMarshalEncoder(t *testing.T) {
	record := newLogFFRecord()
	out := bytes.Buffer{}
	enc := ffjson.NewEncoder(&out)
	err := enc.Encode(record)
	require.NoError(t, err)
	require.NotEqual(t, 0, out.Len(), "encoded buffer size should not be 0")

	out.Reset()
	err = enc.EncodeFast(record)
	require.NoError(t, err)
	require.NotEqual(t, 0, out.Len(), "encoded buffer size should not be 0")
}

func TestMarshalEncoderError(t *testing.T) {
	out := NopWriter{}
	enc := ffjson.NewEncoder(&out)
	v := GiveError{}
	err := enc.Encode(v)
	require.Error(t, err, "excpected error from encoder")
	err = enc.Encode(newLogFFRecord())
	require.NoError(t, err, "error did not clear as expected.")

	err = enc.EncodeFast(newLogRecord())
	require.Error(t, err, "excpected error from encoder on type that isn't fast")
}

func TestUnmarshalFaster(t *testing.T) {
	buf := []byte(`{"id": 123213, "OriginId": 22, "meth": "GET"}`)
	record := newLogFFRecord()
	err := ffjson.UnmarshalFast(buf, record)
	require.NoError(t, err)

	r2 := newLogRecord()
	err = ffjson.UnmarshalFast(buf, r2)
	require.Error(t, err, "Record should not support UnmarshalFast")
	err = ffjson.Unmarshal(buf, r2)
	require.NoError(t, err)
}

func TestSimpleUnmarshal(t *testing.T) {
	record := newLogFFRecord()

	err := record.UnmarshalJSON([]byte(`{"id": 123213, "OriginId": 22, "meth": "GET"}`))
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	if record.Timestamp != 123213 {
		t.Fatalf("record.Timestamp: expected: 0 got: %v", record.Timestamp)
	}

	if record.OriginId != 22 {
		t.Fatalf("record.OriginId: expected: 22 got: %v", record.OriginId)
	}

	if record.Method != "GET" {
		t.Fatalf("record.Method: expected: GET got: %v", record.Method)
	}
}

type marshalerFaster interface {
	MarshalJSONBuf(buf fflib.EncodingBuffer) error
}

type unmarshalFaster interface {
	UnmarshalJSONFFLexer(l *fflib.FFLexer, state fflib.FFParseState) error
}

// emptyInterface creates a new instance of the object sent
// It the returned interface is writable and contains the zero value.
func emptyInterface(a interface{}) interface{} {
	aval := reflect.ValueOf(a)
	indirect := reflect.Indirect(aval)
	newIndirect := reflect.New(indirect.Type())

	return newIndirect.Interface()
}
func testType(t *testing.T, base interface{}, ff interface{}) {
	require.Implements(t, (*json.Marshaler)(nil), ff)
	require.Implements(t, (*json.Unmarshaler)(nil), ff)
	require.Implements(t, (*marshalerFaster)(nil), ff)
	require.Implements(t, (*unmarshalFaster)(nil), ff)

	if _, ok := base.(unmarshalFaster); ok {
		require.FailNow(t, "base should not have a UnmarshalJSONFFLexer")
	}

	if _, ok := base.(marshalerFaster); ok {
		require.FailNow(t, "base should not have a MarshalJSONBuf")
	}

	testSameMarshal(t, base, ff)
	testCycle(t, base, ff)
}

func testSameMarshal(t *testing.T, base interface{}, ff interface{}) {
	bufbase, err := json.MarshalIndent(base, " ", "  ")
	require.NoError(t, err, "base[%T] failed to Marshal", base)

	bufff, err := json.MarshalIndent(ff, " ", "  ")
	if err != nil {
		msg := fmt.Sprintf("golang output:\n%s\n", string(bufbase))
		mf, ok := ff.(json.Marshaler)
		var raw []byte
		if ok {
			var err2 error
			raw, err2 = mf.MarshalJSON()
			msg += fmt.Sprintf("Raw output:\n%s\nErros:%v", string(raw), err2)
		}
		if outputFileOnError {
			typeName := reflect.Indirect(reflect.ValueOf(base)).Type().String()
			file, err := os.Create(fmt.Sprintf("fail-%s-marshal-go.json", typeName))
			if err == nil {
				file.Write(bufbase)
				file.Close()
			}
			if len(raw) != 0 {
				file, err = os.Create(fmt.Sprintf("fail-%s-marshal-ffjson-raw.json", typeName))
				if err == nil {
					file.Write(raw)
					file.Close()
				}
			}
		}
		require.NoError(t, err, "ff[%T] failed to Marshal:%s", ff, msg)
	}

	if outputFileOnError {
		if string(bufbase) != string(bufff) {
			typeName := reflect.Indirect(reflect.ValueOf(base)).Type().String()
			file, err := os.Create(fmt.Sprintf("fail-%s-marshal-base.json", typeName))
			if err == nil {
				file.Write(bufbase)
				file.Close()
			}
			file, err = os.Create(fmt.Sprintf("fail-%s-marshal-ffjson.json", typeName))
			if err == nil {
				file.Write(bufff)
				file.Close()
			}
		}
	}

	require.Equal(t, string(bufbase), string(bufff), "json.Marshal of base[%T] != ff[%T]", base, ff)
}

func testCycle(t *testing.T, base interface{}, ff interface{}) {
	setXValue(t, base)

	buf, err := json.MarshalIndent(base, " ", "  ")
	require.NoError(t, err, "base[%T] failed to Marshal", base)

	ffDst := emptyInterface(ff)
	baseDst := emptyInterface(base)

	err = json.Unmarshal(buf, ffDst)
	errGo := json.Unmarshal(buf, baseDst)
	if outputFileOnError && err != nil {
		typeName := reflect.Indirect(reflect.ValueOf(base)).Type().String()
		file, err := os.Create(fmt.Sprintf("fail-%s-unmarshal-decoder-input.json", typeName))
		if err == nil {
			file.Write(buf)
			file.Close()
		}
		if errGo == nil {
			file, err := os.Create(fmt.Sprintf("fail-%s-unmarshal-decoder-output-base.txt", typeName))
			if err == nil {
				fmt.Fprintf(file, "%#v", baseDst)
				file.Close()
			}
		}
	}
	require.Nil(t, err, "json.Unmarshal of encoded ff[%T],\nErrors golang:%v,\nffjson:%v", ff, errGo, err)
	require.Nil(t, errGo, "json.Unmarshal of encoded ff[%T],\nerrors golang:%v,\nffjson:%v", base, errGo, err)

	require.EqualValues(t, baseDst, ffDst, "json.Unmarshal of base[%T] into ff[%T]", base, ff)
}

func testExpectedX(t *testing.T, expected interface{}, base interface{}, ff interface{}) {
	buf, err := json.Marshal(base)
	require.NoError(t, err, "base[%T] failed to Marshal", base)

	err = json.Unmarshal(buf, ff)
	require.NoError(t, err, "ff[%T] failed to Unmarshal", ff)

	require.Equal(t, expected, getXValue(ff), "json.Unmarshal of base[%T] into ff[%T]", base, ff)
}

func testExpectedXValBare(t *testing.T, expected interface{}, xval string, ff interface{}) {
	buf := []byte(`{"X":` + xval + `}`)
	err := json.Unmarshal(buf, ff)
	require.NoError(t, err, "ff[%T] failed to Unmarshal", ff)

	require.Equal(t, expected, getXValue(ff), "json.Unmarshal of %T into ff[%T]", xval, ff)
}

func testExpectedXVal(t *testing.T, expected interface{}, xval string, ff interface{}) {
	testExpectedXValBare(t, expected, `"`+xval+`"`, ff)
}

func testExpectedError(t *testing.T, expected error, xval string, ff json.Unmarshaler) {
	buf := []byte(`{"X":` + xval + `}`)
	err := ff.UnmarshalJSON(buf)
	require.Error(t, err, "ff[%T] failed to Unmarshal", ff)
	require.IsType(t, expected, err)
}

func setXValue(t *testing.T, thing interface{}) {
	v := reflect.ValueOf(thing)
	v = reflect.Indirect(v)
	f := v.FieldByName("X")
	switch f.Kind() {
	case reflect.Bool:
		f.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		f.SetInt(-42)
	case reflect.Uint, reflect.Uintptr, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		f.SetUint(42)
	case reflect.Float32, reflect.Float64:
		f.SetFloat(3.141592653)
	case reflect.String:
		f.SetString("hello world")
	}
}

func getXValue(thing interface{}) interface{} {
	v := reflect.ValueOf(thing)
	v = reflect.Indirect(v)
	f := v.FieldByName("X")
	switch f.Kind() {
	case reflect.Bool:
		return f.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return f.Int()
	case reflect.Uint, reflect.Uintptr, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return f.Uint()
	case reflect.Float32, reflect.Float64:
		return f.Float()
	case reflect.String:
		return f.String()
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	enc.Encode(f)
	return buf.String()
}

func TestArray(t *testing.T) {
	testType(t, &Tarray{X: []int{}}, &Xarray{X: []int{}})
	testCycle(t, &Tarray{X: []int{42, -42, 44}}, &Xarray{X: []int{}})

	x := Xarray{X: []int{222}}
	buf := []byte(`{"X": null}`)
	err := json.Unmarshal(buf, &x)
	require.NoError(t, err, "Unmarshal of null into array.")
	var eq []int = nil
	require.Equal(t, x.X, eq)
}

func TestArrayPtr(t *testing.T) {
	testType(t, &TarrayPtr{X: []*int{}}, &XarrayPtr{X: []*int{}})
	v := 33
	testCycle(t, &TarrayPtr{X: []*int{&v}}, &XarrayPtr{X: []*int{}})
}

func TestTimeDuration(t *testing.T) {
	testType(t, &Tduration{}, &Xduration{})
}

func TestI18nName(t *testing.T) {
	testType(t, &TI18nName{}, &XI18nName{})
}

func TestTimeTimePtr(t *testing.T) {
	tm := time.Date(2014, 12, 13, 15, 16, 17, 18, time.UTC)
	testType(t, &TtimePtr{X: &tm}, &XtimePtr{X: &tm})
}

func TestTimeNullTimePtr(t *testing.T) {
	testType(t, &TtimePtr{}, &XtimePtr{})
}

func TestBool(t *testing.T) {
	testType(t, &Tbool{}, &Xbool{})
	testExpectedXValBare(t,
		true,
		`null`,
		&Xbool{X: true})
}

func TestInt(t *testing.T) {
	testType(t, &Tint{}, &Xint{})
}

func TestByte(t *testing.T) {
	testType(t, &Tbyte{}, &Xbyte{})
}

func TestInt8(t *testing.T) {
	testType(t, &Tint8{}, &Xint8{})
}

func TestInt16(t *testing.T) {
	testType(t, &Tint16{}, &Xint16{})
}

func TestInt32(t *testing.T) {
	testType(t, &Tint32{}, &Xint32{})
}

func TestInt64(t *testing.T) {
	testType(t, &Tint64{}, &Xint64{})
}

func TestUint(t *testing.T) {
	testType(t, &Tuint{}, &Xuint{})
}

func TestUint8(t *testing.T) {
	testType(t, &Tuint8{}, &Xuint8{})
}

func TestUint16(t *testing.T) {
	testType(t, &Tuint16{}, &Xuint16{})
}

func TestUint32(t *testing.T) {
	testType(t, &Tuint32{}, &Xuint32{})
}

func TestUint64(t *testing.T) {
	testType(t, &Tuint64{}, &Xuint64{})
}

func TestUintptr(t *testing.T) {
	testType(t, &Tuintptr{}, &Xuintptr{})
}

func TestFloat32(t *testing.T) {
	testType(t, &Tfloat32{}, &Xfloat32{})
}

func TestFloat64(t *testing.T) {
	testType(t, &Tfloat64{}, &Xfloat64{})
}

func TestForceStringTagged(t *testing.T) {
	// testSameMarshal is used instead of testType because
	// the string tag is a one way effect, Unmarshaling doesn't
	// work because the receiving type must be a string.
	testSameMarshal(t, &TstringTagged{}, &XstringTagged{})
	testSameMarshal(t, &TintTagged{}, &XintTagged{})
	testSameMarshal(t, &TboolTagged{}, &XboolTagged{})
}

func TestForceStringTaggedEscape(t *testing.T) {
	testSameMarshal(t, &TstringTagged{X: `"`}, &XstringTagged{X: `"`})
}

func TestForceStringTaggedDecoder(t *testing.T) {
	t.Skip("https://github.com/pquerna/ffjson/issues/80")
	testCycle(t, &TstringTagged{}, &XstringTagged{})
	testCycle(t, &TintTagged{}, &XintTagged{})
	testCycle(t, &TboolTagged{}, &XboolTagged{})
}

func TestSortSame(t *testing.T) {
	testSameMarshal(t, &TsortName{C: "foo", B: 12}, &XsortName{C: "foo", B: 12})
}

func TestEncodeRenamedByteSlice(t *testing.T) {
	expect := `{"X":"YWJj"}`

	s := ByteSliceNormal{X: []byte("abc")}
	result, err := s.MarshalJSON()
	require.NoError(t, err)
	require.Equal(t, string(result), expect)

	r := ByteSliceRenamed{X: renamedByteSlice("abc")}
	result, err = r.MarshalJSON()
	require.NoError(t, err)
	require.Equal(t, string(result), expect)

	rr := ByteSliceDoubleRenamed{X: renamedRenamedByteSlice("abc")}
	result, err = rr.MarshalJSON()
	require.NoError(t, err)
	require.Equal(t, string(result), expect)
}

// Test arrays
func TestArrayBool(t *testing.T) {
	testType(t, &ATbool{}, &AXbool{})
}

func TestArrayInt(t *testing.T) {
	testType(t, &ATint{}, &AXint{})
}

func TestArrayByte(t *testing.T) {
	testType(t, &ATbyte{}, &AXbyte{})
}

func TestArrayInt8(t *testing.T) {
	testType(t, &ATint8{}, &AXint8{})
}

func TestArrayInt16(t *testing.T) {
	testType(t, &ATint16{}, &AXint16{})
}

func TestArrayInt32(t *testing.T) {
	testType(t, &ATint32{}, &AXint32{})
}

func TestArrayInt64(t *testing.T) {
	testType(t, &ATint64{}, &AXint64{})
}

func TestArrayUint(t *testing.T) {
	testType(t, &ATuint{}, &AXuint{})
}

func TestArrayUint8(t *testing.T) {
	testType(t, &ATuint8{}, &AXuint8{})
}

func TestArrayUint16(t *testing.T) {
	testType(t, &ATuint16{}, &AXuint16{})
}

func TestArrayUint32(t *testing.T) {
	testType(t, &ATuint32{}, &AXuint32{})
}

func TestArrayUint64(t *testing.T) {
	testType(t, &ATuint64{}, &AXuint64{})
}

func TestArrayUintptr(t *testing.T) {
	testType(t, &ATuintptr{}, &AXuintptr{})
}

func TestArrayFloat32(t *testing.T) {
	testType(t, &ATfloat32{}, &AXfloat32{})
}

func TestArrayFloat64(t *testing.T) {
	testType(t, &ATfloat64{}, &AXfloat64{})
}

func TestArrayTime(t *testing.T) {
	testType(t, &ATtime{}, &AXtime{})
}

func TestNoDecoder(t *testing.T) {
	var test interface{} = &NoDecoder{}
	if _, ok := test.(unmarshalFaster); ok {
		require.FailNow(t, "NoDecoder should not have a UnmarshalJSONFFLexer")
	}
}

func TestNoEncoder(t *testing.T) {
	var test interface{} = &NoEncoder{}
	if _, ok := test.(marshalerFaster); ok {
		require.FailNow(t, "NoEncoder should not have a MarshalJSONBuf")
	}
}

func TestCaseSensitiveUnmarshalSimple(t *testing.T) {
	base := Tint{}
	ff := Xint{}

	err := json.Unmarshal([]byte(`{"x": 123213}`), &base)
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	err = json.Unmarshal([]byte(`{"x": 123213}`), &ff)
	if err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	require.EqualValues(t, base, ff, "json.Unmarshal of Record with mixed case JSON")
}

func TestEmbedded(t *testing.T) {
	a := TEmbeddedStructures{}
	a.X = make([]interface{}, 0)
	a.X = append(a.X, "testString")
	a.Y.X = 73
	a.Z = make([]struct{ X int }, 2)
	a.Z[0].X = 12
	a.Z[1].X = 34
	a.U = make(map[string]struct{ X int })
	a.U["sample"] = struct{ X int }{X: 56}
	a.U["value"] = struct{ X int }{X: 78}
	a.V = make([]map[string]struct{ X int }, 3)
	for i := range a.V {
		a.V[i] = make(map[string]struct{ X int })
		a.V[i]["sample"] = struct{ X int }{X: i * 3}
	}
	for i := range a.W {
		a.W[i] = make(map[string]struct{ X int })
		a.W[i]["sample"] = struct{ X int }{X: i * 3}
		a.W[i]["value"] = struct{ X int }{X: i * 5}
	}
	a.Q = make([][]string, 3)
	for i := range a.Q {
		a.Q[i] = make([]string, 1)
		a.Q[i][0] = fmt.Sprintf("thestring #%d", i)
	}

	b := XEmbeddedStructures{}
	b.X = make([]interface{}, 0)
	b.X = append(b.X, "testString")
	b.Y.X = 73
	b.Z = make([]struct{ X int }, 2)
	b.Z[0].X = 12
	b.Z[1].X = 34
	b.U = make(map[string]struct{ X int })
	b.U["sample"] = struct{ X int }{X: 56}
	b.U["value"] = struct{ X int }{X: 78}
	b.V = make([]map[string]struct{ X int }, 3)
	for i := range b.V {
		b.V[i] = make(map[string]struct{ X int })
		b.V[i]["sample"] = struct{ X int }{X: i * 3}
	}
	for i := range b.W {
		b.W[i] = make(map[string]struct{ X int })
		b.W[i]["sample"] = struct{ X int }{X: i * 3}
		b.W[i]["value"] = struct{ X int }{X: i * 5}
	}
	b.Q = make([][]string, 3)
	for i := range a.Q {
		b.Q[i] = make([]string, 1)
		b.Q[i][0] = fmt.Sprintf("thestring #%d", i)
	}
	testSameMarshal(t, &a, &b)
	testCycle(t, &a, &b)
}

func TestRenameTypes(t *testing.T) {
	testType(t, &TRenameTypes{}, &XRenameTypes{})
}

func TestInlineStructs(t *testing.T) {
	a := TInlineStructs{}
	b := XInlineStructs{}
	testSameMarshal(t, &a, &b)
	testCycle(t, &a, &b)
}

// This tests that we behave the same way as encoding/json.
// That means that if there is more than one field that has the same name
// set via the json tag ALL fields with this name are dropped.
func TestDominantField(t *testing.T) {
	i := 43
	testType(t, &TDominantField{Y: &i}, &XDominantField{Y: &i})
}
