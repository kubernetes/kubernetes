// +build go1.8

package jsoniter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

func Test_read_string(t *testing.T) {
	badInputs := []string{
		``,
		`"`,
		`"\"`,
		`"\\\"`,
		"\"\n\"",
		`"\U0001f64f"`,
		`"\uD83D\u00"`,
	}
	for i := 0; i < 32; i++ {
		// control characters are invalid
		badInputs = append(badInputs, string([]byte{'"', byte(i), '"'}))
	}

	for _, input := range badInputs {
		testReadString(t, input, "", true, "json.Unmarshal", json.Unmarshal)
		testReadString(t, input, "", true, "jsoniter.Unmarshal", Unmarshal)
		testReadString(t, input, "", true, "jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal", ConfigCompatibleWithStandardLibrary.Unmarshal)
	}

	goodInputs := []struct {
		input       string
		expectValue string
	}{
		{`""`, ""},
		{`"a"`, "a"},
		{`null`, ""},
		{`"Iñtërnâtiônàlizætiøn,💝🐹🌇⛔"`, "Iñtërnâtiônàlizætiøn,💝🐹🌇⛔"},
		{`"\uD83D"`, string([]byte{239, 191, 189})},
		{`"\uD83D\\"`, string([]byte{239, 191, 189, '\\'})},
		{`"\uD83D\ub000"`, string([]byte{239, 191, 189, 235, 128, 128})},
		{`"\uD83D\ude04"`, "😄"},
		{`"\uDEADBEEF"`, string([]byte{239, 191, 189, 66, 69, 69, 70})},
	}

	for _, tc := range goodInputs {
		testReadString(t, tc.input, tc.expectValue, false, "json.Unmarshal", json.Unmarshal)
		testReadString(t, tc.input, tc.expectValue, false, "jsoniter.Unmarshal", Unmarshal)
		testReadString(t, tc.input, tc.expectValue, false, "jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal", ConfigCompatibleWithStandardLibrary.Unmarshal)
	}
}

func testReadString(t *testing.T, input string, expectValue string, expectError bool, marshalerName string, marshaler func([]byte, interface{}) error) {
	var value string
	err := marshaler([]byte(input), &value)
	if expectError != (err != nil) {
		t.Errorf("%q: %s: expected error %v, got %v", input, marshalerName, expectError, err)
		return
	}
	if value != expectValue {
		t.Errorf("%q: %s: expected %q, got %q", input, marshalerName, expectValue, value)
		return
	}
}

func Test_read_normal_string(t *testing.T) {
	cases := map[string]string{
		`"0123456789012345678901234567890123456789"`: `0123456789012345678901234567890123456789`,
		`""`:      ``,
		`"hello"`: `hello`,
	}
	for input, output := range cases {
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			should.Equal(output, iter.ReadString())
		})
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input), 2)
			should.Equal(output, iter.ReadString())
		})
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			should.Equal(output, string(iter.ReadStringAsSlice()))
		})
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input), 2)
			should.Equal(output, string(iter.ReadStringAsSlice()))
		})
	}
}

func Test_read_exotic_string(t *testing.T) {
	cases := map[string]string{
		`"hel\"lo"`:      `hel"lo`,
		`"hel\\\/lo"`:    `hel\/lo`,
		`"hel\\blo"`:     `hel\blo`,
		`"hel\\\blo"`:    "hel\\\blo",
		`"hel\\nlo"`:     `hel\nlo`,
		`"hel\\\nlo"`:    "hel\\\nlo",
		`"hel\\tlo"`:     `hel\tlo`,
		`"hel\\flo"`:     `hel\flo`,
		`"hel\\\flo"`:    "hel\\\flo",
		`"hel\\\rlo"`:    "hel\\\rlo",
		`"hel\\\tlo"`:    "hel\\\tlo",
		`"\u4e2d\u6587"`: "中文",
		`"\ud83d\udc4a"`: "\xf0\x9f\x91\x8a", // surrogate
	}
	for input, output := range cases {
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := ParseString(ConfigDefault, input)
			var v string
			should.Nil(json.Unmarshal([]byte(input), &v))
			should.Equal(v, iter.ReadString())
		})
		t.Run(fmt.Sprintf("%v:%v", input, output), func(t *testing.T) {
			should := require.New(t)
			iter := Parse(ConfigDefault, bytes.NewBufferString(input), 2)
			should.Equal(output, iter.ReadString())
		})
	}
}

func Test_read_string_as_interface(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `"hello"`)
	should.Equal("hello", iter.Read())
}

func Test_write_string(t *testing.T) {
	should := require.New(t)
	str, err := MarshalToString("hello")
	should.Equal(`"hello"`, str)
	should.Nil(err)
	str, err = MarshalToString(`hel"lo`)
	should.Equal(`"hel\"lo"`, str)
	should.Nil(err)
}

func Test_write_val_string(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := NewStream(ConfigDefault, buf, 4096)
	stream.WriteVal("hello")
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal(`"hello"`, buf.String())
}

func Test_decode_slash(t *testing.T) {
	should := require.New(t)
	var obj interface{}
	should.NotNil(json.Unmarshal([]byte("\\"), &obj))
	should.NotNil(UnmarshalFromString("\\", &obj))
}

func Test_html_escape(t *testing.T) {
	should := require.New(t)
	output, err := json.Marshal(`>`)
	should.Nil(err)
	should.Equal(`"\u003e"`, string(output))
	output, err = ConfigCompatibleWithStandardLibrary.Marshal(`>`)
	should.Nil(err)
	should.Equal(`"\u003e"`, string(output))
	type MyString string
	output, err = ConfigCompatibleWithStandardLibrary.Marshal(MyString(`>`))
	should.Nil(err)
	should.Equal(`"\u003e"`, string(output))
}

func Test_string_encode_with_std(t *testing.T) {
	should := require.New(t)
	for i := 0; i < utf8.RuneSelf; i++ {
		input := string([]byte{byte(i)})
		stdOutputBytes, err := json.Marshal(input)
		should.Nil(err)
		stdOutput := string(stdOutputBytes)
		jsoniterOutputBytes, err := ConfigCompatibleWithStandardLibrary.Marshal(input)
		should.Nil(err)
		jsoniterOutput := string(jsoniterOutputBytes)
		should.Equal(stdOutput, jsoniterOutput)
	}
}

func Test_unicode(t *testing.T) {
	should := require.New(t)
	output, _ := MarshalToString(map[string]interface{}{"a": "数字山谷"})
	should.Equal(`{"a":"数字山谷"}`, output)
	output, _ = Config{EscapeHTML: false}.Froze().MarshalToString(map[string]interface{}{"a": "数字山谷"})
	should.Equal(`{"a":"数字山谷"}`, output)
}

func Test_unicode_and_escape(t *testing.T) {
	should := require.New(t)
	output, err := MarshalToString(`"数字山谷"`)
	should.Nil(err)
	should.Equal(`"\"数字山谷\""`, output)
	output, err = ConfigFastest.MarshalToString(`"数字山谷"`)
	should.Nil(err)
	should.Equal(`"\"数字山谷\""`, output)
}

func Test_unsafe_unicode(t *testing.T) {
	ConfigDefault.(*frozenConfig).cleanEncoders()
	should := require.New(t)
	output, err := ConfigDefault.MarshalToString("he\u2029\u2028he")
	should.Nil(err)
	should.Equal(`"he\u2029\u2028he"`, output)
	output, err = ConfigFastest.MarshalToString("he\u2029\u2028he")
	should.Nil(err)
	should.Equal("\"he\u2029\u2028he\"", output)
}

func Benchmark_jsoniter_unicode(b *testing.B) {
	for n := 0; n < b.N; n++ {
		iter := ParseString(ConfigDefault, `"\ud83d\udc4a"`)
		iter.ReadString()
	}
}

func Benchmark_jsoniter_ascii(b *testing.B) {
	iter := NewIterator(ConfigDefault)
	input := []byte(`"hello, world! hello, world!"`)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(input)
		iter.ReadString()
	}
}

func Benchmark_jsoniter_string_as_bytes(b *testing.B) {
	iter := ParseString(ConfigDefault, `"hello, world!"`)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		iter.ResetBytes(iter.buf)
		iter.ReadStringAsSlice()
	}
}

func Benchmark_json_unicode(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := ""
		json.Unmarshal([]byte(`"\ud83d\udc4a"`), &result)
	}
}

func Benchmark_json_ascii(b *testing.B) {
	for n := 0; n < b.N; n++ {
		result := ""
		json.Unmarshal([]byte(`"hello"`), &result)
	}
}
