package pktline

import (
	"reflect"
	"strings"
	"testing"
)

var readFromTestCases = []struct {
	desc    string
	in      string
	out     string
	isFlush bool
}{
	{
		desc:    "flush",
		in:      "0000",
		isFlush: true,
	},
	{
		desc: "empty",
		in:   "0004",
		out:  "",
	},
	{
		desc: "simple",
		in:   "000asimple",
		out:  "simple",
	},
}

func TestDecodeEncode(t *testing.T) {
	for _, tt := range readFromTestCases {
		out, err := Decode([]byte(tt.in))
		if err != nil {
			t.Errorf("%s: unexpected error: %v", tt.desc, err)
		} else if tt.isFlush {
			if out != nil {
				t.Errorf("%s: expected flush pkt instead of %v", tt.desc, out)
			}
		} else if !reflect.DeepEqual(out, []byte(tt.out)) {
			t.Errorf("%s: invalid output %v", tt.desc, out)
		} else {
			enc, err := Encode(out)
			if err != nil {
				t.Errorf("%s: unexpected error in encode: %v", tt.desc, err)
			}
			recode, err := Decode(enc)
			if err != nil {
				t.Errorf("%s: unexpected error in recode: %v", tt.desc, err)
			} else if !reflect.DeepEqual(out, recode) {
				t.Errorf("%s: out %q is different from Decode(Encode(out))=%q", tt.desc, out, recode)
			}
		}
	}
}

func TestMultiple(t *testing.T) {
	reader := NewDecoder(strings.NewReader(
		"0005A" +
			"0006BC" +
			"0008\000\001\002\003" +
			"0000" +
			"0004" +
			"0004"))
	expectedLines := [][]byte{
		[]byte("A"),
		[]byte("BC"),
		[]byte("\000\001\002\003"),
		nil,
		[]byte(""),
		[]byte(""),
	}
	for i, expected := range expectedLines {
		var actual []byte
		err := reader.Decode(&actual)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
		} else if !reflect.DeepEqual(expected, actual) {
			t.Errorf("%d: expected %v, got %v", i, expected, actual)
		}
	}
}
