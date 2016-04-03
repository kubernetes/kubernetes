package dbus

import (
	"testing"
)

var sigTests = []struct {
	vs  []interface{}
	sig Signature
}{
	{
		[]interface{}{new(int32)},
		Signature{"i"},
	},
	{
		[]interface{}{new(string)},
		Signature{"s"},
	},
	{
		[]interface{}{new(Signature)},
		Signature{"g"},
	},
	{
		[]interface{}{new([]int16)},
		Signature{"an"},
	},
	{
		[]interface{}{new(int16), new(uint32)},
		Signature{"nu"},
	},
	{
		[]interface{}{new(map[byte]Variant)},
		Signature{"a{yv}"},
	},
	{
		[]interface{}{new(Variant), new([]map[int32]string)},
		Signature{"vaa{is}"},
	},
}

func TestSig(t *testing.T) {
	for i, v := range sigTests {
		sig := SignatureOf(v.vs...)
		if sig != v.sig {
			t.Errorf("test %d: got %q, expected %q", i+1, sig.str, v.sig.str)
		}
	}
}

var getSigTest = []interface{}{
	[]struct {
		b byte
		i int32
		t uint64
		s string
	}{},
	map[string]Variant{},
}

func BenchmarkGetSignatureSimple(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SignatureOf("", int32(0))
	}
}

func BenchmarkGetSignatureLong(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SignatureOf(getSigTest...)
	}
}
