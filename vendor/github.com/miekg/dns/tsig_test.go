package dns

import (
	"testing"
	"time"
)

func newTsig(algo string) *Msg {
	m := new(Msg)
	m.SetQuestion("example.org.", TypeA)
	m.SetTsig("example.", algo, 300, time.Now().Unix())
	return m
}

func TestTsig(t *testing.T) {
	m := newTsig(HmacMD5)
	buf, _, err := TsigGenerate(m, "pRZgBrBvI4NAHZYhxmhs/Q==", "", false)
	if err != nil {
		t.Fatal(err)
	}
	err = TsigVerify(buf, "pRZgBrBvI4NAHZYhxmhs/Q==", "", false)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTsigCase(t *testing.T) {
	m := newTsig("HmAc-mD5.sig-ALg.rEg.int.") // HmacMD5
	buf, _, err := TsigGenerate(m, "pRZgBrBvI4NAHZYhxmhs/Q==", "", false)
	if err != nil {
		t.Fatal(err)
	}
	err = TsigVerify(buf, "pRZgBrBvI4NAHZYhxmhs/Q==", "", false)
	if err != nil {
		t.Fatal(err)
	}
}
