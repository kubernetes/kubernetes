package regretable_test

import (
	. "github.com/elazarl/goproxy/regretable"
	"bytes"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

func TestRegretableReader(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	fivebytes := make([]byte, 5)
	mb.Read(fivebytes)
	mb.Regret()

	s, _ := ioutil.ReadAll(mb)
	if string(s) != word {
		t.Errorf("Uncommited read is gone, [%d,%d] actual '%v' expected '%v'\n", len(s), len(word), string(s), word)
	}
}

func TestRegretableEmptyRead(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	zero := make([]byte, 0)
	mb.Read(zero)
	mb.Regret()

	s, err := ioutil.ReadAll(mb)
	if string(s) != word {
		t.Error("Uncommited read is gone, actual:", string(s), "expected:", word, "err:", err)
	}
}

func TestRegretableAlsoEmptyRead(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	one := make([]byte, 1)
	zero := make([]byte, 0)
	five := make([]byte, 5)
	mb.Read(one)
	mb.Read(zero)
	mb.Read(five)
	mb.Regret()

	s, _ := ioutil.ReadAll(mb)
	if string(s) != word {
		t.Error("Uncommited read is gone", string(s), "expected", word)
	}
}

func TestRegretableRegretBeforeRead(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	five := make([]byte, 5)
	mb.Regret()
	mb.Read(five)

	s, err := ioutil.ReadAll(mb)
	if string(s) != "678" {
		t.Error("Uncommited read is gone", string(s), len(string(s)), "expected", "678", len("678"), "err:", err)
	}
}

func TestRegretableFullRead(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	twenty := make([]byte, 20)
	mb.Read(twenty)
	mb.Regret()

	s, _ := ioutil.ReadAll(mb)
	if string(s) != word {
		t.Error("Uncommited read is gone", string(s), len(string(s)), "expected", word, len(word))
	}
}

func assertEqual(t *testing.T, expected, actual string) {
	if expected!=actual {
		t.Fatal("Expected", expected, "actual", actual)
	}
}

func assertReadAll(t *testing.T, r io.Reader) string {
	s, err := ioutil.ReadAll(r)
	if err!=nil {
		t.Fatal("error when reading", err)
	}
	return string(s)
}

func TestRegretableRegretTwice(t *testing.T) {
	buf := new(bytes.Buffer)
	mb := NewRegretableReader(buf)
	word := "12345678"
	buf.WriteString(word)

	assertEqual(t, word, assertReadAll(t, mb))
	mb.Regret()
	assertEqual(t, word, assertReadAll(t, mb))
	mb.Regret()
	assertEqual(t, word, assertReadAll(t, mb))
}

type CloseCounter struct {
	r      io.Reader
	closed int
}

func (cc *CloseCounter) Read(b []byte) (int, error) {
	return cc.r.Read(b)
}

func (cc *CloseCounter) Close() error {
	cc.closed++
	return nil
}

func assert(t *testing.T, b bool, msg string) {
	if !b {
		t.Errorf("Assertion Error: %s", msg)
	}
}

func TestRegretableCloserSizeRegrets(t *testing.T) {
	defer func() {
		if r := recover(); r == nil || !strings.Contains(r.(string), "regret") {
			t.Error("Did not panic when regretting overread buffer:", r)
		}
	}()
	buf := new(bytes.Buffer)
	buf.WriteString("123456")
	mb := NewRegretableReaderCloserSize(ioutil.NopCloser(buf), 3)
	mb.Read(make([]byte, 4))
	mb.Regret()
}

func TestRegretableCloserRegretsClose(t *testing.T) {
	buf := new(bytes.Buffer)
	cc := &CloseCounter{buf, 0}
	mb := NewRegretableReaderCloser(cc)
	word := "12345678"
	buf.WriteString(word)

	mb.Read([]byte{0})
	mb.Close()
	if cc.closed != 1 {
		t.Error("RegretableReaderCloser ignores Close")
	}
	mb.Regret()
	mb.Close()
	if cc.closed != 2 {
		t.Error("RegretableReaderCloser does ignore Close after regret")
	}
	// TODO(elazar): return an error if client issues Close more than once after regret
}
