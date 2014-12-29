package inf_test

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math/big"
	"strings"
	"testing"

	"speter.net/go/exp/math/dec/inf"
)

type decFunZZ func(z, x, y *inf.Dec) *inf.Dec
type decArgZZ struct {
	z, x, y *inf.Dec
}

var decSumZZ = []decArgZZ{
	{inf.NewDec(0, 0), inf.NewDec(0, 0), inf.NewDec(0, 0)},
	{inf.NewDec(1, 0), inf.NewDec(1, 0), inf.NewDec(0, 0)},
	{inf.NewDec(1111111110, 0), inf.NewDec(123456789, 0), inf.NewDec(987654321, 0)},
	{inf.NewDec(-1, 0), inf.NewDec(-1, 0), inf.NewDec(0, 0)},
	{inf.NewDec(864197532, 0), inf.NewDec(-123456789, 0), inf.NewDec(987654321, 0)},
	{inf.NewDec(-1111111110, 0), inf.NewDec(-123456789, 0), inf.NewDec(-987654321, 0)},
	{inf.NewDec(12, 2), inf.NewDec(1, 1), inf.NewDec(2, 2)},
}

var decProdZZ = []decArgZZ{
	{inf.NewDec(0, 0), inf.NewDec(0, 0), inf.NewDec(0, 0)},
	{inf.NewDec(0, 0), inf.NewDec(1, 0), inf.NewDec(0, 0)},
	{inf.NewDec(1, 0), inf.NewDec(1, 0), inf.NewDec(1, 0)},
	{inf.NewDec(-991*991, 0), inf.NewDec(991, 0), inf.NewDec(-991, 0)},
	{inf.NewDec(2, 3), inf.NewDec(1, 1), inf.NewDec(2, 2)},
	{inf.NewDec(2, -3), inf.NewDec(1, -1), inf.NewDec(2, -2)},
	{inf.NewDec(2, 3), inf.NewDec(1, 1), inf.NewDec(2, 2)},
}

func TestDecSignZ(t *testing.T) {
	var zero inf.Dec
	for _, a := range decSumZZ {
		s := a.z.Sign()
		e := a.z.Cmp(&zero)
		if s != e {
			t.Errorf("got %d; want %d for z = %v", s, e, a.z)
		}
	}
}

func TestDecAbsZ(t *testing.T) {
	var zero inf.Dec
	for _, a := range decSumZZ {
		var z inf.Dec
		z.Abs(a.z)
		var e inf.Dec
		e.Set(a.z)
		if e.Cmp(&zero) < 0 {
			e.Sub(&zero, &e)
		}
		if z.Cmp(&e) != 0 {
			t.Errorf("got z = %v; want %v", z, e)
		}
	}
}

func testDecFunZZ(t *testing.T, msg string, f decFunZZ, a decArgZZ) {
	var z inf.Dec
	f(&z, a.x, a.y)
	if (&z).Cmp(a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, &z, a.z)
	}
}

func TestDecSumZZ(t *testing.T) {
	AddZZ := func(z, x, y *inf.Dec) *inf.Dec { return z.Add(x, y) }
	SubZZ := func(z, x, y *inf.Dec) *inf.Dec { return z.Sub(x, y) }
	for _, a := range decSumZZ {
		arg := a
		testDecFunZZ(t, "AddZZ", AddZZ, arg)

		arg = decArgZZ{a.z, a.y, a.x}
		testDecFunZZ(t, "AddZZ symmetric", AddZZ, arg)

		arg = decArgZZ{a.x, a.z, a.y}
		testDecFunZZ(t, "SubZZ", SubZZ, arg)

		arg = decArgZZ{a.y, a.z, a.x}
		testDecFunZZ(t, "SubZZ symmetric", SubZZ, arg)
	}
}

func TestDecProdZZ(t *testing.T) {
	MulZZ := func(z, x, y *inf.Dec) *inf.Dec { return z.Mul(x, y) }
	for _, a := range decProdZZ {
		arg := a
		testDecFunZZ(t, "MulZZ", MulZZ, arg)

		arg = decArgZZ{a.z, a.y, a.x}
		testDecFunZZ(t, "MulZZ symmetric", MulZZ, arg)
	}
}

var decUnscaledTests = []struct {
	d  *inf.Dec
	u  int64 // ignored when ok == false
	ok bool
}{
	{new(inf.Dec), 0, true},
	{inf.NewDec(-1<<63, 0), -1 << 63, true},
	{inf.NewDec(-(-1<<63 + 1), 0), -(-1<<63 + 1), true},
	{new(inf.Dec).Neg(inf.NewDec(-1<<63, 0)), 0, false},
	{new(inf.Dec).Sub(inf.NewDec(-1<<63, 0), inf.NewDec(1, 0)), 0, false},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), 0, false},
}

func TestDecUnscaled(t *testing.T) {
	for i, tt := range decUnscaledTests {
		u, ok := tt.d.Unscaled()
		if ok != tt.ok {
			t.Errorf("#%d Unscaled: got %v, expected %v", i, ok, tt.ok)
		} else if ok && u != tt.u {
			t.Errorf("#%d Unscaled: got %v, expected %v", i, u, tt.u)
		}
	}
}

var decRoundTests = [...]struct {
	in  *inf.Dec
	s   inf.Scale
	r   inf.Rounder
	exp *inf.Dec
}{
	{inf.NewDec(123424999999999993, 15), 2, inf.RoundHalfUp, inf.NewDec(12342, 2)},
	{inf.NewDec(123425000000000001, 15), 2, inf.RoundHalfUp, inf.NewDec(12343, 2)},
	{inf.NewDec(123424999999999993, 15), 15, inf.RoundHalfUp, inf.NewDec(123424999999999993, 15)},
	{inf.NewDec(123424999999999993, 15), 16, inf.RoundHalfUp, inf.NewDec(1234249999999999930, 16)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -1, inf.RoundHalfUp, inf.NewDec(1844674407370955162, -1)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -2, inf.RoundHalfUp, inf.NewDec(184467440737095516, -2)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -3, inf.RoundHalfUp, inf.NewDec(18446744073709552, -3)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -4, inf.RoundHalfUp, inf.NewDec(1844674407370955, -4)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -5, inf.RoundHalfUp, inf.NewDec(184467440737096, -5)},
	{inf.NewDecBig(new(big.Int).Lsh(big.NewInt(1), 64), 0), -6, inf.RoundHalfUp, inf.NewDec(18446744073710, -6)},
}

func TestDecRound(t *testing.T) {
	for i, tt := range decRoundTests {
		z := new(inf.Dec).Round(tt.in, tt.s, tt.r)
		if tt.exp.Cmp(z) != 0 {
			t.Errorf("#%d Round got %v; expected %v", i, z, tt.exp)
		}
	}
}

var decStringTests = []struct {
	in     string
	out    string
	val    int64
	scale  inf.Scale // skip SetString if negative
	ok     bool
	scanOk bool
}{
	{in: "", ok: false, scanOk: false},
	{in: "a", ok: false, scanOk: false},
	{in: "z", ok: false, scanOk: false},
	{in: "+", ok: false, scanOk: false},
	{in: "-", ok: false, scanOk: false},
	{in: "g", ok: false, scanOk: false},
	{in: ".", ok: false, scanOk: false},
	{in: ".-0", ok: false, scanOk: false},
	{in: ".+0", ok: false, scanOk: false},
	// Scannable but not SetStringable
	{"0b", "ignored", 0, 0, false, true},
	{"0x", "ignored", 0, 0, false, true},
	{"0xg", "ignored", 0, 0, false, true},
	{"0.0g", "ignored", 0, 1, false, true},
	// examples from godoc for Dec
	{"0", "0", 0, 0, true, true},
	{"0.00", "0.00", 0, 2, true, true},
	{"ignored", "0", 0, -2, true, false},
	{"1", "1", 1, 0, true, true},
	{"1.00", "1.00", 100, 2, true, true},
	{"10", "10", 10, 0, true, true},
	{"ignored", "10", 1, -1, true, false},
	// other tests
	{"+0", "0", 0, 0, true, true},
	{"-0", "0", 0, 0, true, true},
	{"0.0", "0.0", 0, 1, true, true},
	{"0.1", "0.1", 1, 1, true, true},
	{"0.", "0", 0, 0, true, true},
	{"-10", "-10", -1, -1, true, true},
	{"-1", "-1", -1, 0, true, true},
	{"-0.1", "-0.1", -1, 1, true, true},
	{"-0.01", "-0.01", -1, 2, true, true},
	{"+0.", "0", 0, 0, true, true},
	{"-0.", "0", 0, 0, true, true},
	{".0", "0.0", 0, 1, true, true},
	{"+.0", "0.0", 0, 1, true, true},
	{"-.0", "0.0", 0, 1, true, true},
	{"0.0000000000", "0.0000000000", 0, 10, true, true},
	{"0.0000000001", "0.0000000001", 1, 10, true, true},
	{"-0.0000000000", "0.0000000000", 0, 10, true, true},
	{"-0.0000000001", "-0.0000000001", -1, 10, true, true},
	{"-10", "-10", -10, 0, true, true},
	{"+10", "10", 10, 0, true, true},
	{"00", "0", 0, 0, true, true},
	{"023", "23", 23, 0, true, true},      // decimal, not octal
	{"-02.3", "-2.3", -23, 1, true, true}, // decimal, not octal
}

func TestDecGetString(t *testing.T) {
	z := new(inf.Dec)
	for i, test := range decStringTests {
		if !test.ok {
			continue
		}
		z.SetUnscaled(test.val)
		z.SetScale(test.scale)

		s := z.String()
		if s != test.out {
			t.Errorf("#%da got %s; want %s", i, s, test.out)
		}

		s = fmt.Sprintf("%d", z)
		if s != test.out {
			t.Errorf("#%db got %s; want %s", i, s, test.out)
		}
	}
}

func TestDecSetString(t *testing.T) {
	tmp := new(inf.Dec)
	for i, test := range decStringTests {
		if test.scale < 0 {
			// SetString only supports scale >= 0
			continue
		}
		// initialize to a non-zero value so that issues with parsing
		// 0 are detected
		tmp.Set(inf.NewDec(1234567890, 123))
		n1, ok1 := new(inf.Dec).SetString(test.in)
		n2, ok2 := tmp.SetString(test.in)
		expected := inf.NewDec(test.val, test.scale)
		if ok1 != test.ok || ok2 != test.ok {
			t.Errorf("#%d (input '%s') ok incorrect (should be %t)", i, test.in, test.ok)
			continue
		}
		if !ok1 {
			if n1 != nil {
				t.Errorf("#%d (input '%s') n1 != nil", i, test.in)
			}
			continue
		}
		if !ok2 {
			if n2 != nil {
				t.Errorf("#%d (input '%s') n2 != nil", i, test.in)
			}
			continue
		}

		if n1.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n1, test.val)
		}
		if n2.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n2, test.val)
		}
	}
}

func TestDecScan(t *testing.T) {
	tmp := new(inf.Dec)
	for i, test := range decStringTests {
		if test.scale < 0 {
			// SetString only supports scale >= 0
			continue
		}
		// initialize to a non-zero value so that issues with parsing
		// 0 are detected
		tmp.Set(inf.NewDec(1234567890, 123))
		n1, n2 := new(inf.Dec), tmp
		nn1, err1 := fmt.Sscan(test.in, n1)
		nn2, err2 := fmt.Sscan(test.in, n2)
		if !test.scanOk {
			if err1 == nil || err2 == nil {
				t.Errorf("#%d (input '%s') ok incorrect, should be %t", i, test.in, test.scanOk)
			}
			continue
		}
		expected := inf.NewDec(test.val, test.scale)
		if nn1 != 1 || err1 != nil || nn2 != 1 || err2 != nil {
			t.Errorf("#%d (input '%s') error %d %v, %d %v", i, test.in, nn1, err1, nn2, err2)
			continue
		}
		if n1.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n1, test.val)
		}
		if n2.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n2, test.val)
		}
	}
}

var decScanNextTests = []struct {
	in   string
	ok   bool
	next rune
}{
	{"", false, 0},
	{"a", false, 'a'},
	{"z", false, 'z'},
	{"+", false, 0},
	{"-", false, 0},
	{"g", false, 'g'},
	{".", false, 0},
	{".-0", false, '-'},
	{".+0", false, '+'},
	{"0b", true, 'b'},
	{"0x", true, 'x'},
	{"0xg", true, 'x'},
	{"0.0g", true, 'g'},
}

func TestDecScanNext(t *testing.T) {
	for i, test := range decScanNextTests {
		rdr := strings.NewReader(test.in)
		n1 := new(inf.Dec)
		nn1, _ := fmt.Fscan(rdr, n1)
		if (test.ok && nn1 == 0) || (!test.ok && nn1 > 0) {
			t.Errorf("#%d (input '%s') ok incorrect should be %t", i, test.in, test.ok)
			continue
		}
		r := rune(0)
		nn2, err := fmt.Fscanf(rdr, "%c", &r)
		if test.next != r {
			t.Errorf("#%d (input '%s') next incorrect, got %c should be %c, %d, %v", i, test.in, r, test.next, nn2, err)
		}
	}
}

var decGobEncodingTests = []string{
	"0",
	"1",
	"2",
	"10",
	"42",
	"1234567890",
	"298472983472983471903246121093472394872319615612417471234712061",
}

func TestDecGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for i, test := range decGobEncodingTests {
		for j := 0; j < 2; j++ {
			for k := inf.Scale(-5); k <= 5; k++ {
				medium.Reset() // empty buffer for each test case (in case of failures)
				stest := test
				if j != 0 {
					// negative numbers
					stest = "-" + test
				}
				var tx inf.Dec
				tx.SetString(stest)
				tx.SetScale(k) // test with positive, negative, and zero scale
				if err := enc.Encode(&tx); err != nil {
					t.Errorf("#%d%c: encoding failed: %s", i, 'a'+j, err)
				}
				var rx inf.Dec
				if err := dec.Decode(&rx); err != nil {
					t.Errorf("#%d%c: decoding failed: %s", i, 'a'+j, err)
				}
				if rx.Cmp(&tx) != 0 {
					t.Errorf("#%d%c: transmission failed: got %s want %s", i, 'a'+j, &rx, &tx)
				}
			}
		}
	}
}
