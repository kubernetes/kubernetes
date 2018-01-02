package idn

import (
	"strings"
	"testing"
)

var testcases = [][2]string{
	{"", ""},
	{"a", "a"},
	{"a-b", "a-b"},
	{"a-b-c", "a-b-c"},
	{"abc", "abc"},
	{"я", "xn--41a"},
	{"zя", "xn--z-0ub"},
	{"яZ", "xn--z-zub"},
	{"а-я", "xn----7sb8g"},
	{"إختبار", "xn--kgbechtv"},
	{"آزمایشی", "xn--hgbk6aj7f53bba"},
	{"测试", "xn--0zwm56d"},
	{"測試", "xn--g6w251d"},
	{"испытание", "xn--80akhbyknj4f"},
	{"परीक्षा", "xn--11b5bs3a9aj6g"},
	{"δοκιμή", "xn--jxalpdlp"},
	{"테스트", "xn--9t4b11yi5a"},
	{"טעסט", "xn--deba0ad"},
	{"テスト", "xn--zckzah"},
	{"பரிட்சை", "xn--hlcj6aya9esc7a"},
	{"mamão-com-açúcar", "xn--mamo-com-acar-yeb1e6q"},
	{"σ", "xn--4xa"},
}

func TestEncodeDecodePunycode(t *testing.T) {
	for _, tst := range testcases {
		enc := encode([]byte(tst[0]))
		if string(enc) != tst[1] {
			t.Errorf("%s encodeded as %s but should be %s", tst[0], enc, tst[1])
		}
		dec := decode([]byte(tst[1]))
		if string(dec) != strings.ToLower(tst[0]) {
			t.Errorf("%s decoded as %s but should be %s", tst[1], dec, strings.ToLower(tst[0]))
		}
	}
}

func TestToFromPunycode(t *testing.T) {
	for _, tst := range testcases {
		// assert unicode.com == punycode.com
		full := ToPunycode(tst[0] + ".com")
		if full != tst[1]+".com" {
			t.Errorf("invalid result from string conversion to punycode, %s and should be %s.com", full, tst[1])
		}
		// assert punycode.punycode == unicode.unicode
		decoded := FromPunycode(tst[1] + "." + tst[1])
		if decoded != strings.ToLower(tst[0]+"."+tst[0]) {
			t.Errorf("invalid result from string conversion to punycode, %s and should be %s.%s", decoded, tst[0], tst[0])
		}
	}
}

func TestEncodeDecodeFinalPeriod(t *testing.T) {
	for _, tst := range testcases {
		// assert unicode.com. == punycode.com.
		full := ToPunycode(tst[0] + ".")
		if full != tst[1]+"." {
			t.Errorf("invalid result from string conversion to punycode when period added at the end, %#v and should be %#v", full, tst[1]+".")
		}
		// assert punycode.com. == unicode.com.
		decoded := FromPunycode(tst[1] + ".")
		if decoded != strings.ToLower(tst[0]+".") {
			t.Errorf("invalid result from string conversion to punycode when period added, %#v and should be %#v", decoded, tst[0]+".")
		}
		full = ToPunycode(tst[0])
		if full != tst[1] {
			t.Errorf("invalid result from string conversion to punycode when no period added at the end, %#v and should be %#v", full, tst[1]+".")
		}
		// assert punycode.com. == unicode.com.
		decoded = FromPunycode(tst[1])
		if decoded != strings.ToLower(tst[0]) {
			t.Errorf("invalid result from string conversion to punycode when no period added, %#v and should be %#v", decoded, tst[0]+".")
		}
	}
}

var invalidACEs = []string{
	"xn--*",
	"xn--",
	"xn---",
	"xn--a000000000",
}

func TestInvalidPunycode(t *testing.T) {
	for _, d := range invalidACEs {
		s := FromPunycode(d)
		if s != d {
			t.Errorf("Changed invalid name %s to %#v", d, s)
		}
	}
}

// You can verify the labels that are valid or not comparing to the Verisign
// website: http://mct.verisign-grs.com/
var invalidUnicodes = []string{
	"Σ",
	"ЯZ",
	"Испытание",
}

func TestInvalidUnicodes(t *testing.T) {
	for _, d := range invalidUnicodes {
		s := ToPunycode(d)
		if s != "" {
			t.Errorf("Changed invalid name %s to %#v", d, s)
		}
	}
}
