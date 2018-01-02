package govalidator

import (
	"reflect"
	"testing"
)

func TestContains(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected bool
	}{
		{"abacada", "", true},
		{"abacada", "ritir", false},
		{"abacada", "a", true},
		{"abacada", "aca", true},
	}
	for _, test := range tests {
		actual := Contains(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected Contains(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestMatches(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected bool
	}{
		{"123456789", "[0-9]+", true},
		{"abacada", "cab$", false},
		{"111222333", "((111|222|333)+)+", true},
		{"abacaba", "((123+]", false},
	}
	for _, test := range tests {
		actual := Matches(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected Matches(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestLeftTrim(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected string
	}{
		{"  \r\n\tfoo  \r\n\t   ", "", "foo  \r\n\t   "},
		{"010100201000", "01", "201000"},
	}
	for _, test := range tests {
		actual := LeftTrim(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected LeftTrim(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestRightTrim(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected string
	}{
		{"  \r\n\tfoo  \r\n\t   ", "", "  \r\n\tfoo"},
		{"010100201000", "01", "0101002"},
	}
	for _, test := range tests {
		actual := RightTrim(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected RightTrim(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestTrim(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected string
	}{
		{"  \r\n\tfoo  \r\n\t   ", "", "foo"},
		{"010100201000", "01", "2"},
		{"1234567890987654321", "1-8", "909"},
	}
	for _, test := range tests {
		actual := Trim(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected Trim(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

// This small example illustrate how to work with Trim function.
func ExampleTrim() {
	// Remove from left and right spaces and "\r", "\n", "\t" characters
	println(Trim("   \r\r\ntext\r   \t\n", "") == "text")
	// Remove from left and right characters that are between "1" and "8".
	// "1-8" is like full list "12345678".
	println(Trim("1234567890987654321", "1-8") == "909")
}

func TestWhiteList(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected string
	}{
		{"abcdef", "abc", "abc"},
		{"aaaaaaaaaabbbbbbbbbb", "abc", "aaaaaaaaaabbbbbbbbbb"},
		{"a1b2c3", "abc", "abc"},
		{"   ", "abc", ""},
		{"a3a43a5a4a3a2a23a4a5a4a3a4", "a-z", "aaaaaaaaaaaa"},
	}
	for _, test := range tests {
		actual := WhiteList(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected WhiteList(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

// This small example illustrate how to work with WhiteList function.
func ExampleWhiteList() {
	// Remove all characters from string ignoring characters between "a" and "z"
	println(WhiteList("a3a43a5a4a3a2a23a4a5a4a3a4", "a-z") == "aaaaaaaaaaaa")
}

func TestBlackList(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		expected string
	}{
		{"abcdef", "abc", "def"},
		{"aaaaaaaaaabbbbbbbbbb", "abc", ""},
		{"a1b2c3", "abc", "123"},
		{"   ", "abc", "   "},
		{"a3a43a5a4a3a2a23a4a5a4a3a4", "a-z", "34354322345434"},
	}
	for _, test := range tests {
		actual := BlackList(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected BlackList(%q,%q) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestStripLow(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   bool
		expected string
	}{
		{"foo\x00", false, "foo"},
		{"\x7Ffoo\x02", false, "foo"},
		{"\x01\x09", false, ""},
		{"foo\x0A\x0D", false, "foo"},
		{"perch\u00e9", false, "perch\u00e9"},
		{"\u20ac", false, "\u20ac"},
		{"\u2206\x0A", false, "\u2206"},
		{"foo\x0A\x0D", true, "foo\x0A\x0D"},
		{"\x03foo\x0A\x0D", true, "foo\x0A\x0D"},
	}
	for _, test := range tests {
		actual := StripLow(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected StripLow(%q,%t) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestReplacePattern(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   string
		param3   string
		expected string
	}{
		{"ab123ba", "[0-9]+", "aca", "abacaba"},
		{"abacaba", "[0-9]+", "aca", "abacaba"},
		{"httpftp://github.comio", "(ftp|io)", "", "http://github.com"},
		{"aaaaaaaaaa", "a", "", ""},
		{"http123123ftp://git534543hub.comio", "(ftp|io|[0-9]+)", "", "http://github.com"},
	}
	for _, test := range tests {
		actual := ReplacePattern(test.param1, test.param2, test.param3)
		if actual != test.expected {
			t.Errorf("Expected ReplacePattern(%q,%q,%q) to be %v, got %v", test.param1, test.param2, test.param3, test.expected, actual)
		}
	}
}

// This small example illustrate how to work with ReplacePattern function.
func ExampleReplacePattern() {
	// Replace in "http123123ftp://git534543hub.comio" following (pattern "(ftp|io|[0-9]+)"):
	// - Sequence "ftp".
	// - Sequence "io".
	// - Sequence of digits.
	// with empty string.
	println(ReplacePattern("http123123ftp://git534543hub.comio", "(ftp|io|[0-9]+)", "") == "http://github.com")
}

func TestEscape(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{`<img alt="foo&bar">`, "&lt;img alt=&#34;foo&amp;bar&#34;&gt;"},
	}
	for _, test := range tests {
		actual := Escape(test.param)
		if actual != test.expected {
			t.Errorf("Expected Escape(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestUnderscoreToCamelCase(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{"a_b_c", "ABC"},
		{"my_func", "MyFunc"},
		{"1ab_cd", "1abCd"},
	}
	for _, test := range tests {
		actual := UnderscoreToCamelCase(test.param)
		if actual != test.expected {
			t.Errorf("Expected UnderscoreToCamelCase(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestCamelCaseToUnderscore(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{"MyFunc", "my_func"},
		{"ABC", "a_b_c"},
		{"1B", "1_b"},
	}
	for _, test := range tests {
		actual := CamelCaseToUnderscore(test.param)
		if actual != test.expected {
			t.Errorf("Expected CamelCaseToUnderscore(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestReverse(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{"abc", "cba"},
		{"ｶﾀｶﾅ", "ﾅｶﾀｶ"},
	}
	for _, test := range tests {
		actual := Reverse(test.param)
		if actual != test.expected {
			t.Errorf("Expected Reverse(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestGetLines(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected []string
	}{
		{"abc", []string{"abc"}},
		{"a\nb\nc", []string{"a", "b", "c"}},
	}
	for _, test := range tests {
		actual := GetLines(test.param)
		if !reflect.DeepEqual(actual, test.expected) {
			t.Errorf("Expected GetLines(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestGetLine(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   int
		expected string
	}{
		{"abc", 0, "abc"},
		{"a\nb\nc", 0, "a"},
		{"abc", -1, ""},
		{"abacaba\n", 1, ""},
		{"abc", 3, ""},
	}
	for _, test := range tests {
		actual, _ := GetLine(test.param1, test.param2)
		if actual != test.expected {
			t.Errorf("Expected GetLine(%q, %d) to be %v, got %v", test.param1, test.param2, test.expected, actual)
		}
	}
}

func TestRemoveTags(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{"abc", "abc"},
		{"<!-- Test -->", ""},
		{"<div><div><p><a>Text</a></p></div></div>", "Text"},
		{`<a href="#">Link</a>`, "Link"},
	}
	for _, test := range tests {
		actual := RemoveTags(test.param)
		if actual != test.expected {
			t.Errorf("Expected RemoveTags(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestSafeFileName(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{"abc", "abc"},
		{"123456789     '_-?ASDF@£$%£%^é.html", "123456789-asdf.html"},
		{"ReadMe.md", "readme.md"},
		{"file:///c:/test.go", "test.go"},
		{"../../../Hello World!.txt", "hello-world.txt"},
	}
	for _, test := range tests {
		actual := SafeFileName(test.param)
		if actual != test.expected {
			t.Errorf("Expected SafeFileName(%q) to be %v, got %v", test.param, test.expected, actual)
		}
	}
}

func TestNormalizeEmail(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param    string
		expected string
	}{
		{`test@me.com`, `test@me.com`},
		{`some.name@gmail.com`, `somename@gmail.com`},
		{`some.name@googlemail.com`, `somename@gmail.com`},
		{`some.name+extension@gmail.com`, `somename@gmail.com`},
		{`some.name+extension@googlemail.com`, `somename@gmail.com`},
		{`some.name.middlename+extension@gmail.com`, `somenamemiddlename@gmail.com`},
		{`some.name.middlename+extension@googlemail.com`, `somenamemiddlename@gmail.com`},
		{`some.name.midd.lena.me.+extension@gmail.com`, `somenamemiddlename@gmail.com`},
		{`some.name.midd.lena.me.+extension@googlemail.com`, `somenamemiddlename@gmail.com`},
		{`some.name+extension@unknown.com`, `some.name+extension@unknown.com`},
		{`hans@m端ller.com`, `hans@m端ller.com`},
		{`hans`, ``},
	}
	for _, test := range tests {
		actual, err := NormalizeEmail(test.param)
		if actual != test.expected {
			t.Errorf("Expected NormalizeEmail(%q) to be %v, got %v, err %v", test.param, test.expected, actual, err)
		}
	}
}

func TestTruncate(t *testing.T) {
	t.Parallel()

	var tests = []struct {
		param1   string
		param2   int
		param3   string
		expected string
	}{
		{`Lorem ipsum dolor sit amet, consectetur adipiscing elit.`, 25, `...`, `Lorem ipsum dolor sit amet...`},
		{`Measuring programming progress by lines of code is like measuring aircraft building progress by weight.`, 35, ` new born babies!`, `Measuring programming progress by new born babies!`},
		{`Testestestestestestestestestest testestestestestestestestest`, 7, `...`, `Testestestestestestestestestest...`},
		{`Testing`, 7, `...`, `Testing`},
	}
	for _, test := range tests {
		actual := Truncate(test.param1, test.param2, test.param3)
		if actual != test.expected {
			t.Errorf("Expected Truncate(%q, %d, %q) to be %v, got %v", test.param1, test.param2, test.param3, test.expected, actual)
		}
	}
}
