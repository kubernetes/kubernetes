package util

import (
	"fmt"
	"regexp"
	"testing"
)

func TestShortRegexpString(t *testing.T) {
	for _, test := range []struct {
		in  []string
		out string
	}{
		{[]string{"abc", "def"}, "abc|def"},
		{[]string{"abc", "def", "abc"}, "abc|def"},
		{[]string{"a", ""}, "a?"},
		{[]string{"a", "e"}, "a|e"},
		{[]string{"man", "men", "min"}, "m[aei]n"},
		{[]string{"man", "men", "min", "mon"}, "m[aeio]n"},
		{[]string{"man", "men", "min", "mn"}, "m[aei]?n"},
		{[]string{"man", "mbn", "mcn", "mdn", "mfn", "mgn"}, "m[a-dfg]n"},
		{[]string{"man", "mbn", "mcn", "mn", "mfn", "mgn"}, "m[abcfg]?n"},
		{[]string{"mbn", "mn", "mfn", "mcn", "man", "mgn"}, "m[abcfg]?n"},
		{[]string{"abccf", "abcde"}, "abc(cf|de)"},
		{[]string{"abcdefabcf", "abcdefabde", "abc"}, "abc(defab(cf|de))?"},
		{[]string{"css/bootstrap.css", "css/bootstrap.min.css", "css/bootstrap-theme.css", "css/bootstrap-theme.min.css"},
			`css/bootstrap(-theme)?(\.min)?\.css`},
		{[]string{`bootstrap-theme`, `main`, `normalize`, `bootstrap-theme.min`, `bootstrap.min`, `bootstrap`, `pygment_highlights`},
			`bootstrap(-theme)?(\.min)?|main|normalize|pygment_highlights`},
		{[]string{"css/bootstrap.css", "css/bootstrap.min.css", "css/bootstrap-theme.css", "css/bootstrap-theme.min.css", "css/main.css", "css/normalize.css", "css/pygment_highlights.css", "feed.xml", "img/avatar-icon.png", "js/bootstrap.js", "js/bootstrap.min.js", "js/jquery-1.11.2.min.js", "js/main.js"},
			`css/(bootstrap(-theme)?(\.min)?|main|normalize|pygment_highlights)\.css|feed\.xml|img/avatar-icon\.png|js/(bootstrap(\.min)?|jquery-1\.11\.2\.min|main)\.js`},
	} {
		input := fmt.Sprintf("%#q", test.in)
		got := ShortRegexpString(test.in...)
		if got != test.out {
			t.Errorf("expected:\n\t%#q,\ngot\n\t%#q for\n\t%s", test.out, got, input)
		}
		re, err := regexp.Compile("^(" + got + ")$")
		if err != nil {
			t.Errorf("regexp compile failure: %q\n\tfor %#q", err, got)
		} else {
			for _, str := range test.in {
				if !re.MatchString(str) {
					t.Errorf("regexp does not match input %#q:\n\t%#q", str, got)
				}
			}
		}
	}
}

func TestCommonPrefixes(t *testing.T) {
	for _, test := range []struct {
		in  []string
		out map[string]int
	}{
		{[]string{"abc", "def"}, nil},
		{[]string{"abcf", "abde"},
			map[string]int{"ab": 2}},
		{[]string{"abcf", "abcde", "abd"},
			map[string]int{"abc": 2, "ab": 3}},
	} {
		got := commonPrefixes(test.in, 2)
		want := test.out
		input := fmt.Sprintf("%v", test.in)
		if len(got) != len(want) {
			t.Errorf("expected: %v, got %v for\n\t%s", test.out, got, input)
			continue
		}
		for k, v := range want {
			if got[k].end-got[k].start != v {
				t.Errorf("expected: %v, got %v for\n\t%s", test.out, got, input)
				break
			}
		}
	}
}

func TestCommonSuffixes(t *testing.T) {
	for _, test := range []struct {
		in  []string
		out map[string]int
	}{
		{[]string{"abc", "def"}, nil},
		{[]string{"fcba", "edba"},
			map[string]int{"ba": 2}},
		{[]string{"fcba", "decba", "dba"},
			map[string]int{"cba": 2, "ba": 3}},
	} {
		// log.Print(test)
		got := commonSuffixes(test.in, 2)
		// log.Print(got)
		want := test.out
		input := fmt.Sprintf("%v", test.in)
		if len(got) != len(want) {
			t.Errorf("expected: %v, got %v for\n\t%s", test.out, got, input)
			continue
		}
		for k, v := range want {
			if got[k].end-got[k].start != v {
				t.Errorf("expected: %v, got %v for\n\t%s", test.out, got, input)
				break
			}
		}
	}
}

func BenchmarkShortRegexpString(b *testing.B) {
	input := []string{"css/bootstrap.css", "css/bootstrap.min.css", "css/bootstrap-theme.css", "css/bootstrap-theme.min.css", "css/main.css", "css/normalize.css", "css/pygment_highlights.css", "feed.xml", "img/avatar-icon.png", "js/bootstrap.js", "js/bootstrap.min.js", "js/jquery-1.11.2.min.js", "js/main.js"}

	for i := 0; i < b.N; i++ {
		arr := make([]string, len(input))
		copy(arr, input)
		_ = ShortRegexpString(arr...)
	}
}
