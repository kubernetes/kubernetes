// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package xurls

import (
	"regexp"
	"testing"
)

type testCase struct {
	in   string
	want interface{}
}

func wantStr(in string, want interface{}) string {
	switch x := want.(type) {
	case string:
		return x
	case bool:
		if x {
			return in
		}
	}
	return ""
}

func doTest(t *testing.T, name string, re *regexp.Regexp, cases []testCase) {
	for _, c := range cases {
		got := re.FindString(c.in)
		want := wantStr(c.in, c.want)
		if got != want {
			t.Errorf(`%s.FindString("%s") got "%s", want "%s"`, name, c.in, got, want)
		}
	}
}

var constantTestCases = []testCase{
	{``, nil},
	{` `, nil},
	{`:`, nil},
	{`::`, nil},
	{`:::`, nil},
	{`::::`, nil},
	{`.`, nil},
	{`..`, nil},
	{`...`, nil},
	{`1.1`, nil},
	{`.1.`, nil},
	{`1.1.1`, nil},
	{`1:1`, nil},
	{`:1:`, nil},
	{`1:1:1`, nil},
	{`://`, nil},
	{`foo`, nil},
	{`foo:`, nil},
	{`mailto:`, nil},
	{`foo://`, nil},
	{`http://`, nil},
	{`http:// foo`, nil},
	{`http:// foo`, nil},
	{`:foo`, nil},
	{`://foo`, nil},
	{`foorandom:bar`, nil},
	{`foo.randombar`, nil},
	{`zzz.`, nil},
	{`.zzz`, nil},
	{`zzz.zzz`, nil},
	{`/some/path`, nil},
	{`rel/path`, nil},
	{`localhost`, nil},
	{`com`, nil},
	{`.com`, nil},
	{`com.`, nil},
	{`http`, nil},

	{`http://foo`, true},
	{`http://FOO`, true},
	{`http://FAÀ`, true},
	{`https://localhost`, true},
	{`git+https://localhost`, true},
	{`foo.bar://localhost`, true},
	{`foo-bar://localhost`, true},
	{`mailto:foo`, true},
	{`MAILTO:foo`, true},
	{`sms:123`, true},
	{`xmpp:foo@bar`, true},
	{`bitcoin:Addr23?amount=1&message=foo`, true},
	{`http://foo.com`, true},
	{`http://foo.co.uk`, true},
	{`http://foo.random`, true},
	{` http://foo.com/bar `, `http://foo.com/bar`},
	{` http://foo.com/bar more`, `http://foo.com/bar`},
	{`<http://foo.com/bar>`, `http://foo.com/bar`},
	{`<http://foo.com/bar>more`, `http://foo.com/bar`},
	{`.http://foo.com/bar.`, `http://foo.com/bar`},
	{`.http://foo.com/bar.more`, `http://foo.com/bar.more`},
	{`,http://foo.com/bar,`, `http://foo.com/bar`},
	{`,http://foo.com/bar,more`, `http://foo.com/bar,more`},
	{`(http://foo.com/bar)`, `http://foo.com/bar`},
	{`(http://foo.com/bar)more`, `http://foo.com/bar`},
	{`[http://foo.com/bar]`, `http://foo.com/bar`},
	{`[http://foo.com/bar]more`, `http://foo.com/bar`},
	{`'http://foo.com/bar'`, `http://foo.com/bar`},
	{`'http://foo.com/bar'more`, `http://foo.com/bar'more`},
	{`"http://foo.com/bar"`, `http://foo.com/bar`},
	{`http://a.b/a0/-+_&~*%=#@.,:;'?!|[]()a`, true},
	{`http://a.b/a0/$€¥`, true},
	{`http://✪foo.bar/pa✪th©more`, true},
	{`http://foo.bar/path/`, true},
	{`http://foo.bar/path-`, true},
	{`http://foo.bar/path+`, true},
	{`http://foo.bar/path_`, true},
	{`http://foo.bar/path&`, true},
	{`http://foo.bar/path~`, true},
	{`http://foo.bar/path*`, true},
	{`http://foo.bar/path%`, true},
	{`http://foo.bar/path=`, true},
	{`http://foo.bar/path#`, true},
	{`http://foo.bar/path.`, `http://foo.bar/path`},
	{`http://foo.bar/path,`, `http://foo.bar/path`},
	{`http://foo.bar/path:`, `http://foo.bar/path`},
	{`http://foo.bar/path;`, `http://foo.bar/path`},
	{`http://foo.bar/path'`, `http://foo.bar/path`},
	{`http://foo.bar/path?`, `http://foo.bar/path`},
	{`http://foo.bar/path!`, `http://foo.bar/path`},
	{`http://foo.bar/path@`, `http://foo.bar/path`},
	{`http://foo.bar/path|`, `http://foo.bar/path`},
	{`http://foo.bar/path<`, `http://foo.bar/path`},
	{`http://foo.bar/path<more`, `http://foo.bar/path`},
	{`http://foo.com/path_(more)`, true},
	{`(http://foo.com/path_(more))`, `http://foo.com/path_(more)`},
	{`http://foo.com/path_(even)-(more)`, true},
	{`http://foo.com/path_(even)(more)`, true},
	{`http://foo.com/path_(even_(nested))`, true},
	{`(http://foo.com/path_(even_(nested)))`, `http://foo.com/path_(even_(nested))`},
	{`http://foo.com/path_[more]`, true},
	{`[http://foo.com/path_[more]]`, `http://foo.com/path_[more]`},
	{`http://foo.com/path_[even]-[more]`, true},
	{`http://foo.com/path_[even][more]`, true},
	{`http://foo.com/path_[even_[nested]]`, true},
	{`[http://foo.com/path_[even_[nested]]]`, `http://foo.com/path_[even_[nested]]`},
	{`http://foo.com/path_{more}`, true},
	{`{http://foo.com/path_{more}}`, `http://foo.com/path_{more}`},
	{`http://foo.com/path_{even}-{more}`, true},
	{`http://foo.com/path_{even}{more}`, true},
	{`http://foo.com/path_{even_{nested}}`, true},
	{`{http://foo.com/path_{even_{nested}}}`, `http://foo.com/path_{even_{nested}}`},
	{`http://foo.com/path#fragment`, true},
	{`http://foo.com/emptyfrag#`, true},
	{`http://foo.com/spaced%20path`, true},
	{`http://foo.com/?p=spaced%20param`, true},
	{`http://test.foo.com/`, true},
	{`http://foo.com/path`, true},
	{`http://foo.com:8080/path`, true},
	{`http://1.1.1.1/path`, true},
	{`http://1080::8:800:200c:417a/path`, true},
	{`http://中国.中国/foo中国`, true},
	{`http://xn-foo.xn--p1acf/path`, true},
	{`what is http://foo.com?`, `http://foo.com`},
	{`go visit http://foo.com/path.`, `http://foo.com/path`},
	{`go visit http://foo.com/path...`, `http://foo.com/path`},
	{`what is http://foo.com/path?`, `http://foo.com/path`},
	{`the http://foo.com!`, `http://foo.com`},
	{`https://test.foo.bar/path?a=b`, `https://test.foo.bar/path?a=b`},
	{`ftp://user@foo.bar`, true},
	{`http://foo.com/@"style="color:red"onmouseover=func()`, `http://foo.com/`},
	{`http://foo.com/base64-bCBwbGVhcw==`, true},
}

func TestRegexes(t *testing.T) {
	doTest(t, "Relaxed", Relaxed, constantTestCases)
	doTest(t, "Strict", Strict, constantTestCases)
	doTest(t, "Relaxed", Relaxed, []testCase{
		{`foo.a`, nil},
		{`foo.com`, true},
		{`foo.com bar.com`, `foo.com`},
		{`foo.com-foo`, `foo.com`},
		{`foo.company`, true},
		{`foo.comrandom`, nil},
		{`foo.example`, true},
		{`foo.i2p`, true},
		{`foo.local`, true},
		{`foo.onion`, true},
		{`中国.中国`, true},
		{`中国.中国/foo中国`, true},
		{`foo.com/`, true},
		{`1.1.1.1`, true},
		{`10.50.23.250`, true},
		{`121.1.1.1`, true},
		{`255.1.1.1`, true},
		{`300.1.1.1`, nil},
		{`1.1.1.300`, nil},
		{`foo@1.2.3.4`, `1.2.3.4`},
		{`1080:0:0:0:8:800:200C:4171`, true},
		{`3ffe:2a00:100:7031::1`, true},
		{`1080::8:800:200c:417a`, true},
		{`foo.com:8080`, true},
		{`foo.com:8080/path`, true},
		{`test.foo.com`, true},
		{`test.foo.com/path`, true},
		{`test.foo.com/path/more/`, true},
		{`TEST.FOO.COM/PATH`, true},
		{`TEST.FÓO.COM/PÁTH`, true},
		{`foo.com/path_(more)`, true},
		{`foo.com/path_(even)_(more)`, true},
		{`foo.com/path_(more)/more`, true},
		{`foo.com/path_(more)/end)`, `foo.com/path_(more)/end`},
		{`www.foo.com`, true},
		{` foo.com/bar `, `foo.com/bar`},
		{` foo.com/bar more`, `foo.com/bar`},
		{`<foo.com/bar>`, `foo.com/bar`},
		{`<foo.com/bar>more`, `foo.com/bar`},
		{`,foo.com/bar.`, `foo.com/bar`},
		{`,foo.com/bar.more`, `foo.com/bar.more`},
		{`,foo.com/bar,`, `foo.com/bar`},
		{`,foo.com/bar,more`, `foo.com/bar,more`},
		{`(foo.com/bar)`, `foo.com/bar`},
		{`"foo.com/bar'`, `foo.com/bar`},
		{`"foo.com/bar'more`, `foo.com/bar'more`},
		{`"foo.com/bar"`, `foo.com/bar`},
		{`what is foo.com?`, `foo.com`},
		{`the foo.com!`, `foo.com`},

		{`foo@bar`, nil},
		{`foo@bar.a`, nil},
		{`foo@bar.com`, "bar.com"},
		{`foo@sub.bar.com`, "sub.bar.com"},
		{`foo@中国.中国`, "中国.中国"},
	})
	doTest(t, "Strict", Strict, []testCase{
		{`http:// foo.com`, nil},
		{`foo.a`, nil},
		{`foo.com`, nil},
		{`foo.com/`, nil},
		{`1.1.1.1`, nil},
		{`3ffe:2a00:100:7031::1`, nil},
		{`test.foo.com:8080/path`, nil},
		{`foo@bar.com`, nil},
	})
}

func TestStrictMatchingSchemeError(t *testing.T) {
	for _, c := range []struct {
		exp     string
		wantErr bool
	}{
		{`http://`, false},
		{`https?://`, false},
		{`http://|mailto:`, false},
		{`http://(`, true},
	} {
		_, err := StrictMatchingScheme(c.exp)
		if c.wantErr && err == nil {
			t.Errorf(`StrictMatchingScheme("%s") did not error as expected`, c.exp)
		} else if !c.wantErr && err != nil {
			t.Errorf(`StrictMatchingScheme("%s") unexpectedly errored`, c.exp)
		}
	}
}

func TestStrictMatchingScheme(t *testing.T) {
	strictMatching, _ := StrictMatchingScheme("http://|ftps?://|mailto:")
	doTest(t, "StrictMatchingScheme", strictMatching, []testCase{
		{`foo.com`, nil},
		{`foo@bar.com`, nil},
		{`http://foo`, true},
		{`Http://foo`, true},
		{`https://foo`, nil},
		{`ftp://foo`, true},
		{`ftps://foo`, true},
		{`mailto:foo`, true},
		{`MAILTO:foo`, true},
		{`sms:123`, nil},
	})
}

func bench(b *testing.B, re *regexp.Regexp, str string) {
	for i := 0; i < b.N; i++ {
		re.FindAllString(str, -1)
	}
}

func BenchmarkStrictEmpty(b *testing.B) {
	bench(b, Strict, "foo")
}

func BenchmarkStrictSingle(b *testing.B) {
	bench(b, Strict, "http://foo.foo foo.com")
}

func BenchmarkStrictMany(b *testing.B) {
	bench(b, Strict, ` foo bar http://foo.foo
	foo.com bitcoin:address ftp://
	xmpp:foo@bar.com`)
}

func BenchmarkRelaxedEmpty(b *testing.B) {
	bench(b, Relaxed, "foo")
}

func BenchmarkRelaxedSingle(b *testing.B) {
	bench(b, Relaxed, "http://foo.foo foo.com")
}

func BenchmarkRelaxedMany(b *testing.B) {
	bench(b, Relaxed, ` foo bar http://foo.foo
	foo.com bitcoin:address ftp://
	xmpp:foo@bar.com`)
}
