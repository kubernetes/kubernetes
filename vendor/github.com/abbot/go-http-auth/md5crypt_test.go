package auth

import "testing"

func Test_MD5Crypt(t *testing.T) {
	test_cases := [][]string{
		{"apache", "$apr1$J.w5a/..$IW9y6DR0oO/ADuhlMF5/X1"},
		{"pass", "$1$YeNsbWdH$wvOF8JdqsoiLix754LTW90"},
	}
	for _, tc := range test_cases {
		e := NewMD5Entry(tc[1])
		result := MD5Crypt([]byte(tc[0]), e.Salt, e.Magic)
		if string(result) != tc[1] {
			t.Fatalf("MD5Crypt returned '%s' instead of '%s'", string(result), tc[1])
		}
		t.Logf("MD5Crypt: '%s' (%s%s$) -> %s", tc[0], e.Magic, e.Salt, result)
	}
}
