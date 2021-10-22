// +build go1.7

package ini

import (
	"reflect"
	"testing"
)

func TestIsSep(t *testing.T) {
	cases := []struct {
		b        []rune
		expected bool
	}{
		{
			b: []rune(``),
		},
		{
			b: []rune(`"wee"`),
		},
		{
			b:        []rune("["),
			expected: true,
		},
		{
			b:        []rune("]"),
			expected: true,
		},
	}

	for i, c := range cases {
		if e, a := c.expected, isSep(c.b); e != a {
			t.Errorf("%d: expected %t, but received %t", i+0, e, a)
		}
	}
}

func TestNewSep(t *testing.T) {
	cases := []struct {
		b             []rune
		expectedRead  int
		expectedError bool
		expectedToken Token
	}{
		{
			b:             []rune("["),
			expectedRead:  1,
			expectedToken: newToken(TokenSep, []rune("["), NoneType),
		},
		{
			b:             []rune("]"),
			expectedRead:  1,
			expectedToken: newToken(TokenSep, []rune("]"), NoneType),
		},
	}

	for i, c := range cases {
		tok, n, err := newSepToken(c.b)

		if e, a := c.expectedToken, tok; !reflect.DeepEqual(e, a) {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}

		if e, a := c.expectedRead, n; e != a {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}

		if e, a := c.expectedError, err != nil; e != a {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}
	}
}
