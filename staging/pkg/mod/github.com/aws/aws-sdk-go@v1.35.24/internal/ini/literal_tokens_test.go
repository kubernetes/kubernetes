// +build go1.7

package ini

import (
	"reflect"
	"testing"
)

func TestIsNumberValue(t *testing.T) {
	cases := []struct {
		name     string
		b        []rune
		expected bool
	}{
		{
			"integer",
			[]rune("123"),
			true,
		},
		{
			"negative integer",
			[]rune("-123"),
			true,
		},
		{
			"decimal",
			[]rune("123.456"),
			true,
		},
		{
			"small e exponent",
			[]rune("1e234"),
			true,
		},
		{
			"big E exponent",
			[]rune("1E234"),
			true,
		},
		{
			"error case exponent base 16",
			[]rune("1ea4"),
			false,
		},
		{
			"error case negative",
			[]rune("1-23"),
			false,
		},
		{
			"error case multiple negative",
			[]rune("-1-23"),
			false,
		},
		{
			"error case end negative",
			[]rune("123-"),
			false,
		},
		{
			"error case non-number",
			[]rune("a"),
			false,
		},
		{
			"utf8 whitespace",
			[]rune("00"),
			true,
		},
	}

	for i, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if e, a := c.expected, isNumberValue(c.b); e != a {
				t.Errorf("%d: expected %t, but received %t", i+1, e, a)
			}
		})
	}
}

// TODO: test errors
func TestNewLiteralToken(t *testing.T) {
	cases := []struct {
		name          string
		b             []rune
		expectedRead  int
		expectedToken Token
		expectedError bool
	}{
		{
			name:         "numbers",
			b:            []rune("123"),
			expectedRead: 3,
			expectedToken: newToken(TokenLit,
				[]rune("123"),
				IntegerType,
			),
		},
		{
			name:         "decimal",
			b:            []rune("123.456"),
			expectedRead: 7,
			expectedToken: newToken(TokenLit,
				[]rune("123.456"),
				DecimalType,
			),
		},
		{
			name:         "two numbers",
			b:            []rune("123 456"),
			expectedRead: 3,
			expectedToken: newToken(TokenLit,
				[]rune("123"),
				IntegerType,
			),
		},
		{
			name:         "number followed by alpha",
			b:            []rune("123 abc"),
			expectedRead: 3,
			expectedToken: newToken(TokenLit,
				[]rune("123"),
				IntegerType,
			),
		},
		{
			name:         "quoted string followed by number",
			b:            []rune(`"Hello" 123`),
			expectedRead: 7,
			expectedToken: newToken(TokenLit,
				[]rune("Hello"),
				QuotedStringType,
			),
		},
		{
			name:         "quoted string",
			b:            []rune(`"Hello World"`),
			expectedRead: 13,
			expectedToken: newToken(TokenLit,
				[]rune("Hello World"),
				QuotedStringType,
			),
		},
		{
			name:         "boolean true",
			b:            []rune("true"),
			expectedRead: 4,
			expectedToken: newToken(TokenLit,
				[]rune("true"),
				BoolType,
			),
		},
		{
			name:         "boolean false",
			b:            []rune("false"),
			expectedRead: 5,
			expectedToken: newToken(TokenLit,
				[]rune("false"),
				BoolType,
			),
		},
		{
			name: "utf8 whitespace",
			b: []rune("00"),
			expectedRead: 1,
			expectedToken: newToken(TokenLit,
				[]rune("0"),
				IntegerType,
			),
		},
		{
			name: "utf8 whitespace expr",
			b: []rune("0=00"),
			expectedRead: 1,
			expectedToken: newToken(TokenLit,
				[]rune("0"),
				StringType,
			),
		},
	}

	for i, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok, n, err := newLitToken(c.b)

			if e, a := c.expectedToken.ValueType, tok.ValueType; !reflect.DeepEqual(e, a) {
				t.Errorf("%d: expected %v, but received %v", i+1, e, a)
			}

			if e, a := c.expectedRead, n; e != a {
				t.Errorf("%d: expected %v, but received %v", i+1, e, a)
			}

			if e, a := c.expectedError, err != nil; e != a {
				t.Errorf("%d: expected %v, but received %v", i+1, e, a)
			}
		})
	}
}
