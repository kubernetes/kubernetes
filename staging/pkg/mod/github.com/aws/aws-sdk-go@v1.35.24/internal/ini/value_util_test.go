// +build go1.7

package ini

import (
	"testing"
)

func TestStringValue(t *testing.T) {
	cases := []struct {
		b             []rune
		expectedRead  int
		expectedError bool
		expectedValue string
	}{
		{
			b:             []rune(`"foo"`),
			expectedRead:  5,
			expectedValue: `"foo"`,
		},
		{
			b:             []rune(`"123 !$_ 456 abc"`),
			expectedRead:  17,
			expectedValue: `"123 !$_ 456 abc"`,
		},
		{
			b:             []rune("foo"),
			expectedError: true,
		},
		{
			b:             []rune(` "foo"`),
			expectedError: true,
		},
	}

	for i, c := range cases {
		n, err := getStringValue(c.b)

		if e, a := c.expectedValue, string(c.b[:n]); e != a {
			t.Errorf("%d: expected %v, but received %v", i, e, a)
		}

		if e, a := c.expectedRead, n; e != a {
			t.Errorf("%d: expected %v, but received %v", i, e, a)
		}

		if e, a := c.expectedError, err != nil; e != a {
			t.Errorf("%d: expected %v, but received %v", i, e, a)
		}
	}
}

func TestBoolValue(t *testing.T) {
	cases := []struct {
		b             []rune
		expectedRead  int
		expectedError bool
		expectedValue string
	}{
		{
			b:             []rune("true"),
			expectedRead:  4,
			expectedValue: "true",
		},
		{
			b:             []rune("false"),
			expectedRead:  5,
			expectedValue: "false",
		},
		{
			b:             []rune(`"false"`),
			expectedError: true,
		},
	}

	for _, c := range cases {
		n, err := getBoolValue(c.b)

		if e, a := c.expectedValue, string(c.b[:n]); e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}

		if e, a := c.expectedRead, n; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}

		if e, a := c.expectedError, err != nil; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
	}
}

func TestNumericalValue(t *testing.T) {
	cases := []struct {
		b             []rune
		expectedRead  int
		expectedError bool
		expectedValue string
		expectedBase  int
	}{
		{
			b:             []rune("1.2"),
			expectedRead:  3,
			expectedValue: "1.2",
			expectedBase:  10,
		},
		{
			b:             []rune("123"),
			expectedRead:  3,
			expectedValue: "123",
			expectedBase:  10,
		},
		{
			b:             []rune("0x123A"),
			expectedRead:  6,
			expectedValue: "0x123A",
			expectedBase:  16,
		},
		{
			b:             []rune("0b101"),
			expectedRead:  5,
			expectedValue: "0b101",
			expectedBase:  2,
		},
		{
			b:             []rune("0o7"),
			expectedRead:  3,
			expectedValue: "0o7",
			expectedBase:  8,
		},
		{
			b:             []rune(`"123"`),
			expectedError: true,
		},
		{
			b:             []rune("0xo123"),
			expectedError: true,
		},
		{
			b:             []rune("123A"),
			expectedError: true,
		},
	}

	for i, c := range cases {
		base, n, err := getNumericalValue(c.b)

		if e, a := c.expectedValue, string(c.b[:n]); e != a {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}

		if e, a := c.expectedRead, n; e != a {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}

		if e, a := c.expectedError, err != nil; e != a {
			t.Errorf("%d: expected %v, but received %v", i+1, e, a)
		}

		if e, a := c.expectedBase, base; e != a {
			t.Errorf("%d: expected %d, but received %d", i+1, e, a)
		}
	}
}
