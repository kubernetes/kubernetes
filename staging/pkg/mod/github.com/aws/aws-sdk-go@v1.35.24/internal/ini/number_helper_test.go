// +build go1.7

package ini

import (
	"testing"
)

func TestNumberHelper(t *testing.T) {
	cases := []struct {
		b              []rune
		determineIndex int

		expectedExists   []bool
		expectedErrors   []bool
		expectedCorrects []bool
		expectedNegative bool
		expectedBase     int
	}{
		{
			b:              []rune("-10"),
			determineIndex: 0,
			expectedExists: []bool{
				false,
				false,
				false,
			},
			expectedErrors: []bool{
				false,
				false,
				false,
			},
			expectedCorrects: []bool{
				true,
				true,
				true,
			},
			expectedNegative: true,
			expectedBase:     10,
		},
		{
			b:              []rune("0x10"),
			determineIndex: 1,
			expectedExists: []bool{
				false,
				false,
				true,
				true,
			},
			expectedErrors: []bool{
				false,
				false,
				false,
				false,
			},
			expectedCorrects: []bool{
				true,
				true,
				true,
				true,
			},
			expectedNegative: false,
			expectedBase:     16,
		},
		{
			b:              []rune("0b101"),
			determineIndex: 1,
			expectedExists: []bool{
				false,
				false,
				true,
				true,
				true,
			},
			expectedErrors: []bool{
				false,
				false,
				false,
				false,
				false,
			},
			expectedCorrects: []bool{
				true,
				true,
				true,
				true,
				true,
			},
			expectedNegative: false,
			expectedBase:     2,
		},
		{
			b:              []rune("0o271"),
			determineIndex: 1,
			expectedExists: []bool{
				false,
				false,
				true,
				true,
				true,
			},
			expectedErrors: []bool{
				false,
				false,
				false,
				false,
				false,
			},
			expectedCorrects: []bool{
				true,
				true,
				true,
				true,
				true,
			},
			expectedNegative: false,
			expectedBase:     8,
		},
		{
			b:              []rune("99"),
			determineIndex: -1,
			expectedExists: []bool{
				false,
				false,
			},
			expectedErrors: []bool{
				false,
				false,
			},
			expectedCorrects: []bool{
				true,
				true,
			},
			expectedNegative: false,
			expectedBase:     10,
		},
		{
			b:              []rune("0o2o71"),
			determineIndex: 1,
			expectedExists: []bool{
				false,
				false,
				true,
				true,
				true,
				true,
			},
			expectedErrors: []bool{
				false,
				false,
				false,
				false,
				false,
				true,
			},
			expectedCorrects: []bool{
				true,
				true,
				true,
				false,
				true,
				true,
			},
			expectedNegative: false,
			expectedBase:     8,
		},
	}

	for _, c := range cases {
		helper := numberHelper{}

		for i := 0; i < len(c.b); i++ {
			if e, a := c.expectedExists[i], helper.Exists(); e != a {
				t.Errorf("expected %t, but received %t", e, a)
			}

			if i == c.determineIndex {
				if e, a := c.expectedErrors[i], helper.Determine(c.b[i]) != nil; e != a {
					t.Errorf("expected %t, but received %t", e, a)
				}
			} else {
				if e, a := c.expectedCorrects[i], helper.CorrectByte(c.b[i]); e != a {
					t.Errorf("expected %t, but received %t", e, a)
				}
			}
		}

		if e, a := c.expectedNegative, helper.IsNegative(); e != a {
			t.Errorf("expected %t, but received %t", e, a)
		}

		if e, a := c.expectedBase, helper.Base(); e != a {
			t.Errorf("expected %d, but received %d", e, a)
		}
	}
}
