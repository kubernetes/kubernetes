// +build go1.7

package expression

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// opeErrorMode will help with error cases and checking error types
type opeErrorMode string

const (
	noOperandError opeErrorMode = ""
	// unsetName error will occur if an empty string is passed into NameBuilder
	unsetName = "unset parameter: NameBuilder"
	// invalidName error will occur if a nested name has an empty intermediary
	// attribute name (i.e. foo.bar..baz)
	invalidName = "invalid parameter: NameBuilder"
	// unsetKey error will occur if an empty string is passed into KeyBuilder
	unsetKey = "unset parameter: KeyBuilder"
)

func TestBuildOperand(t *testing.T) {
	cases := []struct {
		name     string
		input    OperandBuilder
		expected exprNode
		err      opeErrorMode
	}{
		{
			name:  "basic name",
			input: Name("foo"),
			expected: exprNode{
				names:   []string{"foo"},
				fmtExpr: "$n",
			},
		},
		{
			name:  "duplicate name name",
			input: Name("foo.foo"),
			expected: exprNode{
				names:   []string{"foo", "foo"},
				fmtExpr: "$n.$n",
			},
		},
		{
			name:  "basic value",
			input: Value(5),
			expected: exprNode{
				values: []dynamodb.AttributeValue{
					{
						N: aws.String("5"),
					},
				},
				fmtExpr: "$v",
			},
		},
		{
			name:  "dynamodb.AttributeValue as value",
			input: Value(dynamodb.AttributeValue{N: aws.String("5")}),
			expected: exprNode{
				values: []dynamodb.AttributeValue{
					{
						N: aws.String("5"),
					},
				},
				fmtExpr: "$v",
			},
		},
		{
			name:  "*dynamodb.AttributeValue as value",
			input: Value(&dynamodb.AttributeValue{N: aws.String("5")}),
			expected: exprNode{
				values: []dynamodb.AttributeValue{
					{
						N: aws.String("5"),
					},
				},
				fmtExpr: "$v",
			},
		},
		{
			name:  "nested name",
			input: Name("foo.bar"),
			expected: exprNode{
				names:   []string{"foo", "bar"},
				fmtExpr: "$n.$n",
			},
		},
		{
			name:  "nested name with index",
			input: Name("foo.bar[0].baz"),
			expected: exprNode{
				names:   []string{"foo", "bar", "baz"},
				fmtExpr: "$n.$n[0].$n",
			},
		},
		{
			name:  "basic size",
			input: Name("foo").Size(),
			expected: exprNode{
				names:   []string{"foo"},
				fmtExpr: "size ($n)",
			},
		},
		{
			name:  "key",
			input: Key("foo"),
			expected: exprNode{
				names:   []string{"foo"},
				fmtExpr: "$n",
			},
		},
		{
			name:     "unset key error",
			input:    Key(""),
			expected: exprNode{},
			err:      unsetKey,
		},
		{
			name:     "empty name error",
			input:    Name(""),
			expected: exprNode{},
			err:      unsetName,
		},
		{
			name:     "invalid name",
			input:    Name("foo..bar"),
			expected: exprNode{},
			err:      invalidName,
		},
		{
			name:     "invalid index",
			input:    Name("[foo]"),
			expected: exprNode{},
			err:      invalidName,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			operand, err := c.input.BuildOperand()

			if c.err != noOperandError {
				if err == nil {
					t.Errorf("expect error %q, got no error", c.err)
				} else {
					if e, a := string(c.err), err.Error(); !strings.Contains(a, e) {
						t.Errorf("expect %q error message to be in %q", e, a)
					}
				}
			} else {
				if err != nil {
					t.Errorf("expect no error, got unexpected Error %q", err)
				}

				if e, a := c.expected, operand.exprNode; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}
