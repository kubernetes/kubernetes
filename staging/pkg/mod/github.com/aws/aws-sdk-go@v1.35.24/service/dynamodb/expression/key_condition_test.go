// +build go1.7

package expression

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// keyCondErrorMode will help with error cases and checking error types
type keyCondErrorMode string

const (
	noKeyConditionError keyCondErrorMode = ""
	// unsetKeyCondition error will occur when buildTree() is called on an empty
	// KeyConditionBuilder
	unsetKeyCondition = "unset parameter: KeyConditionBuilder"
	// invalidKeyConditionOperand error will occur when an invalid OperandBuilder is used as
	// an argument
	invalidKeyConditionOperand = "BuildOperand error"
	// invalidKeyConditionFormat error will occur when the first key condition is not an equal
	// clause or if more then one And condition is provided
	invalidKeyConditionFormat = "buildKeyCondition error: invalid key condition constructed"
)

func TestKeyCompare(t *testing.T) {
	cases := []struct {
		name         string
		input        KeyConditionBuilder
		expectedNode exprNode
		err          keyCondErrorMode
	}{
		{
			name:  "key equal",
			input: Key("foo").Equal(Value(5)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c = $c",
			},
		},
		{
			name:  "key less than",
			input: Key("foo").LessThan(Value(5)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c < $c",
			},
		},
		{
			name:  "key less than equal",
			input: Key("foo").LessThanEqual(Value(5)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c <= $c",
			},
		},
		{
			name:  "key greater than",
			input: Key("foo").GreaterThan(Value(5)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c > $c",
			},
		},
		{
			name:  "key greater than equal",
			input: Key("foo").GreaterThanEqual(Value(5)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c >= $c",
			},
		},
		{
			name:  "unset KeyConditionBuilder",
			input: KeyConditionBuilder{},
			err:   unsetKeyCondition,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noKeyConditionError {
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

				if e, a := c.expectedNode, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestKeyBetween(t *testing.T) {
	cases := []struct {
		name         string
		input        KeyConditionBuilder
		expectedNode exprNode
		err          keyCondErrorMode
	}{
		{
			name:  "key between",
			input: Key("foo").Between(Value(5), Value(10)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("5"),
							},
						},
						fmtExpr: "$v",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("10"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c BETWEEN $c AND $c",
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noKeyConditionError {
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

				if e, a := c.expectedNode, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestKeyBeginsWith(t *testing.T) {
	cases := []struct {
		name         string
		input        KeyConditionBuilder
		expectedNode exprNode
		err          keyCondErrorMode
	}{
		{
			name:  "key begins with",
			input: Key("foo").BeginsWith("bar"),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "begins_with ($c, $c)",
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noKeyConditionError {
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

				if e, a := c.expectedNode, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestKeyAnd(t *testing.T) {
	cases := []struct {
		name         string
		input        KeyConditionBuilder
		expectedNode exprNode
		err          keyCondErrorMode
	}{
		{
			name:  "key and",
			input: Key("foo").Equal(Value(5)).And(Key("bar").BeginsWith("baz")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						children: []exprNode{
							{
								names:   []string{"foo"},
								fmtExpr: "$n",
							},
							{
								values: []dynamodb.AttributeValue{
									{
										N: aws.String("5"),
									},
								},
								fmtExpr: "$v",
							},
						},
						fmtExpr: "$c = $c",
					},
					{
						children: []exprNode{
							{
								names:   []string{"bar"},
								fmtExpr: "$n",
							},
							{
								values: []dynamodb.AttributeValue{
									{
										S: aws.String("baz"),
									},
								},
								fmtExpr: "$v",
							},
						},
						fmtExpr: "begins_with ($c, $c)",
					},
				},
				fmtExpr: "($c) AND ($c)",
			},
		},
		{
			name:  "first condition is not equal",
			input: Key("foo").LessThan(Value(5)).And(Key("bar").BeginsWith("baz")),
			err:   invalidKeyConditionFormat,
		},
		{
			name:  "more then one condition on key",
			input: Key("foo").Equal(Value(5)).And(Key("bar").Equal(Value(1)).And(Key("baz").BeginsWith("yar"))),
			err:   invalidKeyConditionFormat,
		},
		{
			name:  "operand error",
			input: Key("").Equal(Value("yikes")),
			err:   invalidKeyConditionOperand,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noKeyConditionError {
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

				if e, a := c.expectedNode, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestKeyConditionBuildChildNodes(t *testing.T) {
	cases := []struct {
		name     string
		input    KeyConditionBuilder
		expected []exprNode
		err      keyCondErrorMode
	}{
		{
			name:  "build child nodes",
			input: Key("foo").Equal(Value("bar")).And(Key("baz").LessThan(Value(10))),
			expected: []exprNode{
				{
					children: []exprNode{
						{
							names:   []string{"foo"},
							fmtExpr: "$n",
						},
						{
							values: []dynamodb.AttributeValue{
								{
									S: aws.String("bar"),
								},
							},
							fmtExpr: "$v",
						},
					},
					fmtExpr: "$c = $c",
				},
				{
					children: []exprNode{
						{
							names:   []string{"baz"},
							fmtExpr: "$n",
						},
						{
							values: []dynamodb.AttributeValue{
								{
									N: aws.String("10"),
								},
							},
							fmtExpr: "$v",
						},
					},
					fmtExpr: "$c < $c",
				},
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildChildNodes()
			if c.err != noKeyConditionError {
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

				if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %#v, got %#v", e, a)
				}
			}
		})
	}
}
