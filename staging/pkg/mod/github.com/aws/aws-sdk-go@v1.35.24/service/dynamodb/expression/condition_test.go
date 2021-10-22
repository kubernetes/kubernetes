// +build go1.7

package expression

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// condErrorMode will help with error cases and checking error types
type condErrorMode string

const (
	noConditionError condErrorMode = ""
	// unsetCondition error will occur when BuildExpression is called on an empty
	// ConditionBuilder
	unsetCondition = "unset parameter: ConditionBuilder"
	// invalidOperand error will occur when an invalid OperandBuilder is used as
	// an argument
	invalidConditionOperand = "BuildOperand error"
)

//Compare
func TestCompare(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "name equal name",
			input: Name("foo").Equal(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c = $c",
			},
		},
		{
			name:  "value equal value",
			input: Value(5).Equal(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c = $c",
			},
		},
		{
			name:  "name size equal name size",
			input: Name("foo[1]").Size().Equal(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c = $c",
			},
		},
		{
			name:  "name not equal name",
			input: Name("foo").NotEqual(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c <> $c",
			},
		},
		{
			name:  "value not equal value",
			input: Value(5).NotEqual(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c <> $c",
			},
		},
		{
			name:  "name size not equal name size",
			input: Name("foo[1]").Size().NotEqual(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c <> $c",
			},
		},
		{
			name:  "name less than name",
			input: Name("foo").LessThan(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c < $c",
			},
		},
		{
			name:  "value less than value",
			input: Value(5).LessThan(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c < $c",
			},
		},
		{
			name:  "name size less than name size",
			input: Name("foo[1]").Size().LessThan(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c < $c",
			},
		},
		{
			name:  "name less than equal name",
			input: Name("foo").LessThanEqual(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c <= $c",
			},
		},
		{
			name:  "value less than equal value",
			input: Value(5).LessThanEqual(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c <= $c",
			},
		},
		{
			name:  "name size less than equal name size",
			input: Name("foo[1]").Size().LessThanEqual(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c <= $c",
			},
		},
		{
			name:  "name greater than name",
			input: Name("foo").GreaterThan(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c > $c",
			},
		},
		{
			name:  "value greater than value",
			input: Value(5).GreaterThan(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c > $c",
			},
		},
		{
			name:  "name size greater than name size",
			input: Name("foo[1]").Size().GreaterThan(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c > $c",
			},
		},
		{
			name:  "name greater than equal name",
			input: Name("foo").GreaterThanEqual(Name("bar")),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "$c >= $c",
			},
		},
		{
			name:  "value greater than equal value",
			input: Value(5).GreaterThanEqual(Value("bar")),
			expectedNode: exprNode{
				children: []exprNode{
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
								S: aws.String("bar"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c >= $c",
			},
		},
		{
			name:  "name size greater than equal name size",
			input: Name("foo[1]").Size().GreaterThanEqual(Name("bar").Size()),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n[1])",
					},
					{
						names:   []string{"bar"},
						fmtExpr: "size ($n)",
					},
				},
				fmtExpr: "$c >= $c",
			},
		},
		{
			name:  "invalid operand error Equal",
			input: Name("").Size().Equal(Value(5)),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error NotEqual",
			input: Name("").Size().NotEqual(Value(5)),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error LessThan",
			input: Name("").Size().LessThan(Value(5)),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error LessThanEqual",
			input: Name("").Size().LessThanEqual(Value(5)),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error GreaterThan",
			input: Name("").Size().GreaterThan(Value(5)),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error GreaterThanEqual",
			input: Name("").Size().GreaterThanEqual(Value(5)),
			err:   invalidConditionOperand,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestBuildCondition(t *testing.T) {
	cases := []struct {
		name     string
		input    ConditionBuilder
		expected exprNode
		err      condErrorMode
	}{
		{
			name:  "no match error",
			input: ConditionBuilder{},
			err:   unsetCondition,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestBoolCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic method and",
			input: Name("foo").Equal(Value(5)).And(Name("bar").Equal(Value("baz"))),
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
						fmtExpr: "$c = $c",
					},
				},
				fmtExpr: "($c) AND ($c)",
			},
		},
		{
			name:  "basic method or",
			input: Name("foo").Equal(Value(5)).Or(Name("bar").Equal(Value("baz"))),
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
						fmtExpr: "$c = $c",
					},
				},
				fmtExpr: "($c) OR ($c)",
			},
		},
		{
			name:  "variadic function and",
			input: And(Name("foo").Equal(Value(5)), Name("bar").Equal(Value("baz")), Name("qux").Equal(Value(true))),
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
						fmtExpr: "$c = $c",
					},
					{
						children: []exprNode{
							{
								names:   []string{"qux"},
								fmtExpr: "$n",
							},
							{
								values: []dynamodb.AttributeValue{
									{
										BOOL: aws.Bool(true),
									},
								},
								fmtExpr: "$v",
							},
						},
						fmtExpr: "$c = $c",
					},
				},
				fmtExpr: "($c) AND ($c) AND ($c)",
			},
		},
		{
			name:  "variadic function or",
			input: Or(Name("foo").Equal(Value(5)), Name("bar").Equal(Value("baz")), Name("qux").Equal(Value(true))),
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
						fmtExpr: "$c = $c",
					},
					{
						children: []exprNode{
							{
								names:   []string{"qux"},
								fmtExpr: "$n",
							},
							{
								values: []dynamodb.AttributeValue{
									{
										BOOL: aws.Bool(true),
									},
								},
								fmtExpr: "$v",
							},
						},
						fmtExpr: "$c = $c",
					},
				},
				fmtExpr: "($c) OR ($c) OR ($c)",
			},
		},
		{
			name:  "invalid operand error And",
			input: Name("").Size().GreaterThanEqual(Value(5)).And(Name("[5]").Between(Value(3), Value(9))),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error Or",
			input: Name("").Size().GreaterThanEqual(Value(5)).Or(Name("[5]").Between(Value(3), Value(9))),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestNotCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic method not",
			input: Name("foo").Equal(Value(5)).Not(),
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
				},
				fmtExpr: "NOT ($c)",
			},
		},
		{
			name:  "basic function not",
			input: Not(Name("foo").Equal(Value(5))),
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
				},
				fmtExpr: "NOT ($c)",
			},
		},
		{
			name:  "invalid operand error not",
			input: Name("").Size().GreaterThanEqual(Value(5)).Or(Name("[5]").Between(Value(3), Value(9))).Not(),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestBetweenCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic method between for name",
			input: Name("foo").Between(Value(5), Value(7)),
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c BETWEEN $c AND $c",
			},
		},
		{
			name:  "basic method between for value",
			input: Value(6).Between(Value(5), Value(7)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("6"),
							},
						},
						fmtExpr: "$v",
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c BETWEEN $c AND $c",
			},
		},
		{
			name:  "basic method between for size",
			input: Name("foo").Size().Between(Value(5), Value(7)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n)",
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c BETWEEN $c AND $c",
			},
		},
		{
			name:  "invalid operand error between",
			input: Name("[5]").Between(Value(3), Name("foo..bar")),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestInCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic method in for name",
			input: Name("foo").In(Value(5), Value(7)),
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c IN ($c, $c)",
			},
		},
		{
			name:  "basic method in for value",
			input: Value(6).In(Value(5), Value(7)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						values: []dynamodb.AttributeValue{
							{
								N: aws.String("6"),
							},
						},
						fmtExpr: "$v",
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c IN ($c, $c)",
			},
		},
		{
			name:  "basic method in for size",
			input: Name("foo").Size().In(Value(5), Value(7)),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "size ($n)",
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
								N: aws.String("7"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "$c IN ($c, $c)",
			},
		},
		{
			name:  "invalid operand error in",
			input: Name("[5]").In(Value(3), Name("foo..bar")),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestAttrExistsCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic attr exists",
			input: Name("foo").AttributeExists(),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "attribute_exists ($c)",
			},
		},
		{
			name:  "basic attr not exists",
			input: Name("foo").AttributeNotExists(),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
				},
				fmtExpr: "attribute_not_exists ($c)",
			},
		},
		{
			name:  "invalid operand error attr exists",
			input: AttributeExists(Name("")),
			err:   invalidConditionOperand,
		},
		{
			name:  "invalid operand error attr not exists",
			input: AttributeNotExists(Name("foo..bar")),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestAttrTypeCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "attr type String",
			input: Name("foo").AttributeType(String),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("S"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type String",
			input: Name("foo").AttributeType(String),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("S"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type StringSet",
			input: Name("foo").AttributeType(StringSet),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("SS"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type Number",
			input: Name("foo").AttributeType(Number),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("N"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type NumberSet",
			input: Name("foo").AttributeType(NumberSet),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("NS"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type Binary",
			input: Name("foo").AttributeType(Binary),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("B"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type BinarySet",
			input: Name("foo").AttributeType(BinarySet),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("BS"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type Boolean",
			input: Name("foo").AttributeType(Boolean),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("BOOL"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type Null",
			input: Name("foo").AttributeType(Null),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("NULL"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type List",
			input: Name("foo").AttributeType(List),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("L"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type Map",
			input: Name("foo").AttributeType(Map),
			expectedNode: exprNode{
				children: []exprNode{
					{
						names:   []string{"foo"},
						fmtExpr: "$n",
					},
					{
						values: []dynamodb.AttributeValue{
							{
								S: aws.String("M"),
							},
						},
						fmtExpr: "$v",
					},
				},
				fmtExpr: "attribute_type ($c, $c)",
			},
		},
		{
			name:  "attr type invalid operand",
			input: Name("").AttributeType(Map),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestBeginsWithCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic begins with",
			input: Name("foo").BeginsWith("bar"),
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
		{
			name:  "begins with invalid operand",
			input: Name("").BeginsWith("bar"),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestContainsCondition(t *testing.T) {
	cases := []struct {
		name         string
		input        ConditionBuilder
		expectedNode exprNode
		err          condErrorMode
	}{
		{
			name:  "basic contains",
			input: Name("foo").Contains("bar"),
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
				fmtExpr: "contains ($c, $c)",
			},
		},
		{
			name:  "contains invalid operand",
			input: Name("").Contains("bar"),
			err:   invalidConditionOperand,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.buildTree()
			if c.err != noConditionError {
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

func TestCompoundBuildCondition(t *testing.T) {
	cases := []struct {
		name      string
		inputCond ConditionBuilder
		expected  string
	}{
		{
			name: "and",
			inputCond: ConditionBuilder{
				conditionList: []ConditionBuilder{
					{},
					{},
					{},
					{},
				},
				mode: andCond,
			},
			expected: "($c) AND ($c) AND ($c) AND ($c)",
		},
		{
			name: "or",
			inputCond: ConditionBuilder{
				conditionList: []ConditionBuilder{
					{},
					{},
					{},
					{},
					{},
					{},
					{},
				},
				mode: orCond,
			},
			expected: "($c) OR ($c) OR ($c) OR ($c) OR ($c) OR ($c) OR ($c)",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			en, err := compoundBuildCondition(c.inputCond, exprNode{})
			if err != nil {
				t.Errorf("expect no error, got unexpected Error %q", err)
			}

			if e, a := c.expected, en.fmtExpr; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestInBuildCondition(t *testing.T) {
	cases := []struct {
		name      string
		inputCond ConditionBuilder
		expected  string
	}{
		{
			name: "in",
			inputCond: ConditionBuilder{
				operandList: []OperandBuilder{
					NameBuilder{},
					NameBuilder{},
					NameBuilder{},
					NameBuilder{},
					NameBuilder{},
					NameBuilder{},
					NameBuilder{},
				},
				mode: andCond,
			},
			expected: "$c IN ($c, $c, $c, $c, $c, $c)",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			en, err := inBuildCondition(c.inputCond, exprNode{})
			if err != nil {
				t.Errorf("expect no error, got unexpected Error %q", err)
			}

			if e, a := c.expected, en.fmtExpr; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

// If there is time implement mapEquals
