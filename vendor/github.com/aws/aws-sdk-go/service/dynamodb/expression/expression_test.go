// +build go1.7

package expression

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

type exprErrorMode string

const (
	noExpressionError exprErrorMode = ""
	// invalidEscChar error will occer if the escape char '$' is either followed
	// by an unsupported character or if the escape char is the last character
	invalidEscChar = "invalid escape"
	// outOfRange error will occur if there are more escaped chars than there are
	// actual values to be aliased.
	outOfRange = "out of range"
	// invalidBuilderOperand error will occur if an invalid operand is used
	// as input for Build()
	invalidExpressionBuildOperand = "BuildOperand error"
	// unsetBuilder error will occur if Build() is called on an unset Builder
	unsetBuilder = "unset parameter: Builder"
	// unsetConditionBuilder error will occur if an unset ConditionBuilder is
	// used in WithCondition()
	unsetConditionBuilder = "unset parameter: ConditionBuilder"
)

func TestBuild(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected Expression
		err      exprErrorMode
	}{
		{
			name:  "condition",
			input: NewBuilder().WithCondition(Name("foo").Equal(Value(5))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					condition: "#0 = :0",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
				},
				valuesMap: map[string]*dynamodb.AttributeValue{
					":0": {
						N: aws.String("5"),
					},
				},
			},
		},
		{
			name:  "projection",
			input: NewBuilder().WithProjection(NamesList(Name("foo"), Name("bar"), Name("baz"))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					projection: "#0, #1, #2",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
					"#1": aws.String("bar"),
					"#2": aws.String("baz"),
				},
			},
		},
		{
			name:  "keyCondition",
			input: NewBuilder().WithKeyCondition(Key("foo").Equal(Value(5))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					keyCondition: "#0 = :0",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
				},
				valuesMap: map[string]*dynamodb.AttributeValue{
					":0": {
						N: aws.String("5"),
					},
				},
			},
		},
		{
			name:  "filter",
			input: NewBuilder().WithFilter(Name("foo").Equal(Value(5))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					filter: "#0 = :0",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
				},
				valuesMap: map[string]*dynamodb.AttributeValue{
					":0": {
						N: aws.String("5"),
					},
				},
			},
		},
		{
			name:  "update",
			input: NewBuilder().WithUpdate(Set(Name("foo"), (Value(5)))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					update: "SET #0 = :0\n",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
				},
				valuesMap: map[string]*dynamodb.AttributeValue{
					":0": {
						N: aws.String("5"),
					},
				},
			},
		},
		{
			name: "compound",
			input: NewBuilder().
				WithCondition(Name("foo").Equal(Value(5))).
				WithFilter(Name("bar").LessThan(Value(6))).
				WithProjection(NamesList(Name("foo"), Name("bar"), Name("baz"))).
				WithKeyCondition(Key("foo").Equal(Value(5))).
				WithUpdate(Set(Name("foo"), Value(5))),
			expected: Expression{
				expressionMap: map[expressionType]string{
					condition:    "#0 = :0",
					filter:       "#1 < :1",
					projection:   "#0, #1, #2",
					keyCondition: "#0 = :2",
					update:       "SET #0 = :3\n",
				},
				namesMap: map[string]*string{
					"#0": aws.String("foo"),
					"#1": aws.String("bar"),
					"#2": aws.String("baz"),
				},
				valuesMap: map[string]*dynamodb.AttributeValue{
					":0": {
						N: aws.String("5"),
					},
					":1": {
						N: aws.String("6"),
					},
					":2": {
						N: aws.String("5"),
					},
					":3": {
						N: aws.String("5"),
					},
				},
			},
		},
		{
			name:  "invalid Builder",
			input: NewBuilder().WithCondition(Name("").Equal(Value(5))),
			err:   invalidExpressionBuildOperand,
		},
		{
			name:  "unset Builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual, err := c.input.Build()
			if c.err != noExpressionError {
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

func TestCondition(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected *string
		err      exprErrorMode
	}{
		{
			name: "condition",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					condition: Name("foo").Equal(Value(5)),
				},
			},
			expected: aws.String("#0 = :0"),
		},
		{
			name:  "unset builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Condition()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestFilter(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected *string
		err      exprErrorMode
	}{
		{
			name: "filter",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					filter: Name("foo").Equal(Value(5)),
				},
			},
			expected: aws.String("#0 = :0"),
		},
		{
			name:  "unset builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Filter()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestProjection(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected *string
		err      exprErrorMode
	}{
		{
			name: "projection",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					projection: NamesList(Name("foo"), Name("bar"), Name("baz")),
				},
			},
			expected: aws.String("#0, #1, #2"),
		},
		{
			name:  "unset builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Projection()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestKeyCondition(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected *string
		err      exprErrorMode
	}{
		{
			name: "keyCondition",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					keyCondition: KeyConditionBuilder{
						operandList: []OperandBuilder{
							KeyBuilder{
								key: "foo",
							},
							ValueBuilder{
								value: 5,
							},
						},
						mode: equalKeyCond,
					},
				},
			},
			expected: aws.String("#0 = :0"),
		},
		{
			name:  "empty builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.KeyCondition()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestUpdate(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected *string
		err      exprErrorMode
	}{
		{
			name: "update",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					update: UpdateBuilder{
						operationList: map[operationMode][]operationBuilder{
							setOperation: {
								{
									name: NameBuilder{
										name: "foo",
									},
									value: ValueBuilder{
										value: 5,
									},
									mode: setOperation,
								},
							},
						},
					},
				},
			},
			expected: aws.String("SET #0 = :0\n"),
		},
		{
			name: "multiple sets",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					update: UpdateBuilder{
						operationList: map[operationMode][]operationBuilder{
							setOperation: {
								{
									name: NameBuilder{
										name: "foo",
									},
									value: ValueBuilder{
										value: 5,
									},
									mode: setOperation,
								},
								{
									name: NameBuilder{
										name: "bar",
									},
									value: ValueBuilder{
										value: 6,
									},
									mode: setOperation,
								},
								{
									name: NameBuilder{
										name: "baz",
									},
									value: ValueBuilder{
										value: 7,
									},
									mode: setOperation,
								},
							},
						},
					},
				},
			},
			expected: aws.String("SET #0 = :0, #1 = :1, #2 = :2\n"),
		},
		{
			name:  "unset builder",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Update()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestNames(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected map[string]*string
		err      exprErrorMode
	}{
		{
			name: "projection",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					projection: NamesList(Name("foo"), Name("bar"), Name("baz")),
				},
			},
			expected: map[string]*string{
				"#0": aws.String("foo"),
				"#1": aws.String("bar"),
				"#2": aws.String("baz"),
			},
		},
		{
			name: "aggregate",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					condition: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "foo",
							},
							ValueBuilder{
								value: 5,
							},
						},
						mode: equalCond,
					},
					filter: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "bar",
							},
							ValueBuilder{
								value: 6,
							},
						},
						mode: lessThanCond,
					},
					projection: ProjectionBuilder{
						names: []NameBuilder{
							{
								name: "foo",
							},
							{
								name: "bar",
							},
							{
								name: "baz",
							},
						},
					},
				},
			},
			expected: map[string]*string{
				"#0": aws.String("foo"),
				"#1": aws.String("bar"),
				"#2": aws.String("baz"),
			},
		},
		{
			name:  "unset",
			input: Builder{},
			err:   unsetBuilder,
		},
		{
			name:  "unset ConditionBuilder",
			input: NewBuilder().WithCondition(ConditionBuilder{}),
			err:   unsetConditionBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Names()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestValues(t *testing.T) {
	cases := []struct {
		name     string
		input    Builder
		expected map[string]*dynamodb.AttributeValue
		err      exprErrorMode
	}{
		{
			name: "condition",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					condition: Name("foo").Equal(Value(5)),
				},
			},
			expected: map[string]*dynamodb.AttributeValue{
				":0": {
					N: aws.String("5"),
				},
			},
		},
		{
			name: "aggregate",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					condition: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "foo",
							},
							ValueBuilder{
								value: 5,
							},
						},
						mode: equalCond,
					},
					filter: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "bar",
							},
							ValueBuilder{
								value: 6,
							},
						},
						mode: lessThanCond,
					},
					projection: ProjectionBuilder{
						names: []NameBuilder{
							{
								name: "foo",
							},
							{
								name: "bar",
							},
							{
								name: "baz",
							},
						},
					},
				},
			},
			expected: map[string]*dynamodb.AttributeValue{
				":0": {
					N: aws.String("5"),
				},
				":1": {
					N: aws.String("6"),
				},
			},
		},
		{
			name:  "unset",
			input: Builder{},
			err:   unsetBuilder,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.Build()
			if c.err != noExpressionError {
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
			}
			actual := expr.Values()
			if e, a := c.expected, actual; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestBuildChildTrees(t *testing.T) {
	cases := []struct {
		name              string
		input             Builder
		expectedaliasList aliasList
		expectedStringMap map[expressionType]string
		err               exprErrorMode
	}{
		{
			name: "aggregate",
			input: Builder{
				expressionMap: map[expressionType]treeBuilder{
					condition: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "foo",
							},
							ValueBuilder{
								value: 5,
							},
						},
						mode: equalCond,
					},
					filter: ConditionBuilder{
						operandList: []OperandBuilder{
							NameBuilder{
								name: "bar",
							},
							ValueBuilder{
								value: 6,
							},
						},
						mode: lessThanCond,
					},
					projection: ProjectionBuilder{
						names: []NameBuilder{
							{
								name: "foo",
							},
							{
								name: "bar",
							},
							{
								name: "baz",
							},
						},
					},
				},
			},
			expectedaliasList: aliasList{
				namesList: []string{"foo", "bar", "baz"},
				valuesList: []dynamodb.AttributeValue{
					{
						N: aws.String("5"),
					},
					{
						N: aws.String("6"),
					},
				},
			},
			expectedStringMap: map[expressionType]string{
				condition:  "#0 = :0",
				filter:     "#1 < :1",
				projection: "#0, #1, #2",
			},
		},
		{
			name:              "unset",
			input:             Builder{},
			expectedaliasList: aliasList{},
			expectedStringMap: map[expressionType]string{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actualAL, actualSM, err := c.input.buildChildTrees()
			if c.err != noExpressionError {
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
			}
			if e, a := c.expectedaliasList, actualAL; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
			if e, a := c.expectedStringMap, actualSM; !reflect.DeepEqual(a, e) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestBuildExpressionString(t *testing.T) {
	cases := []struct {
		name               string
		input              exprNode
		expectedNames      map[string]*string
		expectedValues     map[string]*dynamodb.AttributeValue
		expectedExpression string
		err                exprErrorMode
	}{
		{
			name: "basic name",
			input: exprNode{
				names:   []string{"foo"},
				fmtExpr: "$n",
			},

			expectedValues: map[string]*dynamodb.AttributeValue{},
			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
			},
			expectedExpression: "#0",
		},
		{
			name: "basic value",
			input: exprNode{
				values: []dynamodb.AttributeValue{
					{
						N: aws.String("5"),
					},
				},
				fmtExpr: "$v",
			},
			expectedNames: map[string]*string{},
			expectedValues: map[string]*dynamodb.AttributeValue{
				":0": {
					N: aws.String("5"),
				},
			},
			expectedExpression: ":0",
		},
		{
			name: "nested path",
			input: exprNode{
				names:   []string{"foo", "bar"},
				fmtExpr: "$n.$n",
			},

			expectedValues: map[string]*dynamodb.AttributeValue{},
			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
				"#1": aws.String("bar"),
			},
			expectedExpression: "#0.#1",
		},
		{
			name: "nested path with index",
			input: exprNode{
				names:   []string{"foo", "bar", "baz"},
				fmtExpr: "$n.$n[0].$n",
			},
			expectedValues: map[string]*dynamodb.AttributeValue{},
			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
				"#1": aws.String("bar"),
				"#2": aws.String("baz"),
			},
			expectedExpression: "#0.#1[0].#2",
		},
		{
			name: "basic size",
			input: exprNode{
				names:   []string{"foo"},
				fmtExpr: "size ($n)",
			},
			expectedValues: map[string]*dynamodb.AttributeValue{},
			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
			},
			expectedExpression: "size (#0)",
		},
		{
			name: "duplicate path name",
			input: exprNode{
				names:   []string{"foo", "foo"},
				fmtExpr: "$n.$n",
			},
			expectedValues: map[string]*dynamodb.AttributeValue{},
			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
			},
			expectedExpression: "#0.#0",
		},
		{
			name: "equal expression",
			input: exprNode{
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

			expectedNames: map[string]*string{
				"#0": aws.String("foo"),
			},
			expectedValues: map[string]*dynamodb.AttributeValue{
				":0": {
					N: aws.String("5"),
				},
			},
			expectedExpression: "#0 = :0",
		},
		{
			name: "missing char after $",
			input: exprNode{
				names:   []string{"foo", "foo"},
				fmtExpr: "$n.$",
			},
			err: invalidEscChar,
		},
		{
			name: "names out of range",
			input: exprNode{
				names:   []string{"foo"},
				fmtExpr: "$n.$n",
			},
			err: outOfRange,
		},
		{
			name: "values out of range",
			input: exprNode{
				fmtExpr: "$v",
			},
			err: outOfRange,
		},
		{
			name: "children out of range",
			input: exprNode{
				fmtExpr: "$c",
			},
			err: outOfRange,
		},
		{
			name: "invalid escape char",
			input: exprNode{
				fmtExpr: "$!",
			},
			err: invalidEscChar,
		},
		{
			name:               "unset exprNode",
			input:              exprNode{},
			expectedExpression: "",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			expr, err := c.input.buildExpressionString(&aliasList{})
			if c.err != noExpressionError {
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

				if e, a := c.expectedExpression, expr; !reflect.DeepEqual(a, e) {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestAliasValue(t *testing.T) {
	cases := []struct {
		name     string
		input    *aliasList
		expected string
		err      exprErrorMode
	}{
		{
			name:     "first item",
			input:    &aliasList{},
			expected: ":0",
		},
		{
			name: "fifth item",
			input: &aliasList{
				valuesList: []dynamodb.AttributeValue{
					{},
					{},
					{},
					{},
				},
			},
			expected: ":4",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			str, err := c.input.aliasValue(dynamodb.AttributeValue{})

			if c.err != noExpressionError {
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

				if e, a := c.expected, str; e != a {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}

func TestAliasPath(t *testing.T) {
	cases := []struct {
		name      string
		inputList *aliasList
		inputName string
		expected  string
		err       exprErrorMode
	}{
		{
			name:      "new unique item",
			inputList: &aliasList{},
			inputName: "foo",
			expected:  "#0",
		},
		{
			name: "duplicate item",
			inputList: &aliasList{
				namesList: []string{
					"foo",
					"bar",
				},
			},
			inputName: "foo",
			expected:  "#0",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			str, err := c.inputList.aliasPath(c.inputName)

			if c.err != noExpressionError {
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

				if e, a := c.expected, str; e != a {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}
