// +build go1.7

package ini

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"testing"
)

func TestParser(t *testing.T) {
	xID, _, _ := newLitToken([]rune("x = 1234"))
	s3ID, _, _ := newLitToken([]rune("s3 = 1234"))
	fooSlashes, _, _ := newLitToken([]rune("//foo"))

	regionID, _, _ := newLitToken([]rune("region"))
	regionLit, _, _ := newLitToken([]rune(`"us-west-2"`))
	regionNoQuotesLit, _, _ := newLitToken([]rune("us-west-2"))

	credentialID, _, _ := newLitToken([]rune("credential_source"))
	ec2MetadataLit, _, _ := newLitToken([]rune("Ec2InstanceMetadata"))

	outputID, _, _ := newLitToken([]rune("output"))
	outputLit, _, _ := newLitToken([]rune("json"))

	equalOp, _, _ := newOpToken([]rune("= 1234"))
	equalColonOp, _, _ := newOpToken([]rune(": 1234"))
	numLit, _, _ := newLitToken([]rune("1234"))
	defaultID, _, _ := newLitToken([]rune("default"))
	assumeID, _, _ := newLitToken([]rune("assumerole"))

	defaultProfileStmt := newSectionStatement(defaultID)
	assumeProfileStmt := newSectionStatement(assumeID)

	fooSlashesExpr := newExpression(fooSlashes)

	xEQ1234 := newEqualExpr(newExpression(xID), equalOp)
	xEQ1234.AppendChild(newExpression(numLit))
	xEQColon1234 := newEqualExpr(newExpression(xID), equalColonOp)
	xEQColon1234.AppendChild(newExpression(numLit))

	regionEQRegion := newEqualExpr(newExpression(regionID), equalOp)
	regionEQRegion.AppendChild(newExpression(regionLit))

	noQuotesRegionEQRegion := newEqualExpr(newExpression(regionID), equalOp)
	noQuotesRegionEQRegion.AppendChild(newExpression(regionNoQuotesLit))

	credEQExpr := newEqualExpr(newExpression(credentialID), equalOp)
	credEQExpr.AppendChild(newExpression(ec2MetadataLit))

	outputEQExpr := newEqualExpr(newExpression(outputID), equalOp)
	outputEQExpr.AppendChild(newExpression(outputLit))

	cases := []struct {
		name          string
		r             io.Reader
		expectedStack []AST
		expectedError bool
	}{
		{
			name: "semicolon comment",
			r:    bytes.NewBuffer([]byte(`;foo`)),
			expectedStack: []AST{
				newCommentStatement(newToken(TokenComment, []rune(";foo"), NoneType)),
			},
		},
		{
			name:          "0==0",
			r:             bytes.NewBuffer([]byte(`0==0`)),
			expectedError: true,
		},
		{
			name:          "0=:0",
			r:             bytes.NewBuffer([]byte(`0=:0`)),
			expectedError: true,
		},
		{
			name:          "0:=0",
			r:             bytes.NewBuffer([]byte(`0:=0`)),
			expectedError: true,
		},
		{
			name:          "0::0",
			r:             bytes.NewBuffer([]byte(`0::0`)),
			expectedError: true,
		},
		{
			name: "section with variable",
			r:    bytes.NewBuffer([]byte(`[ default ]x`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newExpression(xID),
			},
		},
		{
			name: "# comment",
			r:    bytes.NewBuffer([]byte(`# foo`)),
			expectedStack: []AST{
				newCommentStatement(newToken(TokenComment, []rune("# foo"), NoneType)),
			},
		},
		{
			name: "// not a comment",
			r:    bytes.NewBuffer([]byte(`//foo`)),
			expectedStack: []AST{
				fooSlashesExpr,
			},
		},
		{
			name: "multiple comments",
			r: bytes.NewBuffer([]byte(`;foo
					# baz
					`)),
			expectedStack: []AST{
				newCommentStatement(newToken(TokenComment, []rune(";foo"), NoneType)),
				newCommentStatement(newToken(TokenComment, []rune("# baz"), NoneType)),
			},
		},
		{
			name: "comment followed by skip state",
			r: bytes.NewBuffer([]byte(`;foo
			//foo
					# baz
					`)),
			expectedStack: []AST{
				newCommentStatement(newToken(TokenComment, []rune(";foo"), NoneType)),
			},
		},
		{
			name: "assignment",
			r:    bytes.NewBuffer([]byte(`x = 1234`)),
			expectedStack: []AST{
				newExprStatement(xEQ1234),
			},
		},
		{
			name: "assignment spaceless",
			r:    bytes.NewBuffer([]byte(`x=1234`)),
			expectedStack: []AST{
				newExprStatement(xEQ1234),
			},
		},
		{
			name: "assignment :",
			r:    bytes.NewBuffer([]byte(`x : 1234`)),
			expectedStack: []AST{
				newExprStatement(xEQColon1234),
			},
		},
		{
			name: "assignment : no spaces",
			r:    bytes.NewBuffer([]byte(`x:1234`)),
			expectedStack: []AST{
				newExprStatement(xEQColon1234),
			},
		},
		{
			name: "section expression",
			r:    bytes.NewBuffer([]byte(`[ default ]`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
			},
		},
		{
			name: "section expression no spaces",
			r:    bytes.NewBuffer([]byte(`[default]`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
			},
		},
		{
			name: "section statement",
			r: bytes.NewBuffer([]byte(`[default]
							region="us-west-2"`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newExprStatement(regionEQRegion),
			},
		},
		{
			name: "complex section statement",
			r: bytes.NewBuffer([]byte(`[default]
		region = us-west-2
		credential_source = Ec2InstanceMetadata
		output = json

		[assumerole]
		output = json
		region = us-west-2
				`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newExprStatement(noQuotesRegionEQRegion),
				newExprStatement(credEQExpr),
				newExprStatement(outputEQExpr),
				newCompletedSectionStatement(
					assumeProfileStmt,
				),
				newExprStatement(outputEQExpr),
				newExprStatement(noQuotesRegionEQRegion),
			},
		},
		{
			name: "complex section statement with nested params",
			r: bytes.NewBuffer([]byte(`[default]
s3 =
	foo=bar
	bar=baz
region = us-west-2
credential_source = Ec2InstanceMetadata
output = json

[assumerole]
output = json
region = us-west-2
				`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newSkipStatement(newEqualExpr(newExpression(s3ID), equalOp)),
				newExprStatement(noQuotesRegionEQRegion),
				newExprStatement(credEQExpr),
				newExprStatement(outputEQExpr),
				newCompletedSectionStatement(
					assumeProfileStmt,
				),
				newExprStatement(outputEQExpr),
				newExprStatement(noQuotesRegionEQRegion),
			},
		},
		{
			name: "complex section statement",
			r: bytes.NewBuffer([]byte(`[default]
region = us-west-2
credential_source = Ec2InstanceMetadata
s3 =
	foo=bar
	bar=baz
output = json

[assumerole]
output = json
region = us-west-2
				`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newExprStatement(noQuotesRegionEQRegion),
				newExprStatement(credEQExpr),
				newSkipStatement(newEqualExpr(newExpression(s3ID), equalOp)),
				newExprStatement(outputEQExpr),
				newCompletedSectionStatement(
					assumeProfileStmt,
				),
				newExprStatement(outputEQExpr),
				newExprStatement(noQuotesRegionEQRegion),
			},
		},
		{
			name: "missing section statement",
			r: bytes.NewBuffer([]byte(
				`[default]
s3 =
[assumerole]
output = json
				`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newSkipStatement(newEqualExpr(newExpression(s3ID), equalOp)),
				newCompletedSectionStatement(
					assumeProfileStmt,
				),
				newExprStatement(outputEQExpr),
			},
		},
		{
			name: "missing right hand expression in the last statement in the file",
			r: bytes.NewBuffer([]byte(
				`[default]
region = us-west-2
s3 =`)),
			expectedStack: []AST{
				newCompletedSectionStatement(
					defaultProfileStmt,
				),
				newExprStatement(noQuotesRegionEQRegion),
			},
		},
	}

	for i, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			stack, err := ParseAST(c.r)

			if e, a := c.expectedError, err != nil; e != a {
				t.Errorf("%d: expected %t, but received %t with error %v", i, e, a, err)
			}

			if e, a := len(c.expectedStack), len(stack); e != a {
				t.Errorf("expected same length %d, but received %d", e, a)
			}

			if e, a := c.expectedStack, stack; !reflect.DeepEqual(e, a) {
				buf := bytes.Buffer{}
				buf.WriteString("expected:\n")
				for j := 0; j < len(e); j++ {
					buf.WriteString(fmt.Sprintf("\t%d: %v\n", j, e[j]))
				}

				buf.WriteString("\nreceived:\n")
				for j := 0; j < len(a); j++ {
					buf.WriteString(fmt.Sprintf("\t%d: %v\n", j, a[j]))
				}

				t.Errorf("%s", buf.String())
			}
		})
	}
}
