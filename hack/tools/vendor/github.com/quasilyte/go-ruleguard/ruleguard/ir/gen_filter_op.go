// +build generate

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"io/ioutil"
	"strings"
)

type opInfo struct {
	name      string
	comment   string
	valueType string
	flags     uint64
}

const (
	flagIsBinaryExpr uint64 = 1 << iota
	flagIsBasicLit
	flagHasVar
)

func main() {
	ops := []opInfo{
		{name: "Invalid"},

		{name: "Not", comment: "!$Args[0]"},

		// Binary expressions.
		{name: "And", comment: "$Args[0] && $Args[1]", flags: flagIsBinaryExpr},
		{name: "Or", comment: "$Args[0] || $Args[1]", flags: flagIsBinaryExpr},
		{name: "Eq", comment: "$Args[0] == $Args[1]", flags: flagIsBinaryExpr},
		{name: "Neq", comment: "$Args[0] != $Args[1]", flags: flagIsBinaryExpr},
		{name: "Gt", comment: "$Args[0] > $Args[1]", flags: flagIsBinaryExpr},
		{name: "Lt", comment: "$Args[0] < $Args[1]", flags: flagIsBinaryExpr},
		{name: "GtEq", comment: "$Args[0] >= $Args[1]", flags: flagIsBinaryExpr},
		{name: "LtEq", comment: "$Args[0] <= $Args[1]", flags: flagIsBinaryExpr},

		{name: "VarAddressable", comment: "m[$Value].Addressable", valueType: "string", flags: flagHasVar},
		{name: "VarPure", comment: "m[$Value].Pure", valueType: "string", flags: flagHasVar},
		{name: "VarConst", comment: "m[$Value].Const", valueType: "string", flags: flagHasVar},
		{name: "VarConstSlice", comment: "m[$Value].ConstSlice", valueType: "string", flags: flagHasVar},
		{name: "VarText", comment: "m[$Value].Text", valueType: "string", flags: flagHasVar},
		{name: "VarLine", comment: "m[$Value].Line", valueType: "string", flags: flagHasVar},
		{name: "VarValueInt", comment: "m[$Value].Value.Int()", valueType: "string", flags: flagHasVar},
		{name: "VarTypeSize", comment: "m[$Value].Type.Size", valueType: "string", flags: flagHasVar},
		{name: "VarTypeHasPointers", comment: "m[$Value].Type.HasPointers()", valueType: "string", flags: flagHasVar},

		{name: "VarFilter", comment: "m[$Value].Filter($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarNodeIs", comment: "m[$Value].Node.Is($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarObjectIs", comment: "m[$Value].Object.Is($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeIs", comment: "m[$Value].Type.Is($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeUnderlyingIs", comment: "m[$Value].Type.Underlying().Is($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeOfKind", comment: "m[$Value].Type.OfKind($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeUnderlyingOfKind", comment: "m[$Value].Type.Underlying().OfKind($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeConvertibleTo", comment: "m[$Value].Type.ConvertibleTo($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeAssignableTo", comment: "m[$Value].Type.AssignableTo($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTypeImplements", comment: "m[$Value].Type.Implements($Args[0])", valueType: "string", flags: flagHasVar},
		{name: "VarTextMatches", comment: "m[$Value].Text.Matches($Args[0])", valueType: "string", flags: flagHasVar},

		{name: "Deadcode", comment: "m.Deadcode()"},

		{name: "GoVersionEq", comment: "m.GoVersion().Eq($Value)", valueType: "string"},
		{name: "GoVersionLessThan", comment: "m.GoVersion().LessThan($Value)", valueType: "string"},
		{name: "GoVersionGreaterThan", comment: "m.GoVersion().GreaterThan($Value)", valueType: "string"},
		{name: "GoVersionLessEqThan", comment: "m.GoVersion().LessEqThan($Value)", valueType: "string"},
		{name: "GoVersionGreaterEqThan", comment: "m.GoVersion().GreaterEqThan($Value)", valueType: "string"},

		{name: "FileImports", comment: "m.File.Imports($Value)", valueType: "string"},
		{name: "FilePkgPathMatches", comment: "m.File.PkgPath.Matches($Value)", valueType: "string"},
		{name: "FileNameMatches", comment: "m.File.Name.Matches($Value)", valueType: "string"},

		{name: "FilterFuncRef", comment: "$Value holds a function name", valueType: "string"},

		{name: "String", comment: "$Value holds a string constant", valueType: "string", flags: flagIsBasicLit},
		{name: "Int", comment: "$Value holds an int64 constant", valueType: "int64", flags: flagIsBasicLit},

		{name: "RootNodeParentIs", comment: "m[`$$`].Node.Parent().Is($Args[0])"},
	}

	var buf bytes.Buffer

	buf.WriteString(`// Code generated "gen_filter_op.go"; DO NOT EDIT.` + "\n")
	buf.WriteString("\n")
	buf.WriteString("package ir\n")
	buf.WriteString("const (\n")

	for i, op := range ops {
		if strings.Contains(op.comment, "$Value") && op.valueType == "" {
			fmt.Printf("missing %s valueType\n", op.name)
		}
		if op.comment != "" {
			buf.WriteString("// " + op.comment + "\n")
		}
		if op.valueType != "" {
			buf.WriteString("// $Value type: " + op.valueType + "\n")
		}
		fmt.Fprintf(&buf, "Filter%sOp FilterOp = %d\n", op.name, i)
		buf.WriteString("\n")
	}
	buf.WriteString(")\n")

	buf.WriteString("var filterOpNames = map[FilterOp]string{\n")
	for _, op := range ops {
		fmt.Fprintf(&buf, "Filter%sOp: `%s`,\n", op.name, op.name)
	}
	buf.WriteString("}\n")

	buf.WriteString("var filterOpFlags = map[FilterOp]uint64{\n")
	for _, op := range ops {
		if op.flags == 0 {
			continue
		}
		parts := make([]string, 0, 1)
		if op.flags&flagIsBinaryExpr != 0 {
			parts = append(parts, "flagIsBinaryExpr")
		}
		if op.flags&flagIsBasicLit != 0 {
			parts = append(parts, "flagIsBasicLit")
		}
		if op.flags&flagHasVar != 0 {
			parts = append(parts, "flagHasVar")
		}
		fmt.Fprintf(&buf, "Filter%sOp: %s,\n", op.name, strings.Join(parts, " | "))
	}
	buf.WriteString("}\n")

	pretty, err := format.Source(buf.Bytes())
	if err != nil {
		panic(err)
	}

	if err := ioutil.WriteFile("filter_op.gen.go", pretty, 0644); err != nil {
		panic(err)
	}
}
