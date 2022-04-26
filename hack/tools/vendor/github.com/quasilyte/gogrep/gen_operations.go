//go:build main
// +build main

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"strings"
	"text/template"
)

var opPrototypes = []operationProto{
	{name: "Node", tag: "Node"},
	{name: "NamedNode", tag: "Node", valueIndex: "strings | wildcard name"},
	{name: "NodeSeq"},
	{name: "NamedNodeSeq", valueIndex: "strings | wildcard name"},
	{name: "OptNode"},
	{name: "NamedOptNode", valueIndex: "strings | wildcard name"},

	{name: "FieldNode", tag: "Node"},
	{name: "NamedFieldNode", tag: "Node", valueIndex: "strings | wildcard name"},

	{name: "MultiStmt", tag: "StmtList", args: "stmts...", example: "f(); g()"},
	{name: "MultiExpr", tag: "ExprList", args: "exprs...", example: "f(), g()"},
	{name: "MultiDecl", tag: "DeclList", args: "exprs...", example: "f(), g()"},

	{name: "End"},

	{name: "BasicLit", tag: "BasicLit", valueIndex: "ifaces | parsed literal value"},
	{name: "StrictIntLit", tag: "BasicLit", valueIndex: "strings | raw literal value"},
	{name: "StrictFloatLit", tag: "BasicLit", valueIndex: "strings | raw literal value"},
	{name: "StrictCharLit", tag: "BasicLit", valueIndex: "strings | raw literal value"},
	{name: "StrictStringLit", tag: "BasicLit", valueIndex: "strings | raw literal value"},
	{name: "StrictComplexLit", tag: "BasicLit", valueIndex: "strings | raw literal value"},

	{name: "Ident", tag: "Ident", valueIndex: "strings | ident name"},
	{name: "Pkg", tag: "Ident", valueIndex: "strings | package path"},

	{name: "IndexExpr", tag: "IndexExpr", args: "x expr"},

	{name: "SliceExpr", tag: "SliceExpr", args: "x"},
	{name: "SliceFromExpr", tag: "SliceExpr", args: "x from", example: "x[from:]"},
	{name: "SliceToExpr", tag: "SliceExpr", args: "x to", example: "x[:to]"},
	{name: "SliceFromToExpr", tag: "SliceExpr", args: "x from to", example: "x[from:to]"},
	{name: "SliceToCapExpr", tag: "SliceExpr", args: "x from cap", example: "x[:from:cap]"},
	{name: "SliceFromToCapExpr", tag: "SliceExpr", args: "x from to cap", example: "x[from:to:cap]"},

	{name: "FuncLit", tag: "FuncLit", args: "type block"},

	{name: "CompositeLit", tag: "CompositeLit", args: "elts...", example: "{elts...}"},
	{name: "TypedCompositeLit", tag: "CompositeLit", args: "typ elts...", example: "typ{elts...}"},

	{name: "SimpleSelectorExpr", tag: "SelectorExpr", args: "x", valueIndex: "strings | selector name"},
	{name: "SelectorExpr", tag: "SelectorExpr", args: "x sel"},
	{name: "TypeAssertExpr", tag: "TypeAssertExpr", args: "x typ"},
	{name: "TypeSwitchAssertExpr", tag: "TypeAssertExpr", args: "x"},

	{name: "StructType", tag: "StructType", args: "fields"},
	{name: "InterfaceType", tag: "StructType", args: "fields"},
	{name: "VoidFuncType", tag: "FuncType", args: "params"},
	{name: "FuncType", tag: "FuncType", args: "params results"},
	{name: "ArrayType", tag: "ArrayType", args: "length elem"},
	{name: "SliceType", tag: "ArrayType", args: "elem"},
	{name: "MapType", tag: "MapType", args: "key value"},
	{name: "ChanType", tag: "ChanType", args: "value", value: "ast.ChanDir | channel direction"},
	{name: "KeyValueExpr", tag: "KeyValueExpr", args: "key value"},

	{name: "Ellipsis", tag: "Ellipsis"},
	{name: "TypedEllipsis", tag: "Ellipsis", args: "type"},

	{name: "StarExpr", tag: "StarExpr", args: "x"},
	{name: "UnaryExpr", tag: "UnaryExpr", args: "x", value: "token.Token | unary operator"},
	{name: "BinaryExpr", tag: "BinaryExpr", args: "x y", value: "token.Token | binary operator"},
	{name: "ParenExpr", tag: "ParenExpr", args: "x"},

	{
		name:    "ArgList",
		args:    "exprs...",
		example: "1, 2, 3",
	},
	{
		name:    "SimpleArgList",
		note:    "Like ArgList, but pattern contains no $*",
		args:    "exprs[]",
		value:   "int | slice len",
		example: "1, 2, 3",
	},

	{name: "VariadicCallExpr", tag: "CallExpr", args: "fn args", example: "f(1, xs...)"},
	{name: "NonVariadicCallExpr", tag: "CallExpr", args: "fn args", example: "f(1, xs)"},
	{name: "CallExpr", tag: "CallExpr", args: "fn args", example: "f(1, xs) or f(1, xs...)"},

	{name: "AssignStmt", tag: "AssignStmt", args: "lhs rhs", value: "token.Token | ':=' or '='", example: "lhs := rhs()"},
	{name: "MultiAssignStmt", tag: "AssignStmt", args: "lhs... rhs...", value: "token.Token | ':=' or '='", example: "lhs1, lhs2 := rhs()"},

	{name: "BranchStmt", tag: "BranchStmt", args: "x", value: "token.Token | branch kind"},
	{name: "SimpleLabeledBranchStmt", tag: "BranchStmt", args: "x", valueIndex: "strings | label name", value: "token.Token | branch kind"},
	{name: "LabeledBranchStmt", tag: "BranchStmt", args: "label x", value: "token.Token | branch kind"},
	{name: "SimpleLabeledStmt", tag: "LabeledStmt", args: "x", valueIndex: "strings | label name"},
	{name: "LabeledStmt", tag: "LabeledStmt", args: "label x"},

	{name: "BlockStmt", tag: "BlockStmt", args: "body..."},
	{name: "ExprStmt", tag: "ExprStmt", args: "x"},

	{name: "GoStmt", tag: "GoStmt", args: "x"},
	{name: "DeferStmt", tag: "DeferStmt", args: "x"},

	{name: "SendStmt", tag: "SendStmt", args: "ch value"},

	{name: "EmptyStmt", tag: "EmptyStmt"},
	{name: "IncDecStmt", tag: "IncDecStmt", args: "x", value: "token.Token | '++' or '--'"},
	{name: "ReturnStmt", tag: "ReturnStmt", args: "results..."},

	{name: "IfStmt", tag: "IfStmt", args: "cond block", example: "if cond {}"},
	{name: "IfInitStmt", tag: "IfStmt", args: "init cond block", example: "if init; cond {}"},
	{name: "IfElseStmt", tag: "IfStmt", args: "cond block else", example: "if cond {} else ..."},
	{name: "IfInitElseStmt", tag: "IfStmt", args: "init cond block else", example: "if init; cond {} else ..."},
	{name: "IfNamedOptStmt", tag: "IfStmt", args: "block", valueIndex: "strings | wildcard name", example: "if $*x {}"},
	{name: "IfNamedOptElseStmt", tag: "IfStmt", args: "block else", valueIndex: "strings | wildcard name", example: "if $*x {} else ..."},

	{name: "SwitchStmt", tag: "SwitchStmt", args: "body...", example: "switch {}"},
	{name: "SwitchTagStmt", tag: "SwitchStmt", args: "tag body...", example: "switch tag {}"},
	{name: "SwitchInitStmt", tag: "SwitchStmt", args: "init body...", example: "switch init; {}"},
	{name: "SwitchInitTagStmt", tag: "SwitchStmt", args: "init tag body...", example: "switch init; tag {}"},

	{name: "SelectStmt", tag: "SelectStmt", args: "body..."},

	{name: "TypeSwitchStmt", tag: "TypeSwitchStmt", args: "x block", example: "switch x.(type) {}"},
	{name: "TypeSwitchInitStmt", tag: "TypeSwitchStmt", args: "init x block", example: "switch init; x.(type) {}"},

	{name: "CaseClause", tag: "CaseClause", args: "values... body..."},
	{name: "DefaultCaseClause", tag: "CaseClause", args: "body..."},

	{name: "CommClause", tag: "CommClause", args: "comm body..."},
	{name: "DefaultCommClause", tag: "CommClause", args: "body..."},

	{name: "ForStmt", tag: "ForStmt", args: "blocl", example: "for {}"},
	{name: "ForPostStmt", tag: "ForStmt", args: "post block", example: "for ; ; post {}"},
	{name: "ForCondStmt", tag: "ForStmt", args: "cond block", example: "for ; cond; {}"},
	{name: "ForCondPostStmt", tag: "ForStmt", args: "cond post block", example: "for ; cond; post {}"},
	{name: "ForInitStmt", tag: "ForStmt", args: "init block", example: "for init; ; {}"},
	{name: "ForInitPostStmt", tag: "ForStmt", args: "init post block", example: "for init; ; post {}"},
	{name: "ForInitCondStmt", tag: "ForStmt", args: "init cond block", example: "for init; cond; {}"},
	{name: "ForInitCondPostStmt", tag: "ForStmt", args: "init cond post block", example: "for init; cond; post {}"},

	{name: "RangeStmt", tag: "RangeStmt", args: "x block", example: "for range x {}"},
	{name: "RangeKeyStmt", tag: "RangeStmt", args: "key x block", value: "token.Token | ':=' or '='", example: "for key := range x {}"},
	{name: "RangeKeyValueStmt", tag: "RangeStmt", args: "key value x block", value: "token.Token | ':=' or '='", example: "for key, value := range x {}"},

	{name: "FieldList", args: "fields..."},
	{name: "UnnamedField", args: "typ", example: "type"},
	{name: "SimpleField", args: "typ", valueIndex: "strings | field name", example: "name type"},
	{name: "Field", args: "name typ", example: "$name type"},
	{name: "MultiField", args: "names... typ", example: "name1, name2 type"},

	{name: "ValueSpec", tag: "ValueSpec", args: "value"},
	{name: "ValueInitSpec", tag: "ValueSpec", args: "lhs... rhs...", example: "lhs = rhs"},
	{name: "TypedValueInitSpec", tag: "ValueSpec", args: "lhs... type rhs...", example: "lhs typ = rhs"},
	{name: "TypedValueSpec", tag: "ValueSpec", args: "lhs... type", example: "lhs typ"},

	{name: "TypeSpec", tag: "TypeSpec", args: "name type", example: "name type"},
	{name: "TypeAliasSpec", tag: "TypeSpec", args: "name type", example: "name = type"},

	{name: "FuncDecl", tag: "FuncDecl", args: "name type block"},
	{name: "MethodDecl", tag: "FuncDecl", args: "recv name type block"},
	{name: "FuncProtoDecl", tag: "FuncDecl", args: "name type"},
	{name: "MethodProtoDecl", tag: "FuncDecl", args: "recv name type"},

	{name: "DeclStmt", tag: "DeclStmt", args: "decl"},
	{name: "ConstDecl", tag: "GenDecl", args: "valuespecs..."},
	{name: "VarDecl", tag: "GenDecl", args: "valuespecs..."},
	{name: "TypeDecl", tag: "GenDecl", args: "typespecs..."},

	{name: "AnyImportDecl", tag: "GenDecl"},
	{name: "ImportDecl", tag: "GenDecl", args: "importspecs..."},

	{name: "EmptyPackage", tag: "File", args: "name"},
}

type operationProto struct {
	name       string
	value      string
	valueIndex string
	tag        string
	example    string
	args       string
	note       string
}

type operationInfo struct {
	Example            string
	Note               string
	Args               string
	Enum               uint8
	TagName            string
	Name               string
	ValueDoc           string
	ValueIndexDoc      string
	ExtraValueKindName string
	ValueKindName      string
	VariadicMap        uint64
	NumArgs            int
	SliceIndex         int
}

const stackUnchanged = ""

var fileTemplate = template.Must(template.New("operations.go").Parse(`// Code generated "gen_operations.go"; DO NOT EDIT.

package gogrep

import (
	"github.com/quasilyte/gogrep/nodetag"
)

//go:generate stringer -type=operation -trimprefix=op
type operation uint8

const (
	opInvalid operation = 0
{{ range .Operations }}
	// Tag: {{.TagName}}
	{{- if .Note}}{{print "\n"}}// {{.Note}}{{end}}
	{{- if .Args}}{{print "\n"}}// Args: {{.Args}}{{end}}
	{{- if .Example}}{{print "\n"}}// Example: {{.Example}}{{end}}
	{{- if .ValueDoc}}{{print "\n"}}// Value: {{.ValueDoc}}{{end}}
	{{- if .ValueIndexDoc}}{{print "\n"}}// ValueIndex: {{.ValueIndexDoc}}{{end}}
	op{{ .Name }} operation = {{.Enum}}
{{ end -}}
)

type operationInfo struct {
	Tag nodetag.Value
	NumArgs int
	ValueKind valueKind
	ExtraValueKind valueKind
	VariadicMap bitmap64
	SliceIndex int
}

var operationInfoTable = [256]operationInfo{
	opInvalid: {},

{{ range .Operations -}}
	op{{.Name}}: {
		Tag: nodetag.{{.TagName}}, 
		NumArgs: {{.NumArgs}},
		ValueKind: {{.ValueKindName}},
		ExtraValueKind: {{.ExtraValueKindName}},
		VariadicMap: {{.VariadicMap}}, // {{printf "%b" .VariadicMap}}
		SliceIndex: {{.SliceIndex}},
	},
{{ end }}
}
`))

func main() {
	operations := make([]operationInfo, len(opPrototypes))
	for i, proto := range opPrototypes {
		enum := uint8(i + 1)

		tagName := proto.tag
		if tagName == "" {
			tagName = "Unknown"
		}

		variadicMap := uint64(0)
		numArgs := 0
		sliceLenIndex := -1
		if proto.args != "" {
			args := strings.Split(proto.args, " ")
			numArgs = len(args)
			for i, arg := range args {
				isVariadic := strings.HasSuffix(arg, "...")
				if isVariadic {
					variadicMap |= 1 << i
				}
				if strings.HasSuffix(arg, "[]") {
					sliceLenIndex = i
				}
			}
		}

		extraValueKindName := "emptyValue"
		if proto.valueIndex != "" {
			parts := strings.Split(proto.valueIndex, " | ")
			typ := parts[0]
			switch typ {
			case "strings":
				extraValueKindName = "stringValue"
			case "ifaces":
				extraValueKindName = "ifaceValue"
			default:
				panic(fmt.Sprintf("%s: unexpected %s type", proto.name, typ))
			}
		}
		valueKindName := "emptyValue"
		if proto.value != "" {
			parts := strings.Split(proto.value, " | ")
			typ := parts[0]
			switch typ {
			case "token.Token":
				valueKindName = "tokenValue"
			case "ast.ChanDir":
				valueKindName = "chandirValue"
			case "int":
				valueKindName = "intValue"
			default:
				panic(fmt.Sprintf("%s: unexpected %s type", proto.name, typ))
			}
		}

		operations[i] = operationInfo{
			Example:            proto.example,
			Note:               proto.note,
			Args:               proto.args,
			Enum:               enum,
			TagName:            tagName,
			Name:               proto.name,
			ValueDoc:           proto.value,
			ValueIndexDoc:      proto.valueIndex,
			NumArgs:            numArgs,
			VariadicMap:        variadicMap,
			ExtraValueKindName: extraValueKindName,
			ValueKindName:      valueKindName,
			SliceIndex:         sliceLenIndex,
		}
	}

	var buf bytes.Buffer
	err := fileTemplate.Execute(&buf, map[string]interface{}{
		"Operations": operations,
	})
	if err != nil {
		log.Panicf("execute template: %v", err)
	}
	writeFile("operations.gen.go", buf.Bytes())
}

func writeFile(filename string, data []byte) {
	pretty, err := format.Source(data)
	if err != nil {
		log.Panicf("gofmt: %v", err)
	}
	if err := ioutil.WriteFile(filename, pretty, 0666); err != nil {
		log.Panicf("write %s: %v", filename, err)
	}
}
