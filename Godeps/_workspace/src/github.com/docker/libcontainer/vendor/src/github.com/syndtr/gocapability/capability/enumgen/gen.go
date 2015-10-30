package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"strings"
)

const fileName = "enum.go"
const genName = "enum_gen.go"

type generator struct {
	buf  bytes.Buffer
	caps []string
}

func (g *generator) writeHeader() {
	g.buf.WriteString("// generated file; DO NOT EDIT - use go generate in directory with source\n")
	g.buf.WriteString("\n")
	g.buf.WriteString("package capability")
}

func (g *generator) writeStringFunc() {
	g.buf.WriteString("\n")
	g.buf.WriteString("func (c Cap) String() string {\n")
	g.buf.WriteString("switch c {\n")
	for _, cap := range g.caps {
		fmt.Fprintf(&g.buf, "case %s:\n", cap)
		fmt.Fprintf(&g.buf, "return \"%s\"\n", strings.ToLower(cap[4:]))
	}
	g.buf.WriteString("}\n")
	g.buf.WriteString("return \"unknown\"\n")
	g.buf.WriteString("}\n")
}

func (g *generator) writeListFunc() {
	g.buf.WriteString("\n")
	g.buf.WriteString("// List returns list of all supported capabilities\n")
	g.buf.WriteString("func List() []Cap {\n")
	g.buf.WriteString("return []Cap{\n")
	for _, cap := range g.caps {
		fmt.Fprintf(&g.buf, "%s,\n", cap)
	}
	g.buf.WriteString("}\n")
	g.buf.WriteString("}\n")
}

func main() {
	fs := token.NewFileSet()
	parsedFile, err := parser.ParseFile(fs, fileName, nil, 0)
	if err != nil {
		log.Fatal(err)
	}
	var caps []string
	for _, decl := range parsedFile.Decls {
		decl, ok := decl.(*ast.GenDecl)
		if !ok || decl.Tok != token.CONST {
			continue
		}
		for _, spec := range decl.Specs {
			vspec := spec.(*ast.ValueSpec)
			name := vspec.Names[0].Name
			if strings.HasPrefix(name, "CAP_") {
				caps = append(caps, name)
			}
		}
	}
	g := &generator{caps: caps}
	g.writeHeader()
	g.writeStringFunc()
	g.writeListFunc()
	src, err := format.Source(g.buf.Bytes())
	if err != nil {
		fmt.Println("generated invalid Go code")
		fmt.Println(g.buf.String())
		log.Fatal(err)
	}
	fi, err := os.Stat(fileName)
	if err != nil {
		log.Fatal(err)
	}
	if err := ioutil.WriteFile(genName, src, fi.Mode().Perm()); err != nil {
		log.Fatal(err)
	}
}
