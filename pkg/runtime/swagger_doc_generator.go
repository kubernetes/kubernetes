/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package runtime

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"io"
	"reflect"
	"strings"
)

// Pair of strings. We keed the name of fields and the doc
type Pair struct {
	Name, Doc string
}

// KubeTypes is an array to represent all available types in a parsed file. [0] is for the type itself
type KubeTypes []Pair

func astFrom(filePath string) *doc.Package {
	fset := token.NewFileSet()
	m := make(map[string]*ast.File)

	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		fmt.Println(err)
		return nil
	}

	m[filePath] = f
	apkg, _ := ast.NewPackage(fset, m, nil, nil)

	return doc.New(apkg, "", 0)
}

func fmtRawDoc(rawDoc string) string {
	var buffer bytes.Buffer
	delPrevChar := func() {
		if buffer.Len() > 0 {
			buffer.Truncate(buffer.Len() - 1) // Delete the last " " or "\n"
		}
	}

	// Ignore all lines after ---
	rawDoc = strings.Split(rawDoc, "---")[0]

	for _, line := range strings.Split(rawDoc, "\n") {
		line = strings.TrimRight(line, " ")
		leading := strings.TrimLeft(line, " ")
		switch {
		case len(line) == 0: // Keep paragraphs
			delPrevChar()
			buffer.WriteString("\n\n")
		case strings.HasPrefix(leading, "TODO"): // Ignore one line TODOs
		case strings.HasPrefix(leading, "+"): // Ignore instructions to go2idl
		default:
			if strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
				delPrevChar()
				line = "\n" + line + "\n" // Replace it with newline. This is useful when we have a line with: "Example:\n\tJSON-someting..."
			} else {
				line += " "
			}
			buffer.WriteString(line)
		}
	}

	postDoc := strings.TrimRight(buffer.String(), "\n")
	postDoc = strings.Replace(postDoc, "\\\"", "\"", -1) // replace user's \" to "
	postDoc = strings.Replace(postDoc, "\"", "\\\"", -1) // Escape "
	postDoc = strings.Replace(postDoc, "\n", "\\n", -1)
	postDoc = strings.Replace(postDoc, "\t", "\\t", -1)

	return postDoc
}

// fieldName returns the name of the field as it should appear in JSON format
// "-" indicates that this field is not part of the JSON representation
func fieldName(field *ast.Field) string {
	jsonTag := ""
	if field.Tag != nil {
		jsonTag = reflect.StructTag(field.Tag.Value[1 : len(field.Tag.Value)-1]).Get("json") // Delete first and last quotation
		if strings.Contains(jsonTag, "inline") {
			return "-"
		}
	}

	jsonTag = strings.Split(jsonTag, ",")[0] // This can return "-"
	if jsonTag == "" {
		if field.Names != nil {
			return field.Names[0].Name
		}
		return field.Type.(*ast.Ident).Name
	}
	return jsonTag
}

func writeFuncHeader(b *buffer, structName string, indent int) {
	s := fmt.Sprintf("var map_%s = map[string]string {\n", structName)
	b.addLine(s, indent)
}

func writeFuncFooter(b *buffer, structName string, indent int) {
	b.addLine("}\n", indent) // Closes the map definition

	s := fmt.Sprintf("func (%s) SwaggerDoc() map[string]string {\n", structName)
	b.addLine(s, indent)
	s = fmt.Sprintf("return map_%s\n", structName)
	b.addLine(s, indent+1)
	b.addLine("}\n", indent) // Closes the function definition
}

func writeMapBody(b *buffer, kubeType []Pair, indent int) {
	format := "\"%s\": \"%s\",\n"
	for _, pair := range kubeType {
		s := fmt.Sprintf(format, pair.Name, pair.Doc)
		b.addLine(s, indent+2)
	}
}

// ParseDocumentationFrom gets all types' documentation and returns them as an
// array. Each type is again represented as an array (we have to use arrays as we
// need to be sure for the order of the fields). This function returns fields and
// struct definitions that have no documentation as {name, ""}.
func ParseDocumentationFrom(src string) []KubeTypes {
	var docForTypes []KubeTypes

	pkg := astFrom(src)

	for _, kubType := range pkg.Types {
		if structType, ok := kubType.Decl.Specs[0].(*ast.TypeSpec).Type.(*ast.StructType); ok {
			var ks KubeTypes
			ks = append(ks, Pair{kubType.Name, fmtRawDoc(kubType.Doc)})

			for _, field := range structType.Fields.List {
				if n := fieldName(field); n != "-" {
					fieldDoc := fmtRawDoc(field.Doc.Text())
					ks = append(ks, Pair{n, fieldDoc})
				}
			}
			docForTypes = append(docForTypes, ks)
		}
	}

	return docForTypes
}

// WriteSwaggerDocFunc writes a declaration of a function as a string. This function is used in
// Swagger as a documentation source for structs and theirs fields
func WriteSwaggerDocFunc(kubeTypes []KubeTypes, w io.Writer) error {
	for _, kubeType := range kubeTypes {
		structName := kubeType[0].Name
		kubeType[0].Name = ""

		// Ignore empty documentation
		docfulTypes := make(KubeTypes, 0, len(kubeType))
		for _, pair := range kubeType {
			if pair.Doc != "" {
				docfulTypes = append(docfulTypes, pair)
			}
		}

		if len(docfulTypes) == 0 {
			continue // If no fields and the struct have documentation, skip the function definition
		}

		indent := 0
		buffer := newBuffer()

		writeFuncHeader(buffer, structName, indent)
		writeMapBody(buffer, docfulTypes, indent)
		writeFuncFooter(buffer, structName, indent)
		buffer.addLine("\n", 0)

		if err := buffer.flushLines(w); err != nil {
			return err
		}
	}

	return nil
}

// VerifySwaggerDocsExist writes in a io.Writer a list of structs and fields that
// are missing of documentation.
func VerifySwaggerDocsExist(kubeTypes []KubeTypes, w io.Writer) (int, error) {
	missingDocs := 0
	buffer := newBuffer()

	for _, kubeType := range kubeTypes {
		structName := kubeType[0].Name
		if kubeType[0].Doc == "" {
			format := "Missing documentation for the struct itself: %s\n"
			s := fmt.Sprintf(format, structName)
			buffer.addLine(s, 0)
			missingDocs++
		}
		kubeType = kubeType[1:] // Skip struct definition

		for _, pair := range kubeType { // Iterate only the fields
			if pair.Doc == "" {
				format := "In struct: %s, field documentation is missing: %s\n"
				s := fmt.Sprintf(format, structName, pair.Name)
				buffer.addLine(s, 0)
				missingDocs++
			}
		}
	}

	if err := buffer.flushLines(w); err != nil {
		return -1, err
	}
	return missingDocs, nil
}
