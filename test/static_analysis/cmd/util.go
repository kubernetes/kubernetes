/*
Copyright 2024 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"go/ast"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// identifierName returns the full name of an identifier.
func identifierName(v ast.Expr) string {
	if id, ok := v.(*ast.Ident); ok {
		return id.Name
	}
	if se, ok := v.(*ast.SelectorExpr); ok {
		return identifierName(se.X) + "." + identifierName(se.Sel)
	}
	return ""
}

// importAliasMap returns the mapping from pkg path to import alias.
func importAliasMap(imports []*ast.ImportSpec) map[string]string {
	m := map[string]string{}
	for _, im := range imports {
		var importAlias string
		if im.Name == nil {
			pathSegments := strings.Split(im.Path.Value, "/")
			importAlias = strings.Trim(pathSegments[len(pathSegments)-1], "\"")
		} else {
			importAlias = im.Name.String()
		}
		m[im.Path.Value] = importAlias
	}
	return m
}

// downloadFile will download from a given url to a file. It will
// write as it downloads (useful for large files).
func downloadFile(dest string, url string) error {
	fmt.Printf("download file from %s\n", url)
	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return err
	}
	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	// Create the file
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer func() {
		_ = out.Close()
	}()

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}
