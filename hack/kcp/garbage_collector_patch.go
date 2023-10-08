/*
Copyright 2022 The KCP Authors.

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

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"strings"
)

/*
Process:

1. go run ./hack/kcp/garbage_collector_patch.go > pkg/controller/garbagecollector/garbagecollector_patch.go
(you may need to add -mod=readonly)

2. goimports -w pkg/controller/garbagecollector/garbagecollector_patch.go

3. reapply patch for kcp to pkg/controller/garbagecollector/garbagecollector_patch.go
*/

func main() {
	fileSet := token.NewFileSet()

	file, err := parser.ParseFile(fileSet, "pkg/controller/garbagecollector/garbagecollector.go", nil, parser.ParseComments)
	if err != nil {
		log.Fatal(err)
	}

	// n stores a reference to the node for the function declaration for Sync
	var n ast.Node

	ast.Inspect(file, func(node ast.Node) bool {
		switch x := node.(type) {
		case *ast.FuncDecl:
			if x.Name.Name == "Sync" {
				// Store the reference
				n = node
				// Stop further inspection
				return false
			}
		}

		// Continue recursing
		return true
	})

	startLine := fileSet.Position(n.Pos()).Line
	endLine := fileSet.Position(n.End()).Line

	// To preserve the comments from within the function body itself, we have to write out the entire file to a buffer,
	// then extract only the lines we care about (the function body).
	var buf bytes.Buffer
	if err := format.Node(&buf, fileSet, file); err != nil {
		log.Fatal(err)
	}

	// Convert the buffer to a slice of lines, so we can grab the portion we want
	var lines []string
	scanner := bufio.NewScanner(&buf)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	fmt.Println(`/*
Copyright 2022 The KCP Authors.

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

package garbagecollector
`)

	// Finally, print the line range we need
	fmt.Println(strings.Join(lines[startLine-1:endLine], "\n"))
}
