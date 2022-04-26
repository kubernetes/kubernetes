// +build gofuzz

package pattern

import (
	"go/ast"
	goparser "go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

var files []*ast.File

func init() {
	fset := token.NewFileSet()
	filepath.Walk("/usr/lib/go/src", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// XXX error handling
			panic(err)
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		f, err := goparser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}
		files = append(files, f)
		return nil
	})
}

func Fuzz(data []byte) int {
	p := &Parser{}
	pat, err := p.Parse(string(data))
	if err != nil {
		if strings.Contains(err.Error(), "internal error") {
			panic(err)
		}
		return 0
	}
	_ = pat.Root.String()

	for _, f := range files {
		Match(pat.Root, f)
	}
	return 1
}
