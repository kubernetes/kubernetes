package ngfunc

import (
	"fmt"
	"go/types"
	"strings"

	"github.com/gostaticanalysis/analysisutil"
	"golang.org/x/tools/go/analysis"
)

var errNotFound = fmt.Errorf("function not found")

func typeFuncs(pass *analysis.Pass, funcs []string) []*types.Func {
	fs := make([]*types.Func, 0, len(funcs))

	for _, fn := range funcs {
		f, err := typeFunc(pass, fn)
		if err != nil {
			continue
		}

		fs = append(fs, f)
	}

	return fs
}

func typeFunc(pass *analysis.Pass, funcName string) (*types.Func, error) {
	ss := strings.Split(strings.TrimSpace(funcName), ".")

	switch len(ss) {
	case 2:
		// package function: pkgname.Func
		f, ok := analysisutil.ObjectOf(pass, ss[0], ss[1]).(*types.Func)
		if !ok || f == nil {
			return nil, errNotFound
		}

		return f, nil
	case 3:
		// method: (*pkgname.Type).Method
		pkgname := strings.TrimLeft(ss[0], "(")
		typename := strings.TrimRight(ss[1], ")")

		if pkgname != "" && pkgname[0] == '*' {
			pkgname = pkgname[1:]
			typename = "*" + typename
		}

		typ := analysisutil.TypeOf(pass, pkgname, typename)
		if typ == nil {
			return nil, errNotFound
		}

		m := analysisutil.MethodOf(typ, ss[2])
		if m == nil {
			return nil, errNotFound
		}

		return m, nil
	}

	return nil, errNotFound
}
