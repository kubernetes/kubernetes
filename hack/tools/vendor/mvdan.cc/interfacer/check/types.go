// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package check

import (
	"bytes"
	"fmt"
	"go/types"
	"sort"
	"strings"
)

type methoder interface {
	NumMethods() int
	Method(int) *types.Func
}

func methoderFuncMap(m methoder, skip bool) map[string]string {
	ifuncs := make(map[string]string, m.NumMethods())
	for i := 0; i < m.NumMethods(); i++ {
		f := m.Method(i)
		if !f.Exported() {
			if skip {
				continue
			}
			return nil
		}
		sign := f.Type().(*types.Signature)
		ifuncs[f.Name()] = signString(sign)
	}
	return ifuncs
}

func typeFuncMap(t types.Type) map[string]string {
	switch x := t.(type) {
	case *types.Pointer:
		return typeFuncMap(x.Elem())
	case *types.Named:
		u := x.Underlying()
		if types.IsInterface(u) {
			return typeFuncMap(u)
		}
		return methoderFuncMap(x, true)
	case *types.Interface:
		return methoderFuncMap(x, false)
	default:
		return nil
	}
}

func funcMapString(iface map[string]string) string {
	fnames := make([]string, 0, len(iface))
	for fname := range iface {
		fnames = append(fnames, fname)
	}
	sort.Strings(fnames)
	var b bytes.Buffer
	for i, fname := range fnames {
		if i > 0 {
			fmt.Fprint(&b, "; ")
		}
		fmt.Fprint(&b, fname, iface[fname])
	}
	return b.String()
}

func tupleJoin(buf *bytes.Buffer, t *types.Tuple) {
	buf.WriteByte('(')
	for i := 0; i < t.Len(); i++ {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(t.At(i).Type().String())
	}
	buf.WriteByte(')')
}

// signString is similar to Signature.String(), but it ignores
// param/result names.
func signString(sign *types.Signature) string {
	var buf bytes.Buffer
	tupleJoin(&buf, sign.Params())
	tupleJoin(&buf, sign.Results())
	return buf.String()
}

func interesting(t types.Type) bool {
	switch x := t.(type) {
	case *types.Interface:
		return x.NumMethods() > 1
	case *types.Named:
		if u := x.Underlying(); types.IsInterface(u) {
			return interesting(u)
		}
		return x.NumMethods() >= 1
	case *types.Pointer:
		return interesting(x.Elem())
	default:
		return false
	}
}

func anyInteresting(params *types.Tuple) bool {
	for i := 0; i < params.Len(); i++ {
		t := params.At(i).Type()
		if interesting(t) {
			return true
		}
	}
	return false
}

func fromScope(scope *types.Scope) (ifaces map[string]string, funcs map[string]bool) {
	ifaces = make(map[string]string)
	funcs = make(map[string]bool)
	for _, name := range scope.Names() {
		tn, ok := scope.Lookup(name).(*types.TypeName)
		if !ok {
			continue
		}
		switch x := tn.Type().Underlying().(type) {
		case *types.Interface:
			iface := methoderFuncMap(x, false)
			if len(iface) == 0 {
				continue
			}
			for i := 0; i < x.NumMethods(); i++ {
				f := x.Method(i)
				sign := f.Type().(*types.Signature)
				if !anyInteresting(sign.Params()) {
					continue
				}
				funcs[signString(sign)] = true
			}
			s := funcMapString(iface)
			if _, e := ifaces[s]; !e {
				ifaces[s] = tn.Name()
			}
		case *types.Signature:
			if !anyInteresting(x.Params()) {
				continue
			}
			funcs[signString(x)] = true
		}
	}
	return ifaces, funcs
}

func mentionsName(fname, name string) bool {
	if len(name) < 2 {
		return false
	}
	capit := strings.ToUpper(name[:1]) + name[1:]
	lower := strings.ToLower(name)
	return strings.Contains(fname, capit) || strings.HasPrefix(fname, lower)
}

func typeNamed(t types.Type) *types.Named {
	for {
		switch x := t.(type) {
		case *types.Named:
			return x
		case *types.Pointer:
			t = x.Elem()
		default:
			return nil
		}
	}
}
