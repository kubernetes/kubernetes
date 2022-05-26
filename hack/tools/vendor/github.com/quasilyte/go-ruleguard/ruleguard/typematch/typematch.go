package typematch

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strconv"
	"strings"

	"github.com/quasilyte/go-ruleguard/internal/xtypes"
)

//go:generate stringer -type=patternOp
type patternOp int

const (
	opBuiltinType patternOp = iota
	opPointer
	opVar
	opVarSeq
	opSlice
	opArray
	opMap
	opChan
	opFunc
	opStructNoSeq
	opStruct
	opAnyInterface
	opNamed
)

type Pattern struct {
	typeMatches  map[string]types.Type
	int64Matches map[string]int64

	root *pattern
}

type pattern struct {
	value interface{}
	op    patternOp
	subs  []*pattern
}

func (pat pattern) String() string {
	if len(pat.subs) == 0 {
		return fmt.Sprintf("<%s %#v>", pat.op, pat.value)
	}
	parts := make([]string, len(pat.subs))
	for i, sub := range pat.subs {
		parts[i] = sub.String()
	}
	return fmt.Sprintf("<%s %#v (%s)>", pat.op, pat.value, strings.Join(parts, ", "))
}

type ImportsTab struct {
	imports []map[string]string
}

func NewImportsTab(initial map[string]string) *ImportsTab {
	return &ImportsTab{imports: []map[string]string{initial}}
}

func (itab *ImportsTab) Lookup(pkgName string) (string, bool) {
	for i := len(itab.imports) - 1; i >= 0; i-- {
		pkgPath, ok := itab.imports[i][pkgName]
		if ok {
			return pkgPath, true
		}
	}
	return "", false
}

func (itab *ImportsTab) Load(pkgName, pkgPath string) {
	itab.imports[len(itab.imports)-1][pkgName] = pkgPath
}

func (itab *ImportsTab) EnterScope() {
	itab.imports = append(itab.imports, map[string]string{})
}

func (itab *ImportsTab) LeaveScope() {
	itab.imports = itab.imports[:len(itab.imports)-1]
}

type Context struct {
	Itab *ImportsTab
}

const (
	varPrefix    = `ᐸvarᐳ`
	varSeqPrefix = `ᐸvar_seqᐳ`
)

func Parse(ctx *Context, s string) (*Pattern, error) {
	noDollars := strings.ReplaceAll(s, "$*", varSeqPrefix)
	noDollars = strings.ReplaceAll(noDollars, "$", varPrefix)
	n, err := parser.ParseExpr(noDollars)
	if err != nil {
		return nil, err
	}
	root := parseExpr(ctx, n)
	if root == nil {
		return nil, fmt.Errorf("can't convert %s type expression", s)
	}
	p := &Pattern{
		typeMatches:  map[string]types.Type{},
		int64Matches: map[string]int64{},
		root:         root,
	}
	return p, nil
}

var (
	builtinTypeByName = map[string]types.Type{
		"bool":       types.Typ[types.Bool],
		"int":        types.Typ[types.Int],
		"int8":       types.Typ[types.Int8],
		"int16":      types.Typ[types.Int16],
		"int32":      types.Typ[types.Int32],
		"int64":      types.Typ[types.Int64],
		"uint":       types.Typ[types.Uint],
		"uint8":      types.Typ[types.Uint8],
		"uint16":     types.Typ[types.Uint16],
		"uint32":     types.Typ[types.Uint32],
		"uint64":     types.Typ[types.Uint64],
		"uintptr":    types.Typ[types.Uintptr],
		"float32":    types.Typ[types.Float32],
		"float64":    types.Typ[types.Float64],
		"complex64":  types.Typ[types.Complex64],
		"complex128": types.Typ[types.Complex128],
		"string":     types.Typ[types.String],

		"error": types.Universe.Lookup("error").Type(),

		// Aliases.
		"byte": types.Typ[types.Uint8],
		"rune": types.Typ[types.Int32],
	}

	efaceType = types.NewInterfaceType(nil, nil)
)

func parseExpr(ctx *Context, e ast.Expr) *pattern {
	switch e := e.(type) {
	case *ast.Ident:
		basic, ok := builtinTypeByName[e.Name]
		if ok {
			return &pattern{op: opBuiltinType, value: basic}
		}
		if strings.HasPrefix(e.Name, varPrefix) {
			name := strings.TrimPrefix(e.Name, varPrefix)
			return &pattern{op: opVar, value: name}
		}
		if strings.HasPrefix(e.Name, varSeqPrefix) {
			name := strings.TrimPrefix(e.Name, varSeqPrefix)
			// Only unnamed seq are supported right now.
			if name == "_" {
				return &pattern{op: opVarSeq, value: name}
			}
		}

	case *ast.SelectorExpr:
		pkg, ok := e.X.(*ast.Ident)
		if !ok {
			return nil
		}
		pkgPath, ok := ctx.Itab.Lookup(pkg.Name)
		if !ok {
			return nil
		}
		return &pattern{op: opNamed, value: [2]string{pkgPath, e.Sel.Name}}

	case *ast.StarExpr:
		elem := parseExpr(ctx, e.X)
		if elem == nil {
			return nil
		}
		return &pattern{op: opPointer, subs: []*pattern{elem}}

	case *ast.ArrayType:
		elem := parseExpr(ctx, e.Elt)
		if elem == nil {
			return nil
		}
		if e.Len == nil {
			return &pattern{
				op:   opSlice,
				subs: []*pattern{elem},
			}
		}
		if id, ok := e.Len.(*ast.Ident); ok && strings.HasPrefix(id.Name, varPrefix) {
			name := strings.TrimPrefix(id.Name, varPrefix)
			return &pattern{
				op:    opArray,
				value: name,
				subs:  []*pattern{elem},
			}
		}
		lit, ok := e.Len.(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			return nil
		}
		length, err := strconv.ParseInt(lit.Value, 10, 64)
		if err != nil {
			return nil
		}
		return &pattern{
			op:    opArray,
			value: length,
			subs:  []*pattern{elem},
		}

	case *ast.MapType:
		keyType := parseExpr(ctx, e.Key)
		if keyType == nil {
			return nil
		}
		valType := parseExpr(ctx, e.Value)
		if valType == nil {
			return nil
		}
		return &pattern{
			op:   opMap,
			subs: []*pattern{keyType, valType},
		}

	case *ast.ChanType:
		valType := parseExpr(ctx, e.Value)
		if valType == nil {
			return nil
		}
		var dir types.ChanDir
		switch {
		case e.Dir&ast.SEND != 0 && e.Dir&ast.RECV != 0:
			dir = types.SendRecv
		case e.Dir&ast.SEND != 0:
			dir = types.SendOnly
		case e.Dir&ast.RECV != 0:
			dir = types.RecvOnly
		default:
			return nil
		}
		return &pattern{
			op:    opChan,
			value: dir,
			subs:  []*pattern{valType},
		}

	case *ast.ParenExpr:
		return parseExpr(ctx, e.X)

	case *ast.FuncType:
		var params []*pattern
		var results []*pattern
		if e.Params != nil {
			for _, field := range e.Params.List {
				p := parseExpr(ctx, field.Type)
				if p == nil {
					return nil
				}
				if len(field.Names) != 0 {
					return nil
				}
				params = append(params, p)
			}
		}
		if e.Results != nil {
			for _, field := range e.Results.List {
				p := parseExpr(ctx, field.Type)
				if p == nil {
					return nil
				}
				if len(field.Names) != 0 {
					return nil
				}
				results = append(results, p)
			}
		}
		return &pattern{
			op:    opFunc,
			value: len(params),
			subs:  append(params, results...),
		}

	case *ast.StructType:
		hasSeq := false
		members := make([]*pattern, 0, len(e.Fields.List))
		for _, field := range e.Fields.List {
			p := parseExpr(ctx, field.Type)
			if p == nil {
				return nil
			}
			if len(field.Names) != 0 {
				return nil
			}
			if p.op == opVarSeq {
				hasSeq = true
			}
			members = append(members, p)
		}
		op := opStructNoSeq
		if hasSeq {
			op = opStruct
		}
		return &pattern{
			op:   op,
			subs: members,
		}

	case *ast.InterfaceType:
		if len(e.Methods.List) == 0 {
			return &pattern{op: opBuiltinType, value: efaceType}
		}
		if len(e.Methods.List) == 1 {
			p := parseExpr(ctx, e.Methods.List[0].Type)
			if p == nil {
				return nil
			}
			if p.op != opVarSeq {
				return nil
			}
			return &pattern{op: opAnyInterface}
		}
	}

	return nil
}

// MatchIdentical returns true if the go typ matches pattern p.
func (p *Pattern) MatchIdentical(typ types.Type) bool {
	p.reset()
	return p.matchIdentical(p.root, typ)
}

func (p *Pattern) reset() {
	if len(p.int64Matches) != 0 {
		p.int64Matches = map[string]int64{}
	}
	if len(p.typeMatches) != 0 {
		p.typeMatches = map[string]types.Type{}
	}
}

func (p *Pattern) matchIdenticalFielder(subs []*pattern, f fielder) bool {
	// TODO: do backtracking.

	numFields := f.NumFields()
	fieldsMatched := 0

	if len(subs) == 0 && numFields != 0 {
		return false
	}

	matchAny := false

	i := 0
	for i < len(subs) {
		pat := subs[i]

		if pat.op == opVarSeq {
			matchAny = true
		}

		fieldsLeft := numFields - fieldsMatched
		if matchAny {
			switch {
			// "Nothing left to match" stop condition.
			case fieldsLeft == 0:
				matchAny = false
				i++
			// Lookahead for non-greedy matching.
			case i+1 < len(subs) && p.matchIdentical(subs[i+1], f.Field(fieldsMatched).Type()):
				matchAny = false
				i += 2
				fieldsMatched++
			default:
				fieldsMatched++
			}
			continue
		}

		if fieldsLeft == 0 || !p.matchIdentical(pat, f.Field(fieldsMatched).Type()) {
			return false
		}
		i++
		fieldsMatched++
	}

	return numFields == fieldsMatched
}

func (p *Pattern) matchIdentical(sub *pattern, typ types.Type) bool {
	switch sub.op {
	case opVar:
		name := sub.value.(string)
		if name == "_" {
			return true
		}
		y, ok := p.typeMatches[name]
		if !ok {
			p.typeMatches[name] = typ
			return true
		}
		if y == nil {
			return typ == nil
		}
		return xtypes.Identical(typ, y)

	case opBuiltinType:
		return xtypes.Identical(typ, sub.value.(types.Type))

	case opPointer:
		typ, ok := typ.(*types.Pointer)
		if !ok {
			return false
		}
		return p.matchIdentical(sub.subs[0], typ.Elem())

	case opSlice:
		typ, ok := typ.(*types.Slice)
		if !ok {
			return false
		}
		return p.matchIdentical(sub.subs[0], typ.Elem())

	case opArray:
		typ, ok := typ.(*types.Array)
		if !ok {
			return false
		}
		var wantLen int64
		switch v := sub.value.(type) {
		case string:
			if v == "_" {
				wantLen = typ.Len()
				break
			}
			length, ok := p.int64Matches[v]
			if ok {
				wantLen = length
			} else {
				p.int64Matches[v] = typ.Len()
				wantLen = typ.Len()
			}
		case int64:
			wantLen = v
		}
		return wantLen == typ.Len() && p.matchIdentical(sub.subs[0], typ.Elem())

	case opMap:
		typ, ok := typ.(*types.Map)
		if !ok {
			return false
		}
		return p.matchIdentical(sub.subs[0], typ.Key()) &&
			p.matchIdentical(sub.subs[1], typ.Elem())

	case opChan:
		typ, ok := typ.(*types.Chan)
		if !ok {
			return false
		}
		dir := sub.value.(types.ChanDir)
		return dir == typ.Dir() && p.matchIdentical(sub.subs[0], typ.Elem())

	case opNamed:
		typ, ok := typ.(*types.Named)
		if !ok {
			return false
		}
		obj := typ.Obj()
		pkg := obj.Pkg()
		// pkg can be nil for builtin named types.
		// There is no point in checking anything else as we never
		// generate the opNamed for such types.
		if pkg == nil {
			return false
		}
		pkgPath := sub.value.([2]string)[0]
		typeName := sub.value.([2]string)[1]
		// obj.Pkg().Path() may be in a vendor directory.
		path := strings.SplitAfter(obj.Pkg().Path(), "/vendor/")
		return path[len(path)-1] == pkgPath && typeName == obj.Name()

	case opFunc:
		typ, ok := typ.(*types.Signature)
		if !ok {
			return false
		}
		numParams := sub.value.(int)
		params := sub.subs[:numParams]
		results := sub.subs[numParams:]
		if typ.Params().Len() != len(params) {
			return false
		}
		if typ.Results().Len() != len(results) {
			return false
		}
		for i := 0; i < typ.Params().Len(); i++ {
			if !p.matchIdentical(params[i], typ.Params().At(i).Type()) {
				return false
			}
		}
		for i := 0; i < typ.Results().Len(); i++ {
			if !p.matchIdentical(results[i], typ.Results().At(i).Type()) {
				return false
			}
		}
		return true

	case opStructNoSeq:
		typ, ok := typ.(*types.Struct)
		if !ok {
			return false
		}
		if typ.NumFields() != len(sub.subs) {
			return false
		}
		for i, member := range sub.subs {
			if !p.matchIdentical(member, typ.Field(i).Type()) {
				return false
			}
		}
		return true

	case opStruct:
		typ, ok := typ.(*types.Struct)
		if !ok {
			return false
		}
		if !p.matchIdenticalFielder(sub.subs, typ) {
			return false
		}
		return true

	case opAnyInterface:
		_, ok := typ.(*types.Interface)
		return ok

	default:
		return false
	}
}

type fielder interface {
	Field(i int) *types.Var
	NumFields() int
}
