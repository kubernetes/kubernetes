package pkg

import (
	"fmt"
	"go/ast"
	"go/token"
)

type sliceDeclaration struct {
	name string
	// sType string
	genD *ast.GenDecl
}

type returnsVisitor struct {
	// flags
	simple            bool
	includeRangeLoops bool
	includeForLoops   bool
	// visitor fields
	sliceDeclarations   []*sliceDeclaration
	preallocHints       []Hint
	returnsInsideOfLoop bool
	arrayTypes          []string
}

func Check(files []*ast.File, simple, includeRangeLoops, includeForLoops bool) []Hint {
	hints := []Hint{}
	for _, f := range files {
		retVis := &returnsVisitor{
			simple:            simple,
			includeRangeLoops: includeRangeLoops,
			includeForLoops:   includeForLoops,
		}
		ast.Walk(retVis, f)
		// if simple is true, then we actually have to check if we had returns
		// inside of our loop. Otherwise, we can just report all messages.
		if !retVis.simple || !retVis.returnsInsideOfLoop {
			hints = append(hints, retVis.preallocHints...)
		}
	}

	return hints
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}

	return false
}

func (v *returnsVisitor) Visit(node ast.Node) ast.Visitor {

	v.sliceDeclarations = nil
	v.returnsInsideOfLoop = false

	switch n := node.(type) {
	case *ast.TypeSpec:
		if _, ok := n.Type.(*ast.ArrayType); ok {
			if n.Name != nil {
				v.arrayTypes = append(v.arrayTypes, n.Name.Name)
			}
		}
	case *ast.FuncDecl:
		if n.Body != nil {
			for _, stmt := range n.Body.List {
				switch s := stmt.(type) {
				// Find non pre-allocated slices
				case *ast.DeclStmt:
					genD, ok := s.Decl.(*ast.GenDecl)
					if !ok {
						continue
					}
					if genD.Tok == token.TYPE {
						for _, spec := range genD.Specs {
							tSpec, ok := spec.(*ast.TypeSpec)
							if !ok {
								continue
							}

							if _, ok := tSpec.Type.(*ast.ArrayType); ok {
								if tSpec.Name != nil {
									v.arrayTypes = append(v.arrayTypes, tSpec.Name.Name)
								}
							}
						}
					} else if genD.Tok == token.VAR {
						for _, spec := range genD.Specs {
							vSpec, ok := spec.(*ast.ValueSpec)
							if !ok {
								continue
							}
							var isArrType bool
							switch val := vSpec.Type.(type) {
							case *ast.ArrayType:
								isArrType = true
							case *ast.Ident:
								isArrType = contains(v.arrayTypes, val.Name)
							}
							if isArrType {
								if vSpec.Names != nil {
									/*atID, ok := arrayType.Elt.(*ast.Ident)
									if !ok {
										continue
									}*/

									// We should handle multiple slices declared on same line e.g. var mySlice1, mySlice2 []uint32
									for _, vName := range vSpec.Names {
										v.sliceDeclarations = append(v.sliceDeclarations, &sliceDeclaration{name: vName.Name /*sType: atID.Name,*/, genD: genD})
									}
								}
							}
						}
					}

				case *ast.RangeStmt:
					if v.includeRangeLoops {
						if len(v.sliceDeclarations) == 0 {
							continue
						}
						// Check the value being ranged over and ensure it's not a channel (we cannot offer any recommendations on channel ranges).
						rangeIdent, ok := s.X.(*ast.Ident)
						if ok && rangeIdent.Obj != nil {
							valueSpec, ok := rangeIdent.Obj.Decl.(*ast.ValueSpec)
							if ok {
								if _, rangeTargetIsChannel := valueSpec.Type.(*ast.ChanType); rangeTargetIsChannel {
									continue
								}
							}
						}
						if s.Body != nil {
							v.handleLoops(s.Body)
						}
					}

				case *ast.ForStmt:
					if v.includeForLoops {
						if len(v.sliceDeclarations) == 0 {
							continue
						}
						if s.Body != nil {
							v.handleLoops(s.Body)
						}
					}

				default:
				}
			}
		}
	}
	return v
}

// handleLoops is a helper function to share the logic required for both *ast.RangeLoops and *ast.ForLoops
func (v *returnsVisitor) handleLoops(blockStmt *ast.BlockStmt) {

	for _, stmt := range blockStmt.List {
		switch bodyStmt := stmt.(type) {
		case *ast.AssignStmt:
			asgnStmt := bodyStmt
			for index, expr := range asgnStmt.Rhs {
				if index >= len(asgnStmt.Lhs) {
					continue
				}

				lhsIdent, ok := asgnStmt.Lhs[index].(*ast.Ident)
				if !ok {
					continue
				}

				callExpr, ok := expr.(*ast.CallExpr)
				if !ok {
					continue
				}

				rhsFuncIdent, ok := callExpr.Fun.(*ast.Ident)
				if !ok {
					continue
				}

				if rhsFuncIdent.Name != "append" {
					continue
				}

				// e.g., `x = append(x)`
				// Pointless, but pre-allocation will not help.
				if len(callExpr.Args) < 2 {
					continue
				}

				rhsIdent, ok := callExpr.Args[0].(*ast.Ident)
				if !ok {
					continue
				}

				// e.g., `x = append(y, a)`
				// This is weird (and maybe a logic error),
				// but we cannot recommend pre-allocation.
				if lhsIdent.Name != rhsIdent.Name {
					continue
				}

				// e.g., `x = append(x, y...)`
				// we should ignore this. Pre-allocating in this case
				// is confusing, and is not possible in general.
				if callExpr.Ellipsis.IsValid() {
					continue
				}

				for _, sliceDecl := range v.sliceDeclarations {
					if sliceDecl.name == lhsIdent.Name {
						// This is a potential mark, we just need to make sure there are no returns/continues in the
						// range loop.
						// now we just need to grab whatever we're ranging over
						/*sxIdent, ok := s.X.(*ast.Ident)
						if !ok {
							continue
						}*/

						v.preallocHints = append(v.preallocHints, Hint{
							Pos:               sliceDecl.genD.Pos(),
							DeclaredSliceName: sliceDecl.name,
						})
					}
				}
			}
		case *ast.IfStmt:
			ifStmt := bodyStmt
			if ifStmt.Body != nil {
				for _, ifBodyStmt := range ifStmt.Body.List {
					// TODO should probably handle embedded ifs here
					switch /*ift :=*/ ifBodyStmt.(type) {
					case *ast.BranchStmt, *ast.ReturnStmt:
						v.returnsInsideOfLoop = true
					default:
					}
				}
			}

		default:

		}
	}

}

// Hint stores the information about an occurrence of a slice that could be
// preallocated.
type Hint struct {
	Pos               token.Pos
	DeclaredSliceName string
}

func (h Hint) String() string {
	return fmt.Sprintf("%v: Consider preallocating %v", h.Pos, h.DeclaredSliceName)
}

func (h Hint) StringFromFS(f *token.FileSet) string {
	file := f.File(h.Pos)
	lineNumber := file.Position(h.Pos).Line

	return fmt.Sprintf("%v:%v Consider preallocating %v", file.Name(), lineNumber, h.DeclaredSliceName)
}
