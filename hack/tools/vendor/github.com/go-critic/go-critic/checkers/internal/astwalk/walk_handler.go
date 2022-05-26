package astwalk

import (
	"go/ast"
)

// WalkHandler is a type to be embedded into every checker
// that uses astwalk walkers.
type WalkHandler struct {
	// SkipChilds controls whether currently analyzed
	// node childs should be traversed.
	//
	// Value is reset after each visitor invocation,
	// so there is no need to set value back to false.
	SkipChilds bool
}

// EnterFile is a default walkerEvents.EnterFile implementation
// that reports every file as accepted candidate for checking.
func (w *WalkHandler) EnterFile(f *ast.File) bool {
	return true
}

// EnterFunc is a default walkerEvents.EnterFunc implementation
// that skips extern function (ones that do not have body).
func (w *WalkHandler) EnterFunc(decl *ast.FuncDecl) bool {
	return decl.Body != nil
}

func (w *WalkHandler) skipChilds() bool {
	v := w.SkipChilds
	w.SkipChilds = false
	return v
}
