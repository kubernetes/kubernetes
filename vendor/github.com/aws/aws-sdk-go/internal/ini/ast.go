package ini

// ASTKind represents different states in the parse table
// and the type of AST that is being constructed
type ASTKind int

// ASTKind* is used in the parse table to transition between
// the different states
const (
	ASTKindNone = ASTKind(iota)
	ASTKindStart
	ASTKindExpr
	ASTKindEqualExpr
	ASTKindStatement
	ASTKindSkipStatement
	ASTKindExprStatement
	ASTKindSectionStatement
	ASTKindNestedSectionStatement
	ASTKindCompletedNestedSectionStatement
	ASTKindCommentStatement
	ASTKindCompletedSectionStatement
)

func (k ASTKind) String() string {
	switch k {
	case ASTKindNone:
		return "none"
	case ASTKindStart:
		return "start"
	case ASTKindExpr:
		return "expr"
	case ASTKindStatement:
		return "stmt"
	case ASTKindSectionStatement:
		return "section_stmt"
	case ASTKindExprStatement:
		return "expr_stmt"
	case ASTKindCommentStatement:
		return "comment"
	case ASTKindNestedSectionStatement:
		return "nested_section_stmt"
	case ASTKindCompletedSectionStatement:
		return "completed_stmt"
	case ASTKindSkipStatement:
		return "skip"
	default:
		return ""
	}
}

// AST interface allows us to determine what kind of node we
// are on and casting may not need to be necessary.
//
// The root is always the first node in Children
type AST struct {
	Kind      ASTKind
	Root      Token
	RootToken bool
	Children  []AST
}

func newAST(kind ASTKind, root AST, children ...AST) AST {
	return AST{
		Kind:     kind,
		Children: append([]AST{root}, children...),
	}
}

func newASTWithRootToken(kind ASTKind, root Token, children ...AST) AST {
	return AST{
		Kind:      kind,
		Root:      root,
		RootToken: true,
		Children:  children,
	}
}

// AppendChild will append to the list of children an AST has.
func (a *AST) AppendChild(child AST) {
	a.Children = append(a.Children, child)
}

// GetRoot will return the root AST which can be the first entry
// in the children list or a token.
func (a *AST) GetRoot() AST {
	if a.RootToken {
		return *a
	}

	if len(a.Children) == 0 {
		return AST{}
	}

	return a.Children[0]
}

// GetChildren will return the current AST's list of children
func (a *AST) GetChildren() []AST {
	if len(a.Children) == 0 {
		return []AST{}
	}

	if a.RootToken {
		return a.Children
	}

	return a.Children[1:]
}

// SetChildren will set and override all children of the AST.
func (a *AST) SetChildren(children []AST) {
	if a.RootToken {
		a.Children = children
	} else {
		a.Children = append(a.Children[:1], children...)
	}
}

// Start is used to indicate the starting state of the parse table.
var Start = newAST(ASTKindStart, AST{})
