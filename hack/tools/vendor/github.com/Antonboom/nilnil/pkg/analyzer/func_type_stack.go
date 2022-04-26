package analyzer

import (
	"go/ast"
)

type funcTypeStack []*ast.FuncType

func (s *funcTypeStack) Push(f *ast.FuncType) {
	*s = append(*s, f)
}

func (s *funcTypeStack) Pop() *ast.FuncType {
	if len(*s) == 0 {
		return nil
	}

	last := len(*s) - 1
	f := (*s)[last]
	*s = (*s)[:last]
	return f
}

func (s *funcTypeStack) Top() *ast.FuncType {
	if len(*s) == 0 {
		return nil
	}
	return (*s)[len(*s)-1]
}
