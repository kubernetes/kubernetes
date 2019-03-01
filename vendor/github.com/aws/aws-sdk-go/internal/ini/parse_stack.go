package ini

import (
	"bytes"
	"fmt"
)

// ParseStack is a stack that contains a container, the stack portion,
// and the list which is the list of ASTs that have been successfully
// parsed.
type ParseStack struct {
	top       int
	container []AST
	list      []AST
	index     int
}

func newParseStack(sizeContainer, sizeList int) ParseStack {
	return ParseStack{
		container: make([]AST, sizeContainer),
		list:      make([]AST, sizeList),
	}
}

// Pop will return and truncate the last container element.
func (s *ParseStack) Pop() AST {
	s.top--
	return s.container[s.top]
}

// Push will add the new AST to the container
func (s *ParseStack) Push(ast AST) {
	s.container[s.top] = ast
	s.top++
}

// MarkComplete will append the AST to the list of completed statements
func (s *ParseStack) MarkComplete(ast AST) {
	s.list[s.index] = ast
	s.index++
}

// List will return the completed statements
func (s ParseStack) List() []AST {
	return s.list[:s.index]
}

// Len will return the length of the container
func (s *ParseStack) Len() int {
	return s.top
}

func (s ParseStack) String() string {
	buf := bytes.Buffer{}
	for i, node := range s.list {
		buf.WriteString(fmt.Sprintf("%d: %v\n", i+1, node))
	}

	return buf.String()
}
