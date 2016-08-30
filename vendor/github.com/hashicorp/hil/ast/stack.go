package ast

// Stack is a stack of Node.
type Stack struct {
	stack []Node
}

func (s *Stack) Len() int {
	return len(s.stack)
}

func (s *Stack) Push(n Node) {
	s.stack = append(s.stack, n)
}

func (s *Stack) Pop() Node {
	x := s.stack[len(s.stack)-1]
	s.stack[len(s.stack)-1] = nil
	s.stack = s.stack[:len(s.stack)-1]
	return x
}

func (s *Stack) Reset() {
	s.stack = nil
}
