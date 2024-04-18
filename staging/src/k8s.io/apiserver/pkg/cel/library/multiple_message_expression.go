package library

import "github.com/google/cel-go/cel"

func MultipleMessageExpression() cel.EnvOption {
	return cel.Lib(multipleMessageExpressionLib)
}

var multipleMessageExpressionLib = &multipleMessageExpression{}

var MultipleMessageExpressionName = "MultipleMessageExpression"

type multipleMessageExpression struct{}

func (*multipleMessageExpression) LibraryName() string {
	return MultipleMessageExpressionName
}

func (*multipleMessageExpression) CompileOptions() []cel.EnvOption {
	return nil
}

func (*multipleMessageExpression) ProgramOptions() []cel.ProgramOption {
	return nil
}
