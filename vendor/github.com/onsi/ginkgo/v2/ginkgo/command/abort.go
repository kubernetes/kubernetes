package command

import "fmt"

type AbortDetails struct {
	ExitCode  int
	Error     error
	EmitUsage bool
}

func Abort(details AbortDetails) {
	panic(details)
}

func AbortGracefullyWith(format string, args ...interface{}) {
	Abort(AbortDetails{
		ExitCode:  0,
		Error:     fmt.Errorf(format, args...),
		EmitUsage: false,
	})
}

func AbortWith(format string, args ...interface{}) {
	Abort(AbortDetails{
		ExitCode:  1,
		Error:     fmt.Errorf(format, args...),
		EmitUsage: false,
	})
}

func AbortWithUsage(format string, args ...interface{}) {
	Abort(AbortDetails{
		ExitCode:  1,
		Error:     fmt.Errorf(format, args...),
		EmitUsage: true,
	})
}

func AbortIfError(preamble string, err error) {
	if err != nil {
		Abort(AbortDetails{
			ExitCode:  1,
			Error:     fmt.Errorf("%s\n%s", preamble, err.Error()),
			EmitUsage: false,
		})
	}
}

func AbortIfErrors(preamble string, errors []error) {
	if len(errors) > 0 {
		out := ""
		for _, err := range errors {
			out += err.Error()
		}
		Abort(AbortDetails{
			ExitCode:  1,
			Error:     fmt.Errorf("%s\n%s", preamble, out),
			EmitUsage: false,
		})
	}
}
