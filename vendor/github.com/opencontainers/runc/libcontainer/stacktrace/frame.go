package stacktrace

import (
	"path/filepath"
	"runtime"
	"strings"
)

// NewFrame returns a new stack frame for the provided information
func NewFrame(pc uintptr, file string, line int) Frame {
	fn := runtime.FuncForPC(pc)
	if fn == nil {
		return Frame{}
	}
	pack, name := parseFunctionName(fn.Name())
	return Frame{
		Line:     line,
		File:     filepath.Base(file),
		Package:  pack,
		Function: name,
	}
}

func parseFunctionName(name string) (string, string) {
	i := strings.LastIndex(name, ".")
	if i == -1 {
		return "", name
	}
	return name[:i], name[i+1:]
}

// Frame contains all the information for a stack frame within a go program
type Frame struct {
	File     string
	Function string
	Package  string
	Line     int
}
