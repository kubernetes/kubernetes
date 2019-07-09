package godog

import (
	"fmt"
	"go/build"
	"io"
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

// Frame represents a program counter inside a stack frame.
type stackFrame uintptr

// pc returns the program counter for this frame;
// multiple frames may have the same PC value.
func (f stackFrame) pc() uintptr { return uintptr(f) - 1 }

// file returns the full path to the file that contains the
// function for this Frame's pc.
func (f stackFrame) file() string {
	fn := runtime.FuncForPC(f.pc())
	if fn == nil {
		return "unknown"
	}
	file, _ := fn.FileLine(f.pc())
	return file
}

func trimGoPath(file string) string {
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		file = strings.Replace(file, filepath.Join(p, "src")+string(filepath.Separator), "", 1)
	}
	return file
}

// line returns the line number of source code of the
// function for this Frame's pc.
func (f stackFrame) line() int {
	fn := runtime.FuncForPC(f.pc())
	if fn == nil {
		return 0
	}
	_, line := fn.FileLine(f.pc())
	return line
}

// Format formats the frame according to the fmt.Formatter interface.
//
//    %s    source file
//    %d    source line
//    %n    function name
//    %v    equivalent to %s:%d
//
// Format accepts flags that alter the printing of some verbs, as follows:
//
//    %+s   path of source file relative to the compile time GOPATH
//    %+v   equivalent to %+s:%d
func (f stackFrame) Format(s fmt.State, verb rune) {
	funcname := func(name string) string {
		i := strings.LastIndex(name, "/")
		name = name[i+1:]
		i = strings.Index(name, ".")
		return name[i+1:]
	}

	switch verb {
	case 's':
		switch {
		case s.Flag('+'):
			pc := f.pc()
			fn := runtime.FuncForPC(pc)
			if fn == nil {
				io.WriteString(s, "unknown")
			} else {
				file, _ := fn.FileLine(pc)
				fmt.Fprintf(s, "%s\n\t%s", fn.Name(), trimGoPath(file))
			}
		default:
			io.WriteString(s, path.Base(f.file()))
		}
	case 'd':
		fmt.Fprintf(s, "%d", f.line())
	case 'n':
		name := runtime.FuncForPC(f.pc()).Name()
		io.WriteString(s, funcname(name))
	case 'v':
		f.Format(s, 's')
		io.WriteString(s, ":")
		f.Format(s, 'd')
	}
}

// stack represents a stack of program counters.
type stack []uintptr

func (s *stack) Format(st fmt.State, verb rune) {
	switch verb {
	case 'v':
		switch {
		case st.Flag('+'):
			for _, pc := range *s {
				f := stackFrame(pc)
				fmt.Fprintf(st, "\n%+v", f)
			}
		}
	}
}

func callStack() *stack {
	const depth = 32
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:])
	var st stack = pcs[0:n]
	return &st
}

// fundamental is an error that has a message and a stack, but no caller.
type traceError struct {
	msg string
	*stack
}

func (f *traceError) Error() string { return f.msg }

func (f *traceError) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v':
		if s.Flag('+') {
			io.WriteString(s, f.msg)
			f.stack.Format(s, verb)
			return
		}
		fallthrough
	case 's':
		io.WriteString(s, f.msg)
	case 'q':
		fmt.Fprintf(s, "%q", f.msg)
	}
}
