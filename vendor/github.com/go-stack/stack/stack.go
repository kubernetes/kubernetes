// +build go1.7

// Package stack implements utilities to capture, manipulate, and format call
// stacks. It provides a simpler API than package runtime.
//
// The implementation takes care of the minutia and special cases of
// interpreting the program counter (pc) values returned by runtime.Callers.
//
// Package stack's types implement fmt.Formatter, which provides a simple and
// flexible way to declaratively configure formatting when used with logging
// or error tracking packages.
package stack

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
)

// Call records a single function invocation from a goroutine stack.
type Call struct {
	frame runtime.Frame
}

// Caller returns a Call from the stack of the current goroutine. The argument
// skip is the number of stack frames to ascend, with 0 identifying the
// calling function.
func Caller(skip int) Call {
	// As of Go 1.9 we need room for up to three PC entries.
	//
	// 0. An entry for the stack frame prior to the target to check for
	//    special handling needed if that prior entry is runtime.sigpanic.
	// 1. A possible second entry to hold metadata about skipped inlined
	//    functions. If inline functions were not skipped the target frame
	//    PC will be here.
	// 2. A third entry for the target frame PC when the second entry
	//    is used for skipped inline functions.
	var pcs [3]uintptr
	n := runtime.Callers(skip+1, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	frame, _ := frames.Next()
	frame, _ = frames.Next()

	return Call{
		frame: frame,
	}
}

// String implements fmt.Stinger. It is equivalent to fmt.Sprintf("%v", c).
func (c Call) String() string {
	return fmt.Sprint(c)
}

// MarshalText implements encoding.TextMarshaler. It formats the Call the same
// as fmt.Sprintf("%v", c).
func (c Call) MarshalText() ([]byte, error) {
	if c.frame == (runtime.Frame{}) {
		return nil, ErrNoFunc
	}

	buf := bytes.Buffer{}
	fmt.Fprint(&buf, c)
	return buf.Bytes(), nil
}

// ErrNoFunc means that the Call has a nil *runtime.Func. The most likely
// cause is a Call with the zero value.
var ErrNoFunc = errors.New("no call stack information")

// Format implements fmt.Formatter with support for the following verbs.
//
//    %s    source file
//    %d    line number
//    %n    function name
//    %k    last segment of the package path
//    %v    equivalent to %s:%d
//
// It accepts the '+' and '#' flags for most of the verbs as follows.
//
//    %+s   path of source file relative to the compile time GOPATH,
//          or the module path joined to the path of source file relative
//          to module root
//    %#s   full path of source file
//    %+n   import path qualified function name
//    %+k   full package path
//    %+v   equivalent to %+s:%d
//    %#v   equivalent to %#s:%d
func (c Call) Format(s fmt.State, verb rune) {
	if c.frame == (runtime.Frame{}) {
		fmt.Fprintf(s, "%%!%c(NOFUNC)", verb)
		return
	}

	switch verb {
	case 's', 'v':
		file := c.frame.File
		switch {
		case s.Flag('#'):
			// done
		case s.Flag('+'):
			file = pkgFilePath(&c.frame)
		default:
			const sep = "/"
			if i := strings.LastIndex(file, sep); i != -1 {
				file = file[i+len(sep):]
			}
		}
		io.WriteString(s, file)
		if verb == 'v' {
			buf := [7]byte{':'}
			s.Write(strconv.AppendInt(buf[:1], int64(c.frame.Line), 10))
		}

	case 'd':
		buf := [6]byte{}
		s.Write(strconv.AppendInt(buf[:0], int64(c.frame.Line), 10))

	case 'k':
		name := c.frame.Function
		const pathSep = "/"
		start, end := 0, len(name)
		if i := strings.LastIndex(name, pathSep); i != -1 {
			start = i + len(pathSep)
		}
		const pkgSep = "."
		if i := strings.Index(name[start:], pkgSep); i != -1 {
			end = start + i
		}
		if s.Flag('+') {
			start = 0
		}
		io.WriteString(s, name[start:end])

	case 'n':
		name := c.frame.Function
		if !s.Flag('+') {
			const pathSep = "/"
			if i := strings.LastIndex(name, pathSep); i != -1 {
				name = name[i+len(pathSep):]
			}
			const pkgSep = "."
			if i := strings.Index(name, pkgSep); i != -1 {
				name = name[i+len(pkgSep):]
			}
		}
		io.WriteString(s, name)
	}
}

// Frame returns the call frame infomation for the Call.
func (c Call) Frame() runtime.Frame {
	return c.frame
}

// PC returns the program counter for this call frame; multiple frames may
// have the same PC value.
//
// Deprecated: Use Call.Frame instead.
func (c Call) PC() uintptr {
	return c.frame.PC
}

// CallStack records a sequence of function invocations from a goroutine
// stack.
type CallStack []Call

// String implements fmt.Stinger. It is equivalent to fmt.Sprintf("%v", cs).
func (cs CallStack) String() string {
	return fmt.Sprint(cs)
}

var (
	openBracketBytes  = []byte("[")
	closeBracketBytes = []byte("]")
	spaceBytes        = []byte(" ")
)

// MarshalText implements encoding.TextMarshaler. It formats the CallStack the
// same as fmt.Sprintf("%v", cs).
func (cs CallStack) MarshalText() ([]byte, error) {
	buf := bytes.Buffer{}
	buf.Write(openBracketBytes)
	for i, pc := range cs {
		if i > 0 {
			buf.Write(spaceBytes)
		}
		fmt.Fprint(&buf, pc)
	}
	buf.Write(closeBracketBytes)
	return buf.Bytes(), nil
}

// Format implements fmt.Formatter by printing the CallStack as square brackets
// ([, ]) surrounding a space separated list of Calls each formatted with the
// supplied verb and options.
func (cs CallStack) Format(s fmt.State, verb rune) {
	s.Write(openBracketBytes)
	for i, pc := range cs {
		if i > 0 {
			s.Write(spaceBytes)
		}
		pc.Format(s, verb)
	}
	s.Write(closeBracketBytes)
}

// Trace returns a CallStack for the current goroutine with element 0
// identifying the calling function.
func Trace() CallStack {
	var pcs [512]uintptr
	n := runtime.Callers(1, pcs[:])

	frames := runtime.CallersFrames(pcs[:n])
	cs := make(CallStack, 0, n)

	// Skip extra frame retrieved just to make sure the runtime.sigpanic
	// special case is handled.
	frame, more := frames.Next()

	for more {
		frame, more = frames.Next()
		cs = append(cs, Call{frame: frame})
	}

	return cs
}

// TrimBelow returns a slice of the CallStack with all entries below c
// removed.
func (cs CallStack) TrimBelow(c Call) CallStack {
	for len(cs) > 0 && cs[0] != c {
		cs = cs[1:]
	}
	return cs
}

// TrimAbove returns a slice of the CallStack with all entries above c
// removed.
func (cs CallStack) TrimAbove(c Call) CallStack {
	for len(cs) > 0 && cs[len(cs)-1] != c {
		cs = cs[:len(cs)-1]
	}
	return cs
}

// pkgIndex returns the index that results in file[index:] being the path of
// file relative to the compile time GOPATH, and file[:index] being the
// $GOPATH/src/ portion of file. funcName must be the name of a function in
// file as returned by runtime.Func.Name.
func pkgIndex(file, funcName string) int {
	// As of Go 1.6.2 there is no direct way to know the compile time GOPATH
	// at runtime, but we can infer the number of path segments in the GOPATH.
	// We note that runtime.Func.Name() returns the function name qualified by
	// the import path, which does not include the GOPATH. Thus we can trim
	// segments from the beginning of the file path until the number of path
	// separators remaining is one more than the number of path separators in
	// the function name. For example, given:
	//
	//    GOPATH     /home/user
	//    file       /home/user/src/pkg/sub/file.go
	//    fn.Name()  pkg/sub.Type.Method
	//
	// We want to produce:
	//
	//    file[:idx] == /home/user/src/
	//    file[idx:] == pkg/sub/file.go
	//
	// From this we can easily see that fn.Name() has one less path separator
	// than our desired result for file[idx:]. We count separators from the
	// end of the file path until it finds two more than in the function name
	// and then move one character forward to preserve the initial path
	// segment without a leading separator.
	const sep = "/"
	i := len(file)
	for n := strings.Count(funcName, sep) + 2; n > 0; n-- {
		i = strings.LastIndex(file[:i], sep)
		if i == -1 {
			i = -len(sep)
			break
		}
	}
	// get back to 0 or trim the leading separator
	return i + len(sep)
}

// pkgFilePath returns the frame's filepath relative to the compile-time GOPATH,
// or its module path joined to its path relative to the module root.
//
// As of Go 1.11 there is no direct way to know the compile time GOPATH or
// module paths at runtime, but we can piece together the desired information
// from available information. We note that runtime.Frame.Function contains the
// function name qualified by the package path, which includes the module path
// but not the GOPATH. We can extract the package path from that and append the
// last segments of the file path to arrive at the desired package qualified
// file path. For example, given:
//
//    GOPATH          /home/user
//    import path     pkg/sub
//    frame.File      /home/user/src/pkg/sub/file.go
//    frame.Function  pkg/sub.Type.Method
//    Desired return  pkg/sub/file.go
//
// It appears that we simply need to trim ".Type.Method" from frame.Function and
// append "/" + path.Base(file).
//
// But there are other wrinkles. Although it is idiomatic to do so, the internal
// name of a package is not required to match the last segment of its import
// path. In addition, the introduction of modules in Go 1.11 allows working
// without a GOPATH. So we also must make these work right:
//
//    GOPATH          /home/user
//    import path     pkg/go-sub
//    package name    sub
//    frame.File      /home/user/src/pkg/go-sub/file.go
//    frame.Function  pkg/sub.Type.Method
//    Desired return  pkg/go-sub/file.go
//
//    Module path     pkg/v2
//    import path     pkg/v2/go-sub
//    package name    sub
//    frame.File      /home/user/cloned-pkg/go-sub/file.go
//    frame.Function  pkg/v2/sub.Type.Method
//    Desired return  pkg/v2/go-sub/file.go
//
// We can handle all of these situations by using the package path extracted
// from frame.Function up to, but not including, the last segment as the prefix
// and the last two segments of frame.File as the suffix of the returned path.
// This preserves the existing behavior when working in a GOPATH without modules
// and a semantically equivalent behavior when used in module aware project.
func pkgFilePath(frame *runtime.Frame) string {
	pre := pkgPrefix(frame.Function)
	post := pathSuffix(frame.File)
	if pre == "" {
		return post
	}
	return pre + "/" + post
}

// pkgPrefix returns the import path of the function's package with the final
// segment removed.
func pkgPrefix(funcName string) string {
	const pathSep = "/"
	end := strings.LastIndex(funcName, pathSep)
	if end == -1 {
		return ""
	}
	return funcName[:end]
}

// pathSuffix returns the last two segments of path.
func pathSuffix(path string) string {
	const pathSep = "/"
	lastSep := strings.LastIndex(path, pathSep)
	if lastSep == -1 {
		return path
	}
	return path[strings.LastIndex(path[:lastSep], pathSep)+1:]
}

var runtimePath string

func init() {
	var pcs [3]uintptr
	runtime.Callers(0, pcs[:])
	frames := runtime.CallersFrames(pcs[:])
	frame, _ := frames.Next()
	file := frame.File

	idx := pkgIndex(frame.File, frame.Function)

	runtimePath = file[:idx]
	if runtime.GOOS == "windows" {
		runtimePath = strings.ToLower(runtimePath)
	}
}

func inGoroot(c Call) bool {
	file := c.frame.File
	if len(file) == 0 || file[0] == '?' {
		return true
	}
	if runtime.GOOS == "windows" {
		file = strings.ToLower(file)
	}
	return strings.HasPrefix(file, runtimePath) || strings.HasSuffix(file, "/_testmain.go")
}

// TrimRuntime returns a slice of the CallStack with the topmost entries from
// the go runtime removed. It considers any calls originating from unknown
// files, files under GOROOT, or _testmain.go as part of the runtime.
func (cs CallStack) TrimRuntime() CallStack {
	for len(cs) > 0 && inGoroot(cs[len(cs)-1]) {
		cs = cs[:len(cs)-1]
	}
	return cs
}
