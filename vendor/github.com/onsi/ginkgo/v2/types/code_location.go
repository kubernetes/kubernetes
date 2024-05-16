package types

import (
	"fmt"
	"os"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
)

type CodeLocation struct {
	FileName       string `json:",omitempty"`
	LineNumber     int    `json:",omitempty"`
	FullStackTrace string `json:",omitempty"`
	CustomMessage  string `json:",omitempty"`
}

func (codeLocation CodeLocation) String() string {
	if codeLocation.CustomMessage != "" {
		return codeLocation.CustomMessage
	}
	return fmt.Sprintf("%s:%d", codeLocation.FileName, codeLocation.LineNumber)
}

func (codeLocation CodeLocation) ContentsOfLine() string {
	if codeLocation.CustomMessage != "" {
		return ""
	}
	contents, err := os.ReadFile(codeLocation.FileName)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(contents), "\n")
	if len(lines) < codeLocation.LineNumber {
		return ""
	}
	return lines[codeLocation.LineNumber-1]
}

type codeLocationLocator struct {
	pcs     map[uintptr]bool
	helpers map[string]bool
	lock    *sync.Mutex
}

func (c *codeLocationLocator) addHelper(pc uintptr) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if c.pcs[pc] {
		return
	}
	c.lock.Unlock()
	f := runtime.FuncForPC(pc)
	c.lock.Lock()
	if f == nil {
		return
	}
	c.helpers[f.Name()] = true
	c.pcs[pc] = true
}

func (c *codeLocationLocator) hasHelper(name string) bool {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.helpers[name]
}

func (c *codeLocationLocator) getCodeLocation(skip int) CodeLocation {
	pc := make([]uintptr, 40)
	n := runtime.Callers(skip+2, pc)
	if n == 0 {
		return CodeLocation{}
	}
	pc = pc[:n]
	frames := runtime.CallersFrames(pc)
	for {
		frame, more := frames.Next()
		if !c.hasHelper(frame.Function) {
			return CodeLocation{FileName: frame.File, LineNumber: frame.Line}
		}
		if !more {
			break
		}
	}
	return CodeLocation{}
}

var clLocator = &codeLocationLocator{
	pcs:     map[uintptr]bool{},
	helpers: map[string]bool{},
	lock:    &sync.Mutex{},
}

// MarkAsHelper is used by GinkgoHelper to mark the caller (appropriately offset by skip)as a helper.  You can use this directly if you need to provide an optional `skip` to mark functions further up the call stack as helpers.
func MarkAsHelper(optionalSkip ...int) {
	skip := 1
	if len(optionalSkip) > 0 {
		skip += optionalSkip[0]
	}
	pc, _, _, ok := runtime.Caller(skip)
	if ok {
		clLocator.addHelper(pc)
	}
}

func NewCustomCodeLocation(message string) CodeLocation {
	return CodeLocation{
		CustomMessage: message,
	}
}

func NewCodeLocation(skip int) CodeLocation {
	return clLocator.getCodeLocation(skip + 1)
}

func NewCodeLocationWithStackTrace(skip int) CodeLocation {
	cl := clLocator.getCodeLocation(skip + 1)
	cl.FullStackTrace = PruneStack(string(debug.Stack()), skip+1)
	return cl
}

// PruneStack removes references to functions that are internal to Ginkgo
// and the Go runtime from a stack string and a certain number of stack entries
// at the beginning of the stack. The stack string has the format
// as returned by runtime/debug.Stack. The leading goroutine information is
// optional and always removed if present. Beware that runtime/debug.Stack
// adds itself as first entry, so typically skip must be >= 1 to remove that
// entry.
func PruneStack(fullStackTrace string, skip int) string {
	stack := strings.Split(fullStackTrace, "\n")
	// Ensure that the even entries are the method names and the
	// odd entries the source code information.
	if len(stack) > 0 && strings.HasPrefix(stack[0], "goroutine ") {
		// Ignore "goroutine 29 [running]:" line.
		stack = stack[1:]
	}
	// The "+1" is for skipping over the initial entry, which is
	// runtime/debug.Stack() itself.
	if len(stack) > 2*(skip+1) {
		stack = stack[2*(skip+1):]
	}
	prunedStack := []string{}
	if os.Getenv("GINKGO_PRUNE_STACK") == "FALSE" {
		prunedStack = stack
	} else {
		re := regexp.MustCompile(`\/ginkgo\/|\/pkg\/testing\/|\/pkg\/runtime\/`)
		for i := 0; i < len(stack)/2; i++ {
			// We filter out based on the source code file name.
			if !re.MatchString(stack[i*2+1]) {
				prunedStack = append(prunedStack, stack[i*2])
				prunedStack = append(prunedStack, stack[i*2+1])
			}
		}
	}
	return strings.Join(prunedStack, "\n")
}
