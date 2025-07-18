package codelocation

import (
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/onsi/ginkgo/types"
)

func New(skip int) types.CodeLocation {
	_, file, line, _ := runtime.Caller(skip + 1)
	stackTrace := PruneStack(string(debug.Stack()), skip+1)
	return types.CodeLocation{FileName: file, LineNumber: line, FullStackTrace: stackTrace}
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
	// the odd entries the source code information.
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
	re := regexp.MustCompile(`\/ginkgo\/|\/pkg\/testing\/|\/pkg\/runtime\/`)
	for i := 0; i < len(stack)/2; i++ {
		// We filter out based on the source code file name.
		if !re.Match([]byte(stack[i*2+1])) {
			prunedStack = append(prunedStack, stack[i*2])
			prunedStack = append(prunedStack, stack[i*2+1])
		}
	}
	return strings.Join(prunedStack, "\n")
}
