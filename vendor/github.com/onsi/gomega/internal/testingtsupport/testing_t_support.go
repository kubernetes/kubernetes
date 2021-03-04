package testingtsupport

import (
	"regexp"
	"runtime/debug"
	"strings"

	"github.com/onsi/gomega/types"
)

var StackTracePruneRE = regexp.MustCompile(`\/gomega\/|\/ginkgo\/|\/pkg\/testing\/|\/pkg\/runtime\/`)

type EmptyTWithHelper struct{}

func (e EmptyTWithHelper) Helper() {}

type gomegaTestingT interface {
	Fatalf(format string, args ...interface{})
}

func BuildTestingTGomegaFailWrapper(t gomegaTestingT) *types.GomegaFailWrapper {
	tWithHelper, hasHelper := t.(types.TWithHelper)
	if !hasHelper {
		tWithHelper = EmptyTWithHelper{}
	}

	fail := func(message string, callerSkip ...int) {
		if hasHelper {
			tWithHelper.Helper()
			t.Fatalf("\n%s", message)
		} else {
			skip := 2
			if len(callerSkip) > 0 {
				skip += callerSkip[0]
			}
			stackTrace := pruneStack(string(debug.Stack()), skip)
			t.Fatalf("\n%s\n%s\n", stackTrace, message)
		}
	}

	return &types.GomegaFailWrapper{
		Fail:        fail,
		TWithHelper: tWithHelper,
	}
}

func pruneStack(fullStackTrace string, skip int) string {
	stack := strings.Split(fullStackTrace, "\n")[1:]
	if len(stack) > 2*skip {
		stack = stack[2*skip:]
	}
	prunedStack := []string{}
	for i := 0; i < len(stack)/2; i++ {
		if !StackTracePruneRE.Match([]byte(stack[i*2])) {
			prunedStack = append(prunedStack, stack[i*2])
			prunedStack = append(prunedStack, stack[i*2+1])
		}
	}
	return strings.Join(prunedStack, "\n")
}
