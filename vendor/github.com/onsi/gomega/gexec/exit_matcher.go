package gexec

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

/*
The Exit matcher operates on a session:

	Ω(session).Should(Exit(<optional status code>))

Exit passes if the session has already exited.

If no status code is provided, then Exit will succeed if the session has exited regardless of exit code.
Otherwise, Exit will only succeed if the process has exited with the provided status code.

Note that the process must have already exited.  To wait for a process to exit, use Eventually:

	Eventually(session, 3).Should(Exit(0))
*/
func Exit(optionalExitCode ...int) *exitMatcher {
	exitCode := -1
	if len(optionalExitCode) > 0 {
		exitCode = optionalExitCode[0]
	}

	return &exitMatcher{
		exitCode: exitCode,
	}
}

type exitMatcher struct {
	exitCode       int
	didExit        bool
	actualExitCode int
}

type Exiter interface {
	ExitCode() int
}

func (m *exitMatcher) Match(actual interface{}) (success bool, err error) {
	exiter, ok := actual.(Exiter)
	if !ok {
		return false, fmt.Errorf("Exit must be passed a gexec.Exiter (Missing method ExitCode() int) Got:\n%s", format.Object(actual, 1))
	}

	m.actualExitCode = exiter.ExitCode()

	if m.actualExitCode == -1 {
		return false, nil
	}

	if m.exitCode == -1 {
		return true, nil
	}
	return m.exitCode == m.actualExitCode, nil
}

func (m *exitMatcher) FailureMessage(actual interface{}) (message string) {
	if m.actualExitCode == -1 {
		return "Expected process to exit.  It did not."
	} else {
		return format.Message(m.actualExitCode, "to match exit code:", m.exitCode)
	}
}

func (m *exitMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	if m.actualExitCode == -1 {
		return "you really shouldn't be able to see this!"
	} else {
		if m.exitCode == -1 {
			return "Expected process not to exit.  It did."
		} else {
			return format.Message(m.actualExitCode, "not to match exit code:", m.exitCode)
		}
	}
}

func (m *exitMatcher) MatchMayChangeInTheFuture(actual interface{}) bool {
	session, ok := actual.(*Session)
	if ok {
		return session.ExitCode() == -1
	}
	return true
}
