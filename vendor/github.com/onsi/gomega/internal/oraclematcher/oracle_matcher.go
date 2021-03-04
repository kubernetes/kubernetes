package oraclematcher

import "github.com/onsi/gomega/types"

/*
GomegaMatchers that also match the OracleMatcher interface can convey information about
whether or not their result will change upon future attempts.

This allows `Eventually` and `Consistently` to short circuit if success becomes impossible.

For example, a process' exit code can never change.  So, gexec's Exit matcher returns `true`
for `MatchMayChangeInTheFuture` until the process exits, at which point it returns `false` forevermore.
*/
type OracleMatcher interface {
	MatchMayChangeInTheFuture(actual interface{}) bool
}

func MatchMayChangeInTheFuture(matcher types.GomegaMatcher, value interface{}) bool {
	oracleMatcher, ok := matcher.(OracleMatcher)
	if !ok {
		return true
	}

	return oracleMatcher.MatchMayChangeInTheFuture(value)
}
