package types

type TWithHelper interface {
	Helper()
}

type GomegaFailHandler func(message string, callerSkip ...int)

type GomegaFailWrapper struct {
	Fail        GomegaFailHandler
	TWithHelper TWithHelper
}

//A simple *testing.T interface wrapper
type GomegaTestingT interface {
	Fatalf(format string, args ...interface{})
}

//All Gomega matchers must implement the GomegaMatcher interface
//
//For details on writing custom matchers, check out: http://onsi.github.io/gomega/#adding-your-own-matchers
type GomegaMatcher interface {
	Match(actual interface{}) (success bool, err error)
	FailureMessage(actual interface{}) (message string)
	NegatedFailureMessage(actual interface{}) (message string)
}
