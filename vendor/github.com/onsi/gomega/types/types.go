package types

type GomegaFailHandler func(message string, callerSkip ...int)

//A simple *testing.T interface wrapper
type GomegaTestingT interface {
	Errorf(format string, args ...interface{})
}

//All Gomega matchers must implement the GomegaMatcher interface
//
//For details on writing custom matchers, check out: http://onsi.github.io/gomega/#adding_your_own_matchers
type GomegaMatcher interface {
	Match(actual interface{}) (success bool, err error)
	FailureMessage(actual interface{}) (message string)
	NegatedFailureMessage(actual interface{}) (message string)
}
