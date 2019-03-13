package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeSentMatcher struct {
	Arg           interface{}
	channelClosed bool
}

func (matcher *BeSentMatcher) Match(actual interface{}) (success bool, err error) {
	if !isChan(actual) {
		return false, fmt.Errorf("BeSent expects a channel.  Got:\n%s", format.Object(actual, 1))
	}

	channelType := reflect.TypeOf(actual)
	channelValue := reflect.ValueOf(actual)

	if channelType.ChanDir() == reflect.RecvDir {
		return false, fmt.Errorf("BeSent matcher cannot be passed a receive-only channel.  Got:\n%s", format.Object(actual, 1))
	}

	argType := reflect.TypeOf(matcher.Arg)
	assignable := argType.AssignableTo(channelType.Elem())

	if !assignable {
		return false, fmt.Errorf("Cannot pass:\n%s to the channel:\n%s\nThe types don't match.", format.Object(matcher.Arg, 1), format.Object(actual, 1))
	}

	argValue := reflect.ValueOf(matcher.Arg)

	defer func() {
		if e := recover(); e != nil {
			success = false
			err = fmt.Errorf("Cannot send to a closed channel")
			matcher.channelClosed = true
		}
	}()

	winnerIndex, _, _ := reflect.Select([]reflect.SelectCase{
		{Dir: reflect.SelectSend, Chan: channelValue, Send: argValue},
		{Dir: reflect.SelectDefault},
	})

	var didSend bool
	if winnerIndex == 0 {
		didSend = true
	}

	return didSend, nil
}

func (matcher *BeSentMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to send:", matcher.Arg)
}

func (matcher *BeSentMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to send:", matcher.Arg)
}

func (matcher *BeSentMatcher) MatchMayChangeInTheFuture(actual interface{}) bool {
	if !isChan(actual) {
		return false
	}

	return !matcher.channelClosed
}
