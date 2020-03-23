// untested sections: 3

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type ReceiveMatcher struct {
	Arg           interface{}
	receivedValue reflect.Value
	channelClosed bool
}

func (matcher *ReceiveMatcher) Match(actual interface{}) (success bool, err error) {
	if !isChan(actual) {
		return false, fmt.Errorf("ReceiveMatcher expects a channel.  Got:\n%s", format.Object(actual, 1))
	}

	channelType := reflect.TypeOf(actual)
	channelValue := reflect.ValueOf(actual)

	if channelType.ChanDir() == reflect.SendDir {
		return false, fmt.Errorf("ReceiveMatcher matcher cannot be passed a send-only channel.  Got:\n%s", format.Object(actual, 1))
	}

	var subMatcher omegaMatcher
	var hasSubMatcher bool

	if matcher.Arg != nil {
		subMatcher, hasSubMatcher = (matcher.Arg).(omegaMatcher)
		if !hasSubMatcher {
			argType := reflect.TypeOf(matcher.Arg)
			if argType.Kind() != reflect.Ptr {
				return false, fmt.Errorf("Cannot assign a value from the channel:\n%s\nTo:\n%s\nYou need to pass a pointer!", format.Object(actual, 1), format.Object(matcher.Arg, 1))
			}
		}
	}

	winnerIndex, value, open := reflect.Select([]reflect.SelectCase{
		{Dir: reflect.SelectRecv, Chan: channelValue},
		{Dir: reflect.SelectDefault},
	})

	var closed bool
	var didReceive bool
	if winnerIndex == 0 {
		closed = !open
		didReceive = open
	}
	matcher.channelClosed = closed

	if closed {
		return false, nil
	}

	if hasSubMatcher {
		if didReceive {
			matcher.receivedValue = value
			return subMatcher.Match(matcher.receivedValue.Interface())
		}
		return false, nil
	}

	if didReceive {
		if matcher.Arg != nil {
			outValue := reflect.ValueOf(matcher.Arg)

			if value.Type().AssignableTo(outValue.Elem().Type()) {
				outValue.Elem().Set(value)
				return true, nil
			}
			if value.Type().Kind() == reflect.Interface && value.Elem().Type().AssignableTo(outValue.Elem().Type()) {
				outValue.Elem().Set(value.Elem())
				return true, nil
			} else {
				return false, fmt.Errorf("Cannot assign a value from the channel:\n%s\nType:\n%s\nTo:\n%s", format.Object(actual, 1), format.Object(value.Interface(), 1), format.Object(matcher.Arg, 1))
			}

		}

		return true, nil
	}
	return false, nil
}

func (matcher *ReceiveMatcher) FailureMessage(actual interface{}) (message string) {
	subMatcher, hasSubMatcher := (matcher.Arg).(omegaMatcher)

	closedAddendum := ""
	if matcher.channelClosed {
		closedAddendum = " The channel is closed."
	}

	if hasSubMatcher {
		if matcher.receivedValue.IsValid() {
			return subMatcher.FailureMessage(matcher.receivedValue.Interface())
		}
		return "When passed a matcher, ReceiveMatcher's channel *must* receive something."
	}
	return format.Message(actual, "to receive something."+closedAddendum)
}

func (matcher *ReceiveMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	subMatcher, hasSubMatcher := (matcher.Arg).(omegaMatcher)

	closedAddendum := ""
	if matcher.channelClosed {
		closedAddendum = " The channel is closed."
	}

	if hasSubMatcher {
		if matcher.receivedValue.IsValid() {
			return subMatcher.NegatedFailureMessage(matcher.receivedValue.Interface())
		}
		return "When passed a matcher, ReceiveMatcher's channel *must* receive something."
	}
	return format.Message(actual, "not to receive anything."+closedAddendum)
}

func (matcher *ReceiveMatcher) MatchMayChangeInTheFuture(actual interface{}) bool {
	if !isChan(actual) {
		return false
	}

	return !matcher.channelClosed
}
