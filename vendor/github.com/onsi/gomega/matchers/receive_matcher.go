// untested sections: 3

package matchers

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type ReceiveMatcher struct {
	Args          []any
	receivedValue reflect.Value
	channelClosed bool
}

func (matcher *ReceiveMatcher) Match(actual any) (success bool, err error) {
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
	var resultReference any

	// Valid arg formats are as follows, always with optional POINTER before
	// optional MATCHER:
	//   - Receive()
	//   - Receive(POINTER)
	//   - Receive(MATCHER)
	//   - Receive(POINTER, MATCHER)
	args := matcher.Args
	if len(args) > 0 {
		arg := args[0]
		_, isSubMatcher := arg.(omegaMatcher)
		if !isSubMatcher && reflect.ValueOf(arg).Kind() == reflect.Ptr {
			// Consume optional POINTER arg first, if it ain't no matcher ;)
			resultReference = arg
			args = args[1:]
		}
	}
	if len(args) > 0 {
		arg := args[0]
		subMatcher, hasSubMatcher = arg.(omegaMatcher)
		if !hasSubMatcher {
			// At this point we assume the dev user wanted to assign a received
			// value, so [POINTER,]MATCHER.
			return false, fmt.Errorf("Cannot assign a value from the channel:\n%s\nTo:\n%s\nYou need to pass a pointer!", format.Object(actual, 1), format.Object(arg, 1))
		}
		// Consume optional MATCHER arg.
		args = args[1:]
	}
	if len(args) > 0 {
		// If there are still args present, reject all.
		return false, errors.New("Receive matcher expects at most an optional pointer and/or an optional matcher")
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
		if !didReceive {
			return false, nil
		}
		matcher.receivedValue = value
		if match, err := subMatcher.Match(matcher.receivedValue.Interface()); err != nil || !match {
			return match, err
		}
		// if we received a match, then fall through in order to handle an
		// optional assignment of the received value to the specified reference.
	}

	if didReceive {
		if resultReference != nil {
			outValue := reflect.ValueOf(resultReference)

			if value.Type().AssignableTo(outValue.Elem().Type()) {
				outValue.Elem().Set(value)
				return true, nil
			}
			if value.Type().Kind() == reflect.Interface && value.Elem().Type().AssignableTo(outValue.Elem().Type()) {
				outValue.Elem().Set(value.Elem())
				return true, nil
			} else {
				return false, fmt.Errorf("Cannot assign a value from the channel:\n%s\nType:\n%s\nTo:\n%s", format.Object(actual, 1), format.Object(value.Interface(), 1), format.Object(resultReference, 1))
			}

		}

		return true, nil
	}
	return false, nil
}

func (matcher *ReceiveMatcher) FailureMessage(actual any) (message string) {
	var matcherArg any
	if len(matcher.Args) > 0 {
		matcherArg = matcher.Args[len(matcher.Args)-1]
	}
	subMatcher, hasSubMatcher := (matcherArg).(omegaMatcher)

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

func (matcher *ReceiveMatcher) NegatedFailureMessage(actual any) (message string) {
	var matcherArg any
	if len(matcher.Args) > 0 {
		matcherArg = matcher.Args[len(matcher.Args)-1]
	}
	subMatcher, hasSubMatcher := (matcherArg).(omegaMatcher)

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

func (matcher *ReceiveMatcher) MatchMayChangeInTheFuture(actual any) bool {
	if !isChan(actual) {
		return false
	}

	return !matcher.channelClosed
}
