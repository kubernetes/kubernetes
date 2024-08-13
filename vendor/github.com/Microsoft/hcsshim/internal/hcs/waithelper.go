package hcs

import (
	"context"
	"time"

	"github.com/Microsoft/hcsshim/internal/log"
)

func processAsyncHcsResult(ctx context.Context, err error, resultJSON string, callbackNumber uintptr, expectedNotification hcsNotification, timeout *time.Duration) ([]ErrorEvent, error) {
	events := processHcsResult(ctx, resultJSON)
	if IsPending(err) {
		return nil, waitForNotification(ctx, callbackNumber, expectedNotification, timeout)
	}

	return events, err
}

func waitForNotification(ctx context.Context, callbackNumber uintptr, expectedNotification hcsNotification, timeout *time.Duration) error {
	callbackMapLock.RLock()
	if _, ok := callbackMap[callbackNumber]; !ok {
		callbackMapLock.RUnlock()
		log.G(ctx).WithField("callbackNumber", callbackNumber).Error("failed to waitForNotification: callbackNumber does not exist in callbackMap")
		return ErrHandleClose
	}
	channels := callbackMap[callbackNumber].channels
	callbackMapLock.RUnlock()

	expectedChannel := channels[expectedNotification]
	if expectedChannel == nil {
		log.G(ctx).WithField("type", expectedNotification).Error("unknown notification type in waitForNotification")
		return ErrInvalidNotificationType
	}

	var c <-chan time.Time
	if timeout != nil {
		timer := time.NewTimer(*timeout)
		c = timer.C
		defer timer.Stop()
	}

	select {
	case err, ok := <-expectedChannel:
		if !ok {
			return ErrHandleClose
		}
		return err
	case err, ok := <-channels[hcsNotificationSystemExited]:
		if !ok {
			return ErrHandleClose
		}
		// If the expected notification is hcsNotificationSystemExited which of the two selects
		// chosen is random. Return the raw error if hcsNotificationSystemExited is expected
		if channels[hcsNotificationSystemExited] == expectedChannel {
			return err
		}
		return ErrUnexpectedContainerExit
	case _, ok := <-channels[hcsNotificationServiceDisconnect]:
		if !ok {
			return ErrHandleClose
		}
		// hcsNotificationServiceDisconnect should never be an expected notification
		// it does not need the same handling as hcsNotificationSystemExited
		return ErrUnexpectedProcessAbort
	case <-c:
		return ErrTimeout
	}
}
