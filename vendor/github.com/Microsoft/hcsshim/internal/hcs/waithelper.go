package hcs

import (
	"time"

	"github.com/sirupsen/logrus"
)

func processAsyncHcsResult(err error, resultp *uint16, callbackNumber uintptr, expectedNotification hcsNotification, timeout *time.Duration) ([]ErrorEvent, error) {
	events := processHcsResult(resultp)
	if IsPending(err) {
		return nil, waitForNotification(callbackNumber, expectedNotification, timeout)
	}

	return events, err
}

func waitForNotification(callbackNumber uintptr, expectedNotification hcsNotification, timeout *time.Duration) error {
	callbackMapLock.RLock()
	if _, ok := callbackMap[callbackNumber]; !ok {
		callbackMapLock.RUnlock()
		logrus.Errorf("failed to waitForNotification: callbackNumber %d does not exist in callbackMap", callbackNumber)
		return ErrHandleClose
	}
	channels := callbackMap[callbackNumber].channels
	callbackMapLock.RUnlock()

	expectedChannel := channels[expectedNotification]
	if expectedChannel == nil {
		logrus.Errorf("unknown notification type in waitForNotification %x", expectedNotification)
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
	return nil
}
