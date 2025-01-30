package internal

import (
	"errors"
	"fmt"
	"time"
)

type PollingSignalErrorType int

const (
	PollingSignalErrorTypeStopTrying PollingSignalErrorType = iota
	PollingSignalErrorTypeTryAgainAfter
)

type PollingSignalError interface {
	error
	Wrap(err error) PollingSignalError
	Attach(description string, obj any) PollingSignalError
	Successfully() PollingSignalError
	Now()
}

var StopTrying = func(message string) PollingSignalError {
	return &PollingSignalErrorImpl{
		message:                message,
		pollingSignalErrorType: PollingSignalErrorTypeStopTrying,
	}
}

var TryAgainAfter = func(duration time.Duration) PollingSignalError {
	return &PollingSignalErrorImpl{
		message:                fmt.Sprintf("told to try again after %s", duration),
		duration:               duration,
		pollingSignalErrorType: PollingSignalErrorTypeTryAgainAfter,
	}
}

type PollingSignalErrorAttachment struct {
	Description string
	Object      any
}

type PollingSignalErrorImpl struct {
	message                string
	wrappedErr             error
	pollingSignalErrorType PollingSignalErrorType
	duration               time.Duration
	successful             bool
	Attachments            []PollingSignalErrorAttachment
}

func (s *PollingSignalErrorImpl) Wrap(err error) PollingSignalError {
	s.wrappedErr = err
	return s
}

func (s *PollingSignalErrorImpl) Attach(description string, obj any) PollingSignalError {
	s.Attachments = append(s.Attachments, PollingSignalErrorAttachment{description, obj})
	return s
}

func (s *PollingSignalErrorImpl) Error() string {
	if s.wrappedErr == nil {
		return s.message
	} else {
		return s.message + ": " + s.wrappedErr.Error()
	}
}

func (s *PollingSignalErrorImpl) Unwrap() error {
	if s == nil {
		return nil
	}
	return s.wrappedErr
}

func (s *PollingSignalErrorImpl) Successfully() PollingSignalError {
	s.successful = true
	return s
}

func (s *PollingSignalErrorImpl) Now() {
	panic(s)
}

func (s *PollingSignalErrorImpl) IsStopTrying() bool {
	return s.pollingSignalErrorType == PollingSignalErrorTypeStopTrying
}

func (s *PollingSignalErrorImpl) IsSuccessful() bool {
	return s.successful
}

func (s *PollingSignalErrorImpl) IsTryAgainAfter() bool {
	return s.pollingSignalErrorType == PollingSignalErrorTypeTryAgainAfter
}

func (s *PollingSignalErrorImpl) TryAgainDuration() time.Duration {
	return s.duration
}

func AsPollingSignalError(actual interface{}) (*PollingSignalErrorImpl, bool) {
	if actual == nil {
		return nil, false
	}
	if actualErr, ok := actual.(error); ok {
		var target *PollingSignalErrorImpl
		if errors.As(actualErr, &target) {
			return target, true
		} else {
			return nil, false
		}
	}

	return nil, false
}
