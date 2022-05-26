package logutils

import (
	"github.com/stretchr/testify/mock"
)

type MockLog struct {
	mock.Mock
}

func NewMockLog() *MockLog {
	return &MockLog{}
}

func (m *MockLog) Fatalf(format string, args ...interface{}) {
	mArgs := []interface{}{format}
	m.Called(append(mArgs, args...)...)
}

func (m *MockLog) Panicf(format string, args ...interface{}) {
	mArgs := []interface{}{format}
	m.Called(append(mArgs, args...)...)
}

func (m *MockLog) Errorf(format string, args ...interface{}) {
	mArgs := []interface{}{format}
	m.Called(append(mArgs, args...)...)
}

func (m *MockLog) Warnf(format string, args ...interface{}) {
	mArgs := []interface{}{format}
	m.Called(append(mArgs, args...)...)
}

func (m *MockLog) Infof(format string, args ...interface{}) {
	mArgs := []interface{}{format}
	m.Called(append(mArgs, args...)...)
}

func (m *MockLog) Child(name string) Log {
	m.Called(name)
	return m
}

func (m *MockLog) SetLevel(level LogLevel) {
	m.Called(level)
}
