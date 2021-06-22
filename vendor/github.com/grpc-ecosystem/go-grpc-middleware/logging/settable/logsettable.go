package grpc_logsettable

import (
	"io/ioutil"
	"sync"

	"google.golang.org/grpc/grpclog"
)

// SettableLoggerV2 is thread-safe.
type SettableLoggerV2 interface {
	grpclog.LoggerV2
	// Sets given logger as the underlying implementation.
	Set(loggerv2 grpclog.LoggerV2)
	// Sets `discard` logger as the underlying implementation.
	Reset()
}

// ReplaceGrpcLoggerV2 creates and configures SettableLoggerV2 as grpc logger.
func ReplaceGrpcLoggerV2() SettableLoggerV2 {
	settable := &settableLoggerV2{}
	settable.Reset()
	grpclog.SetLoggerV2(settable)
	return settable
}

// SettableLoggerV2 implements SettableLoggerV2
type settableLoggerV2 struct {
	log grpclog.LoggerV2
	mu  sync.RWMutex
}

func (s *settableLoggerV2) Set(log grpclog.LoggerV2) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.log = log
}

func (s *settableLoggerV2) Reset() {
	s.Set(grpclog.NewLoggerV2(ioutil.Discard, ioutil.Discard, ioutil.Discard))
}

func (s *settableLoggerV2) get() grpclog.LoggerV2 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.log
}

func (s *settableLoggerV2) Info(args ...interface{}) {
	s.get().Info(args)
}

func (s *settableLoggerV2) Infoln(args ...interface{}) {
	s.get().Infoln(args)
}

func (s *settableLoggerV2) Infof(format string, args ...interface{}) {
	s.get().Infof(format, args)
}

func (s *settableLoggerV2) Warning(args ...interface{}) {
	s.get().Warning(args)
}

func (s *settableLoggerV2) Warningln(args ...interface{}) {
	s.get().Warningln(args)
}

func (s *settableLoggerV2) Warningf(format string, args ...interface{}) {
	s.get().Warningf(format, args)
}

func (s *settableLoggerV2) Error(args ...interface{}) {
	s.get().Error(args)
}

func (s *settableLoggerV2) Errorln(args ...interface{}) {
	s.get().Errorln(args)
}

func (s *settableLoggerV2) Errorf(format string, args ...interface{}) {
	s.get().Errorf(format, args)
}

func (s *settableLoggerV2) Fatal(args ...interface{}) {
	s.get().Fatal(args)
}

func (s *settableLoggerV2) Fatalln(args ...interface{}) {
	s.get().Fatalln(args)
}

func (s *settableLoggerV2) Fatalf(format string, args ...interface{}) {
	s.get().Fatalf(format, args)
}

func (s *settableLoggerV2) V(l int) bool {
	return s.get().V(l)
}
