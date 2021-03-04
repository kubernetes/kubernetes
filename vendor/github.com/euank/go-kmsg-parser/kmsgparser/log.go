/*
Copyright 2016 Euan Kemp

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kmsgparser

import stdlog "log"

// Logger is a glog compatible logging interface
// The StandardLogger struct can be used to wrap a log.Logger from the golang
// "log" package to create a standard a logger fulfilling this interface as
// well.
type Logger interface {
	Warningf(string, ...interface{})
	Infof(string, ...interface{})
	Errorf(string, ...interface{})
}

// StandardLogger adapts the "log" package's Logger interface to be a Logger
type StandardLogger struct {
	*stdlog.Logger
}

func (s *StandardLogger) Warningf(fmt string, args ...interface{}) {
	if s.Logger == nil {
		return
	}
	s.Logger.Printf("[WARNING] "+fmt, args)
}

func (s *StandardLogger) Infof(fmt string, args ...interface{}) {
	if s.Logger == nil {
		return
	}
	s.Logger.Printf("[INFO] "+fmt, args)
}

func (s *StandardLogger) Errorf(fmt string, args ...interface{}) {
	if s.Logger == nil {
		return
	}
	s.Logger.Printf("[INFO] "+fmt, args)
}
