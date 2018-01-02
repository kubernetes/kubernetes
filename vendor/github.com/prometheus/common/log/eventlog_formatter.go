// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build windows

package log

import (
	"fmt"
	"os"

	"golang.org/x/sys/windows/svc/eventlog"

	"github.com/Sirupsen/logrus"
)

func init() {
	setEventlogFormatter = func(name string, debugAsInfo bool) error {
		if name == "" {
			return fmt.Errorf("missing name parameter")
		}

		fmter, err := newEventlogger(name, debugAsInfo, origLogger.Formatter)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error creating eventlog formatter: %v\n", err)
			origLogger.Errorf("can't connect logger to eventlog: %v", err)
			return err
		}
		origLogger.Formatter = fmter
		return nil
	}
}

type eventlogger struct {
	log         *eventlog.Log
	debugAsInfo bool
	wrap        logrus.Formatter
}

func newEventlogger(name string, debugAsInfo bool, fmter logrus.Formatter) (*eventlogger, error) {
	logHandle, err := eventlog.Open(name)
	if err != nil {
		return nil, err
	}
	return &eventlogger{log: logHandle, debugAsInfo: debugAsInfo, wrap: fmter}, nil
}

func (s *eventlogger) Format(e *logrus.Entry) ([]byte, error) {
	data, err := s.wrap.Format(e)
	if err != nil {
		fmt.Fprintf(os.Stderr, "eventlogger: can't format entry: %v\n", err)
		return data, err
	}

	switch e.Level {
	case logrus.PanicLevel:
		fallthrough
	case logrus.FatalLevel:
		fallthrough
	case logrus.ErrorLevel:
		err = s.log.Error(102, e.Message)
	case logrus.WarnLevel:
		err = s.log.Warning(101, e.Message)
	case logrus.InfoLevel:
		err = s.log.Info(100, e.Message)
	case logrus.DebugLevel:
		if s.debugAsInfo {
			err = s.log.Info(100, e.Message)
		}
	default:
		err = s.log.Info(100, e.Message)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "eventlogger: can't send log to eventlog: %v\n", err)
	}

	return data, err
}
