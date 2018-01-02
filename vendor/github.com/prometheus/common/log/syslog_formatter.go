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

// +build !windows,!nacl,!plan9

package log

import (
	"fmt"
	"log/syslog"
	"os"

	"github.com/Sirupsen/logrus"
)

var _ logrus.Formatter = (*syslogger)(nil)

func init() {
	setSyslogFormatter = func(appname, local string) error {
		if appname == "" {
			return fmt.Errorf("missing appname parameter")
		}
		if local == "" {
			return fmt.Errorf("missing local parameter")
		}

		fmter, err := newSyslogger(appname, local, origLogger.Formatter)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error creating syslog formatter: %v\n", err)
			origLogger.Errorf("can't connect logger to syslog: %v", err)
			return err
		}
		origLogger.Formatter = fmter
		return nil
	}
}

var prefixTag []byte

type syslogger struct {
	wrap logrus.Formatter
	out  *syslog.Writer
}

func newSyslogger(appname string, facility string, fmter logrus.Formatter) (*syslogger, error) {
	priority, err := getFacility(facility)
	if err != nil {
		return nil, err
	}
	out, err := syslog.New(priority, appname)
	_, isJSON := fmter.(*logrus.JSONFormatter)
	if isJSON {
		// add cee tag to json formatted syslogs
		prefixTag = []byte("@cee:")
	}
	return &syslogger{
		out:  out,
		wrap: fmter,
	}, err
}

func getFacility(facility string) (syslog.Priority, error) {
	switch facility {
	case "0":
		return syslog.LOG_LOCAL0, nil
	case "1":
		return syslog.LOG_LOCAL1, nil
	case "2":
		return syslog.LOG_LOCAL2, nil
	case "3":
		return syslog.LOG_LOCAL3, nil
	case "4":
		return syslog.LOG_LOCAL4, nil
	case "5":
		return syslog.LOG_LOCAL5, nil
	case "6":
		return syslog.LOG_LOCAL6, nil
	case "7":
		return syslog.LOG_LOCAL7, nil
	}
	return syslog.LOG_LOCAL0, fmt.Errorf("invalid local(%s) for syslog", facility)
}

func (s *syslogger) Format(e *logrus.Entry) ([]byte, error) {
	data, err := s.wrap.Format(e)
	if err != nil {
		fmt.Fprintf(os.Stderr, "syslogger: can't format entry: %v\n", err)
		return data, err
	}
	// only append tag to data sent to syslog (line), not to what
	// is returned
	line := string(append(prefixTag, data...))

	switch e.Level {
	case logrus.PanicLevel:
		err = s.out.Crit(line)
	case logrus.FatalLevel:
		err = s.out.Crit(line)
	case logrus.ErrorLevel:
		err = s.out.Err(line)
	case logrus.WarnLevel:
		err = s.out.Warning(line)
	case logrus.InfoLevel:
		err = s.out.Info(line)
	case logrus.DebugLevel:
		err = s.out.Debug(line)
	default:
		err = s.out.Notice(line)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "syslogger: can't send log to syslog: %v\n", err)
	}

	return data, err
}
