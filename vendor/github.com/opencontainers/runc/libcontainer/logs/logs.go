package logs

import (
	"bufio"
	"encoding/json"
	"io"

	"github.com/sirupsen/logrus"
)

func ForwardLogs(logPipe io.ReadCloser) chan error {
	done := make(chan error, 1)
	s := bufio.NewScanner(logPipe)

	logger := logrus.StandardLogger()
	if logger.ReportCaller {
		// Need a copy of the standard logger, but with ReportCaller
		// turned off, as the logs are merely forwarded and their
		// true source is not this file/line/function.
		logNoCaller := *logrus.StandardLogger()
		logNoCaller.ReportCaller = false
		logger = &logNoCaller
	}

	go func() {
		for s.Scan() {
			processEntry(s.Bytes(), logger)
		}
		if err := logPipe.Close(); err != nil {
			logrus.Errorf("error closing log source: %v", err)
		}
		// The only error we want to return is when reading from
		// logPipe has failed.
		done <- s.Err()
		close(done)
	}()

	return done
}

func processEntry(text []byte, logger *logrus.Logger) {
	if len(text) == 0 {
		return
	}

	var jl struct {
		Level logrus.Level `json:"level"`
		Msg   string       `json:"msg"`
	}
	if err := json.Unmarshal(text, &jl); err != nil {
		logrus.Errorf("failed to decode %q to json: %v", text, err)
		return
	}

	logger.Log(jl.Level, jl.Msg)
}
