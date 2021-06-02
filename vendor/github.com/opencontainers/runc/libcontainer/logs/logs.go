package logs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

var (
	configureMutex sync.Mutex
	// loggingConfigured will be set once logging has been configured via invoking `ConfigureLogging`.
	// Subsequent invocations of `ConfigureLogging` would be no-op
	loggingConfigured = false
)

type Config struct {
	LogLevel    logrus.Level
	LogFormat   string
	LogFilePath string
	LogPipeFd   int
	LogCaller   bool
}

func ForwardLogs(logPipe io.ReadCloser) chan error {
	done := make(chan error, 1)
	s := bufio.NewScanner(logPipe)

	go func() {
		for s.Scan() {
			processEntry(s.Bytes())
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

func processEntry(text []byte) {
	if len(text) == 0 {
		return
	}

	var jl struct {
		Level string `json:"level"`
		Msg   string `json:"msg"`
	}
	if err := json.Unmarshal(text, &jl); err != nil {
		logrus.Errorf("failed to decode %q to json: %v", text, err)
		return
	}

	lvl, err := logrus.ParseLevel(jl.Level)
	if err != nil {
		logrus.Errorf("failed to parse log level %q: %v", jl.Level, err)
		return
	}
	logrus.StandardLogger().Logf(lvl, jl.Msg)
}

func ConfigureLogging(config Config) error {
	configureMutex.Lock()
	defer configureMutex.Unlock()

	if loggingConfigured {
		return errors.New("logging has already been configured")
	}

	logrus.SetLevel(config.LogLevel)
	logrus.SetReportCaller(config.LogCaller)

	// XXX: while 0 is a valid fd (usually stdin), here we assume
	// that we never deliberately set LogPipeFd to 0.
	if config.LogPipeFd > 0 {
		logrus.SetOutput(os.NewFile(uintptr(config.LogPipeFd), "logpipe"))
	} else if config.LogFilePath != "" {
		f, err := os.OpenFile(config.LogFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND|os.O_SYNC, 0644)
		if err != nil {
			return err
		}
		logrus.SetOutput(f)
	}

	switch config.LogFormat {
	case "text":
		// retain logrus's default.
	case "json":
		logrus.SetFormatter(new(logrus.JSONFormatter))
	default:
		return fmt.Errorf("unknown log-format %q", config.LogFormat)
	}

	loggingConfigured = true
	return nil
}
