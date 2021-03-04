package logs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"

	"github.com/sirupsen/logrus"
)

var (
	configureMutex = sync.Mutex{}
	// loggingConfigured will be set once logging has been configured via invoking `ConfigureLogging`.
	// Subsequent invocations of `ConfigureLogging` would be no-op
	loggingConfigured = false
)

type Config struct {
	LogLevel    logrus.Level
	LogFormat   string
	LogFilePath string
	LogPipeFd   string
}

func ForwardLogs(logPipe io.Reader) {
	lineReader := bufio.NewReader(logPipe)
	for {
		line, err := lineReader.ReadBytes('\n')
		if len(line) > 0 {
			processEntry(line)
		}
		if err == io.EOF {
			logrus.Debugf("log pipe has been closed: %+v", err)
			return
		}
		if err != nil {
			logrus.Errorf("log pipe read error: %+v", err)
		}
	}
}

func processEntry(text []byte) {
	type jsonLog struct {
		Level string `json:"level"`
		Msg   string `json:"msg"`
	}

	var jl jsonLog
	if err := json.Unmarshal(text, &jl); err != nil {
		logrus.Errorf("failed to decode %q to json: %+v", text, err)
		return
	}

	lvl, err := logrus.ParseLevel(jl.Level)
	if err != nil {
		logrus.Errorf("failed to parse log level %q: %v\n", jl.Level, err)
		return
	}
	logrus.StandardLogger().Logf(lvl, jl.Msg)
}

func ConfigureLogging(config Config) error {
	configureMutex.Lock()
	defer configureMutex.Unlock()

	if loggingConfigured {
		logrus.Debug("logging has already been configured")
		return nil
	}

	logrus.SetLevel(config.LogLevel)

	if config.LogPipeFd != "" {
		logPipeFdInt, err := strconv.Atoi(config.LogPipeFd)
		if err != nil {
			return fmt.Errorf("failed to convert _LIBCONTAINER_LOGPIPE environment variable value %q to int: %v", config.LogPipeFd, err)
		}
		logrus.SetOutput(os.NewFile(uintptr(logPipeFdInt), "logpipe"))
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
