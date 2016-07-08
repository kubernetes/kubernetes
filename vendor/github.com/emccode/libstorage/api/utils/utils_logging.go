package utils

import (
	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gofig"
	"github.com/emccode/libstorage/api/types"
)

// LoggingConfig is the logging configuration.
type LoggingConfig struct {

	// Level is the log level.
	Level log.Level

	// Stdout is the path to the file to which to log stdout.
	Stdout string

	// Stderr is the path to the file to which to log stderr.
	Stderr string

	// HTTPRequests is a flag indicating whether or not to log HTTP requests.
	HTTPRequests bool

	// HTTPResponses is a flag indicating whether or not to log HTTP responses.
	HTTPResponses bool
}

// ParseLoggingConfig returns a new LoggingConfig instance.
func ParseLoggingConfig(
	config gofig.Config,
	fields log.Fields,
	roots ...string) (*LoggingConfig, error) {

	f := func(k string, v interface{}) {
		if fields == nil {
			return
		}
		fields[k] = v
	}

	logConfig := &LoggingConfig{
		Level: log.WarnLevel,
	}

	if lvl, err := log.ParseLevel(
		getString(config, types.ConfigLogLevel, roots...)); err == nil {
		logConfig.Level = lvl
		f(types.ConfigLogLevel, lvl)
	}

	stdOutPath := getString(config, types.ConfigLogStdout, roots...)
	if stdOutPath != "" {
		logConfig.Stdout = stdOutPath
		f(types.ConfigLogStdout, stdOutPath)
	}

	stdErrPath := getString(config, types.ConfigLogStderr, roots...)
	if stdErrPath != "" {
		logConfig.Stderr = stdErrPath
		f(types.ConfigLogStderr, stdErrPath)
	}

	if isSet(config, types.ConfigLogHTTPRequests, roots...) {
		logConfig.HTTPRequests = getBool(
			config, types.ConfigLogHTTPRequests, roots...)
		f(types.ConfigLogHTTPRequests, logConfig.HTTPRequests)
	}

	if isSet(config, types.ConfigLogHTTPResponses, roots...) {
		logConfig.HTTPResponses = getBool(
			config, types.ConfigLogHTTPResponses, roots...)
		f(types.ConfigLogHTTPResponses, logConfig.HTTPResponses)
	}

	return logConfig, nil
}
