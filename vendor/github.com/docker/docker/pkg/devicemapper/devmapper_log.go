// +build linux,cgo

package devicemapper

import "C"

import (
	"fmt"
	"strings"

	"github.com/sirupsen/logrus"
)

// DevmapperLogger defines methods required to register as a callback for
// logging events received from devicemapper. Note that devicemapper will send
// *all* logs regardless to callbacks (including debug logs) so it's
// recommended to not spam the console with the outputs.
type DevmapperLogger interface {
	// DMLog is the logging callback containing all of the information from
	// devicemapper. The interface is identical to the C libdm counterpart.
	DMLog(level int, file string, line int, dmError int, message string)
}

// dmLogger is the current logger in use that is being forwarded our messages.
var dmLogger DevmapperLogger

// LogInit changes the logging callback called after processing libdm logs for
// error message information. The default logger simply forwards all logs to
// logrus. Calling LogInit(nil) disables the calling of callbacks.
func LogInit(logger DevmapperLogger) {
	dmLogger = logger
}

// Due to the way cgo works this has to be in a separate file, as devmapper.go has
// definitions in the cgo block, which is incompatible with using "//export"

// DevmapperLogCallback exports the devmapper log callback for cgo. Note that
// because we are using callbacks, this function will be called for *every* log
// in libdm (even debug ones because there's no way of setting the verbosity
// level for an external logging callback).
//export DevmapperLogCallback
func DevmapperLogCallback(level C.int, file *C.char, line, dmErrnoOrClass C.int, message *C.char) {
	msg := C.GoString(message)

	// Track what errno libdm saw, because the library only gives us 0 or 1.
	if level < LogLevelDebug {
		if strings.Contains(msg, "busy") {
			dmSawBusy = true
		}

		if strings.Contains(msg, "File exists") {
			dmSawExist = true
		}

		if strings.Contains(msg, "No such device or address") {
			dmSawEnxio = true
		}
	}

	if dmLogger != nil {
		dmLogger.DMLog(int(level), C.GoString(file), int(line), int(dmErrnoOrClass), msg)
	}
}

// DefaultLogger is the default logger used by pkg/devicemapper. It forwards
// all logs that are of higher or equal priority to the given level to the
// corresponding logrus level.
type DefaultLogger struct {
	// Level corresponds to the highest libdm level that will be forwarded to
	// logrus. In order to change this, register a new DefaultLogger.
	Level int
}

// DMLog is the logging callback containing all of the information from
// devicemapper. The interface is identical to the C libdm counterpart.
func (l DefaultLogger) DMLog(level int, file string, line, dmError int, message string) {
	if level <= l.Level {
		// Forward the log to the correct logrus level, if allowed by dmLogLevel.
		logMsg := fmt.Sprintf("libdevmapper(%d): %s:%d (%d) %s", level, file, line, dmError, message)
		switch level {
		case LogLevelFatal, LogLevelErr:
			logrus.Error(logMsg)
		case LogLevelWarn:
			logrus.Warn(logMsg)
		case LogLevelNotice, LogLevelInfo:
			logrus.Info(logMsg)
		case LogLevelDebug:
			logrus.Debug(logMsg)
		default:
			// Don't drop any "unknown" levels.
			logrus.Info(logMsg)
		}
	}
}

// registerLogCallback registers our own logging callback function for libdm
// (which is DevmapperLogCallback).
//
// Because libdm only gives us {0,1} error codes we need to parse the logs
// produced by libdm (to set dmSawBusy and so on). Note that by registering a
// callback using DevmapperLogCallback, libdm will no longer output logs to
// stderr so we have to log everything ourselves. None of this handling is
// optional because we depend on log callbacks to parse the logs, and if we
// don't forward the log information we'll be in a lot of trouble when
// debugging things.
func registerLogCallback() {
	LogWithErrnoInit()
}

func init() {
	// Use the default logger by default. We only allow LogLevelFatal by
	// default, because internally we mask a lot of libdm errors by retrying
	// and similar tricks. Also, libdm is very chatty and we don't want to
	// worry users for no reason.
	dmLogger = DefaultLogger{
		Level: LogLevelFatal,
	}

	// Register as early as possible so we don't miss anything.
	registerLogCallback()
}
