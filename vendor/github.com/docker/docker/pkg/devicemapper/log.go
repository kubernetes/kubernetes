package devicemapper

// definitions from lvm2 lib/log/log.h
const (
	LogLevelFatal  = 2 + iota // _LOG_FATAL
	LogLevelErr               // _LOG_ERR
	LogLevelWarn              // _LOG_WARN
	LogLevelNotice            // _LOG_NOTICE
	LogLevelInfo              // _LOG_INFO
	LogLevelDebug             // _LOG_DEBUG
)
