package logrus

import "time"

const (
	// defaultTimestampFormat is the layout used to format entry timestamps
	// when a formatter has not specified a custom TimestampFormat.
	// It follows time.RFC3339 and is applied unless timestamps are disabled.
	defaultTimestampFormat = time.RFC3339

	// defaultFields is the number of commonly included predefined log entry fields
	// (msg, level, time). It is used as a capacity hint when constructing
	// intermediate collections during formatting (for example, the fixed key list).
	//
	// It does not include the optional "logrus_error", "func", or "file" fields.
	defaultFields = 3
)

// Default key names for the default fields
const (
	FieldKeyMsg         = "msg"
	FieldKeyLevel       = "level"
	FieldKeyTime        = "time"
	FieldKeyLogrusError = "logrus_error"
	FieldKeyFunc        = "func"
	FieldKeyFile        = "file"
)

// Formatter is implemented by types that format log entries. It receives an
// [*Entry], which contains:
//
//   - entry.Message: the message passed to logging methods such as [Info], [Warn], [Error]
//   - entry.Time: the timestamp
//   - entry.Level: the log level
//
// Additional fields added with [WithField] or [WithFields] are available in
// [Entry.Data]. Format should return the formatted log entry as a byte slice,
// which is written to [Logger.Out].
type Formatter interface {
	Format(*Entry) ([]byte, error)
}

// This is to not silently overwrite `time`, `msg`, `func` and `level` fields when
// dumping it. If this code wasn't there doing:
//
//	logrus.WithField("level", 1).Info("hello")
//
// Would just silently drop the user provided level. Instead with this code
// it'll logged as:
//
//	{"level": "info", "fields.level": 1, "msg": "hello", "time": "..."}
//
// It's not exported because it's still using Data in an opinionated way. It's to
// avoid code duplication between the two default formatters.
func prefixFieldClashes(data Fields, fieldMap FieldMap, reportCaller bool) {
	timeKey := fieldMap.resolve(FieldKeyTime)
	if t, ok := data[timeKey]; ok {
		data["fields."+timeKey] = t
		delete(data, timeKey)
	}

	msgKey := fieldMap.resolve(FieldKeyMsg)
	if m, ok := data[msgKey]; ok {
		data["fields."+msgKey] = m
		delete(data, msgKey)
	}

	levelKey := fieldMap.resolve(FieldKeyLevel)
	if l, ok := data[levelKey]; ok {
		data["fields."+levelKey] = l
		delete(data, levelKey)
	}

	logrusErrKey := fieldMap.resolve(FieldKeyLogrusError)
	if l, ok := data[logrusErrKey]; ok {
		data["fields."+logrusErrKey] = l
		delete(data, logrusErrKey)
	}

	// If reportCaller is not set, 'func' will not conflict.
	if reportCaller {
		funcKey := fieldMap.resolve(FieldKeyFunc)
		if l, ok := data[funcKey]; ok {
			data["fields."+funcKey] = l
		}
		fileKey := fieldMap.resolve(FieldKeyFile)
		if l, ok := data[fileKey]; ok {
			data["fields."+fileKey] = l
		}
	}
}
