package logrus

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"
)

var (
	bufferPool *sync.Pool

	// qualified package name, cached at first use
	logrusPackage string

	// Positions in the call stack when tracing to report the calling method
	minimumCallerDepth int

	// Used for caller information initialisation
	callerInitOnce sync.Once
)

const (
	maximumCallerDepth int = 25
	knownLogrusFrames  int = 4
)

func init() {
	bufferPool = &sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}

	// start at the bottom of the stack before the package-name cache is primed
	minimumCallerDepth = 1
}

// Defines the key when adding errors using WithError.
var ErrorKey = "error"

// An entry is the final or intermediate Logrus logging entry. It contains all
// the fields passed with WithField{,s}. It's finally logged when Trace, Debug,
// Info, Warn, Error, Fatal or Panic is called on it. These objects can be
// reused and passed around as much as you wish to avoid field duplication.
type Entry struct {
	Logger *Logger

	// Contains all the fields set by the user.
	Data Fields

	// Time at which the log entry was created
	Time time.Time

	// Level the log entry was logged at: Trace, Debug, Info, Warn, Error, Fatal or Panic
	// This field will be set on entry firing and the value will be equal to the one in Logger struct field.
	Level Level

	// Calling method, with package name
	Caller *runtime.Frame

	// Message passed to Trace, Debug, Info, Warn, Error, Fatal or Panic
	Message string

	// When formatter is called in entry.log(), a Buffer may be set to entry
	Buffer *bytes.Buffer

	// err may contain a field formatting error
	err string
}

func NewEntry(logger *Logger) *Entry {
	return &Entry{
		Logger: logger,
		// Default is three fields, plus one optional.  Give a little extra room.
		Data: make(Fields, 6),
	}
}

// Returns the string representation from the reader and ultimately the
// formatter.
func (entry *Entry) String() (string, error) {
	serialized, err := entry.Logger.Formatter.Format(entry)
	if err != nil {
		return "", err
	}
	str := string(serialized)
	return str, nil
}

// Add an error as single field (using the key defined in ErrorKey) to the Entry.
func (entry *Entry) WithError(err error) *Entry {
	return entry.WithField(ErrorKey, err)
}

// Add a single field to the Entry.
func (entry *Entry) WithField(key string, value interface{}) *Entry {
	return entry.WithFields(Fields{key: value})
}

// Add a map of fields to the Entry.
func (entry *Entry) WithFields(fields Fields) *Entry {
	data := make(Fields, len(entry.Data)+len(fields))
	for k, v := range entry.Data {
		data[k] = v
	}
	var field_err string
	for k, v := range fields {
		if t := reflect.TypeOf(v); t != nil && t.Kind() == reflect.Func {
			field_err = fmt.Sprintf("can not add field %q", k)
			if entry.err != "" {
				field_err = entry.err + ", " + field_err
			}
		} else {
			data[k] = v
		}
	}
	return &Entry{Logger: entry.Logger, Data: data, Time: entry.Time, err: field_err}
}

// Overrides the time of the Entry.
func (entry *Entry) WithTime(t time.Time) *Entry {
	return &Entry{Logger: entry.Logger, Data: entry.Data, Time: t}
}

// getPackageName reduces a fully qualified function name to the package name
// There really ought to be to be a better way...
func getPackageName(f string) string {
	for {
		lastPeriod := strings.LastIndex(f, ".")
		lastSlash := strings.LastIndex(f, "/")
		if lastPeriod > lastSlash {
			f = f[:lastPeriod]
		} else {
			break
		}
	}

	return f
}

// getCaller retrieves the name of the first non-logrus calling function
func getCaller() *runtime.Frame {
	// Restrict the lookback frames to avoid runaway lookups
	pcs := make([]uintptr, maximumCallerDepth)
	depth := runtime.Callers(minimumCallerDepth, pcs)
	frames := runtime.CallersFrames(pcs[:depth])

	// cache this package's fully-qualified name
	callerInitOnce.Do(func() {
		logrusPackage = getPackageName(runtime.FuncForPC(pcs[0]).Name())

		// now that we have the cache, we can skip a minimum count of known-logrus functions
		// XXX this is dubious, the number of frames may vary store an entry in a logger interface
		minimumCallerDepth = knownLogrusFrames
	})

	for f, again := frames.Next(); again; f, again = frames.Next() {
		pkg := getPackageName(f.Function)

		// If the caller isn't part of this package, we're done
		if pkg != logrusPackage {
			return &f
		}
	}

	// if we got here, we failed to find the caller's context
	return nil
}

func (entry Entry) HasCaller() (has bool) {
	return entry.Logger != nil &&
		entry.Logger.ReportCaller &&
		entry.Caller != nil
}

// This function is not declared with a pointer value because otherwise
// race conditions will occur when using multiple goroutines
func (entry Entry) log(level Level, msg string) {
	var buffer *bytes.Buffer

	// Default to now, but allow users to override if they want.
	//
	// We don't have to worry about polluting future calls to Entry#log()
	// with this assignment because this function is declared with a
	// non-pointer receiver.
	if entry.Time.IsZero() {
		entry.Time = time.Now()
	}

	entry.Level = level
	entry.Message = msg
	if entry.Logger.ReportCaller {
		entry.Caller = getCaller()
	}

	entry.fireHooks()

	buffer = bufferPool.Get().(*bytes.Buffer)
	buffer.Reset()
	defer bufferPool.Put(buffer)
	entry.Buffer = buffer

	entry.write()

	entry.Buffer = nil

	// To avoid Entry#log() returning a value that only would make sense for
	// panic() to use in Entry#Panic(), we avoid the allocation by checking
	// directly here.
	if level <= PanicLevel {
		panic(&entry)
	}
}

func (entry *Entry) fireHooks() {
	entry.Logger.mu.Lock()
	defer entry.Logger.mu.Unlock()
	err := entry.Logger.Hooks.Fire(entry.Level, entry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to fire hook: %v\n", err)
	}
}

func (entry *Entry) write() {
	entry.Logger.mu.Lock()
	defer entry.Logger.mu.Unlock()
	serialized, err := entry.Logger.Formatter.Format(entry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to obtain reader, %v\n", err)
	} else {
		_, err = entry.Logger.Out.Write(serialized)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to write to log, %v\n", err)
		}
	}
}

func (entry *Entry) Trace(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(TraceLevel) {
		entry.log(TraceLevel, fmt.Sprint(args...))
	}
}

func (entry *Entry) Debug(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(DebugLevel) {
		entry.log(DebugLevel, fmt.Sprint(args...))
	}
}

func (entry *Entry) Print(args ...interface{}) {
	entry.Info(args...)
}

func (entry *Entry) Info(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(InfoLevel) {
		entry.log(InfoLevel, fmt.Sprint(args...))
	}
}

func (entry *Entry) Warn(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(WarnLevel) {
		entry.log(WarnLevel, fmt.Sprint(args...))
	}
}

func (entry *Entry) Warning(args ...interface{}) {
	entry.Warn(args...)
}

func (entry *Entry) Error(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(ErrorLevel) {
		entry.log(ErrorLevel, fmt.Sprint(args...))
	}
}

func (entry *Entry) Fatal(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(FatalLevel) {
		entry.log(FatalLevel, fmt.Sprint(args...))
	}
	entry.Logger.Exit(1)
}

func (entry *Entry) Panic(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.log(PanicLevel, fmt.Sprint(args...))
	}
	panic(fmt.Sprint(args...))
}

// Entry Printf family functions

func (entry *Entry) Tracef(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(TraceLevel) {
		entry.Trace(fmt.Sprintf(format, args...))
	}
}

func (entry *Entry) Debugf(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(DebugLevel) {
		entry.Debug(fmt.Sprintf(format, args...))
	}
}

func (entry *Entry) Infof(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(InfoLevel) {
		entry.Info(fmt.Sprintf(format, args...))
	}
}

func (entry *Entry) Printf(format string, args ...interface{}) {
	entry.Infof(format, args...)
}

func (entry *Entry) Warnf(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(WarnLevel) {
		entry.Warn(fmt.Sprintf(format, args...))
	}
}

func (entry *Entry) Warningf(format string, args ...interface{}) {
	entry.Warnf(format, args...)
}

func (entry *Entry) Errorf(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(ErrorLevel) {
		entry.Error(fmt.Sprintf(format, args...))
	}
}

func (entry *Entry) Fatalf(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(FatalLevel) {
		entry.Fatal(fmt.Sprintf(format, args...))
	}
	entry.Logger.Exit(1)
}

func (entry *Entry) Panicf(format string, args ...interface{}) {
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.Panic(fmt.Sprintf(format, args...))
	}
}

// Entry Println family functions

func (entry *Entry) Traceln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(TraceLevel) {
		entry.Trace(entry.sprintlnn(args...))
	}
}

func (entry *Entry) Debugln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(DebugLevel) {
		entry.Debug(entry.sprintlnn(args...))
	}
}

func (entry *Entry) Infoln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(InfoLevel) {
		entry.Info(entry.sprintlnn(args...))
	}
}

func (entry *Entry) Println(args ...interface{}) {
	entry.Infoln(args...)
}

func (entry *Entry) Warnln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(WarnLevel) {
		entry.Warn(entry.sprintlnn(args...))
	}
}

func (entry *Entry) Warningln(args ...interface{}) {
	entry.Warnln(args...)
}

func (entry *Entry) Errorln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(ErrorLevel) {
		entry.Error(entry.sprintlnn(args...))
	}
}

func (entry *Entry) Fatalln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(FatalLevel) {
		entry.Fatal(entry.sprintlnn(args...))
	}
	entry.Logger.Exit(1)
}

func (entry *Entry) Panicln(args ...interface{}) {
	if entry.Logger.IsLevelEnabled(PanicLevel) {
		entry.Panic(entry.sprintlnn(args...))
	}
}

// Sprintlnn => Sprint no newline. This is to get the behavior of how
// fmt.Sprintln where spaces are always added between operands, regardless of
// their type. Instead of vendoring the Sprintln implementation to spare a
// string allocation, we do the simplest thing.
func (entry *Entry) sprintlnn(args ...interface{}) string {
	msg := fmt.Sprintln(args...)
	return msg[:len(msg)-1]
}
