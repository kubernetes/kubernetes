/*
Copyright 2021 The logr Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package funcr implements formatting of structured log messages and
// optionally captures the call site and timestamp.
//
// The simplest way to use it is via its implementation of a
// github.com/go-logr/logr.LogSink with output through an arbitrary
// "write" function.  See New and NewJSON for details.
//
// # Custom LogSinks
//
// For users who need more control, a funcr.Formatter can be embedded inside
// your own custom LogSink implementation. This is useful when the LogSink
// needs to implement additional methods, for example.
//
// # Formatting
//
// This will respect logr.Marshaler, fmt.Stringer, and error interfaces for
// values which are being logged.  When rendering a struct, funcr will use Go's
// standard JSON tags (all except "string").
package funcr

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
)

// New returns a logr.Logger which is implemented by an arbitrary function.
func New(fn func(prefix, args string), opts Options) logr.Logger {
	return logr.New(newSink(fn, NewFormatter(opts)))
}

// NewJSON returns a logr.Logger which is implemented by an arbitrary function
// and produces JSON output.
func NewJSON(fn func(obj string), opts Options) logr.Logger {
	fnWrapper := func(_, obj string) {
		fn(obj)
	}
	return logr.New(newSink(fnWrapper, NewFormatterJSON(opts)))
}

// Underlier exposes access to the underlying logging function. Since
// callers only have a logr.Logger, they have to know which
// implementation is in use, so this interface is less of an
// abstraction and more of a way to test type conversion.
type Underlier interface {
	GetUnderlying() func(prefix, args string)
}

func newSink(fn func(prefix, args string), formatter Formatter) logr.LogSink {
	l := &fnlogger{
		Formatter: formatter,
		write:     fn,
	}
	// For skipping fnlogger.Info and fnlogger.Error.
	l.AddCallDepth(1) // via Formatter
	return l
}

// Options carries parameters which influence the way logs are generated.
type Options struct {
	// LogCaller tells funcr to add a "caller" key to some or all log lines.
	// This has some overhead, so some users might not want it.
	LogCaller MessageClass

	// LogCallerFunc tells funcr to also log the calling function name.  This
	// has no effect if caller logging is not enabled (see Options.LogCaller).
	LogCallerFunc bool

	// LogTimestamp tells funcr to add a "ts" key to log lines.  This has some
	// overhead, so some users might not want it.
	LogTimestamp bool

	// TimestampFormat tells funcr how to render timestamps when LogTimestamp
	// is enabled.  If not specified, a default format will be used.  For more
	// details, see docs for Go's time.Layout.
	TimestampFormat string

	// LogInfoLevel tells funcr what key to use to log the info level.
	// If not specified, the info level will be logged as "level".
	// If this is set to "", the info level will not be logged at all.
	LogInfoLevel *string

	// Verbosity tells funcr which V logs to produce.  Higher values enable
	// more logs.  Info logs at or below this level will be written, while logs
	// above this level will be discarded.
	Verbosity int

	// RenderBuiltinsHook allows users to mutate the list of key-value pairs
	// while a log line is being rendered.  The kvList argument follows logr
	// conventions - each pair of slice elements is comprised of a string key
	// and an arbitrary value (verified and sanitized before calling this
	// hook).  The value returned must follow the same conventions.  This hook
	// can be used to audit or modify logged data.  For example, you might want
	// to prefix all of funcr's built-in keys with some string.  This hook is
	// only called for built-in (provided by funcr itself) key-value pairs.
	// Equivalent hooks are offered for key-value pairs saved via
	// logr.Logger.WithValues or Formatter.AddValues (see RenderValuesHook) and
	// for user-provided pairs (see RenderArgsHook).
	RenderBuiltinsHook func(kvList []any) []any

	// RenderValuesHook is the same as RenderBuiltinsHook, except that it is
	// only called for key-value pairs saved via logr.Logger.WithValues.  See
	// RenderBuiltinsHook for more details.
	RenderValuesHook func(kvList []any) []any

	// RenderArgsHook is the same as RenderBuiltinsHook, except that it is only
	// called for key-value pairs passed directly to Info and Error.  See
	// RenderBuiltinsHook for more details.
	RenderArgsHook func(kvList []any) []any

	// MaxLogDepth tells funcr how many levels of nested fields (e.g. a struct
	// that contains a struct, etc.) it may log.  Every time it finds a struct,
	// slice, array, or map the depth is increased by one.  When the maximum is
	// reached, the value will be converted to a string indicating that the max
	// depth has been exceeded.  If this field is not specified, a default
	// value will be used.
	MaxLogDepth int
}

// MessageClass indicates which category or categories of messages to consider.
type MessageClass int

const (
	// None ignores all message classes.
	None MessageClass = iota
	// All considers all message classes.
	All
	// Info only considers info messages.
	Info
	// Error only considers error messages.
	Error
)

// fnlogger inherits some of its LogSink implementation from Formatter
// and just needs to add some glue code.
type fnlogger struct {
	Formatter
	write func(prefix, args string)
}

func (l fnlogger) WithName(name string) logr.LogSink {
	l.AddName(name) // via Formatter
	return &l
}

func (l fnlogger) WithValues(kvList ...any) logr.LogSink {
	l.AddValues(kvList) // via Formatter
	return &l
}

func (l fnlogger) WithCallDepth(depth int) logr.LogSink {
	l.AddCallDepth(depth) // via Formatter
	return &l
}

func (l fnlogger) Info(level int, msg string, kvList ...any) {
	prefix, args := l.FormatInfo(level, msg, kvList)
	l.write(prefix, args)
}

func (l fnlogger) Error(err error, msg string, kvList ...any) {
	prefix, args := l.FormatError(err, msg, kvList)
	l.write(prefix, args)
}

func (l fnlogger) GetUnderlying() func(prefix, args string) {
	return l.write
}

// Assert conformance to the interfaces.
var _ logr.LogSink = &fnlogger{}
var _ logr.CallDepthLogSink = &fnlogger{}
var _ Underlier = &fnlogger{}

// NewFormatter constructs a Formatter which emits a JSON-like key=value format.
func NewFormatter(opts Options) Formatter {
	return newFormatter(opts, outputKeyValue)
}

// NewFormatterJSON constructs a Formatter which emits strict JSON.
func NewFormatterJSON(opts Options) Formatter {
	return newFormatter(opts, outputJSON)
}

// Defaults for Options.
const defaultTimestampFormat = "2006-01-02 15:04:05.000000"
const defaultMaxLogDepth = 16

func newFormatter(opts Options, outfmt outputFormat) Formatter {
	if opts.TimestampFormat == "" {
		opts.TimestampFormat = defaultTimestampFormat
	}
	if opts.MaxLogDepth == 0 {
		opts.MaxLogDepth = defaultMaxLogDepth
	}
	if opts.LogInfoLevel == nil {
		opts.LogInfoLevel = new(string)
		*opts.LogInfoLevel = "level"
	}
	f := Formatter{
		outputFormat: outfmt,
		prefix:       "",
		values:       nil,
		depth:        0,
		opts:         &opts,
	}
	return f
}

// Formatter is an opaque struct which can be embedded in a LogSink
// implementation. It should be constructed with NewFormatter. Some of
// its methods directly implement logr.LogSink.
type Formatter struct {
	outputFormat outputFormat
	prefix       string
	values       []any
	valuesStr    string
	depth        int
	opts         *Options
	groupName    string // for slog groups
	groups       []groupDef
}

// outputFormat indicates which outputFormat to use.
type outputFormat int

const (
	// outputKeyValue emits a JSON-like key=value format, but not strict JSON.
	outputKeyValue outputFormat = iota
	// outputJSON emits strict JSON.
	outputJSON
)

// groupDef represents a saved group.  The values may be empty, but we don't
// know if we need to render the group until the final record is rendered.
type groupDef struct {
	name   string
	values string
}

// PseudoStruct is a list of key-value pairs that gets logged as a struct.
type PseudoStruct []any

// render produces a log line, ready to use.
func (f Formatter) render(builtins, args []any) string {
	// Empirically bytes.Buffer is faster than strings.Builder for this.
	buf := bytes.NewBuffer(make([]byte, 0, 1024))

	if f.outputFormat == outputJSON {
		buf.WriteByte('{') // for the whole record
	}

	// Render builtins
	vals := builtins
	if hook := f.opts.RenderBuiltinsHook; hook != nil {
		vals = hook(f.sanitize(vals))
	}
	f.flatten(buf, vals, false) // keys are ours, no need to escape
	continuing := len(builtins) > 0

	// Turn the inner-most group into a string
	argsStr := func() string {
		buf := bytes.NewBuffer(make([]byte, 0, 1024))

		vals = args
		if hook := f.opts.RenderArgsHook; hook != nil {
			vals = hook(f.sanitize(vals))
		}
		f.flatten(buf, vals, true) // escape user-provided keys

		return buf.String()
	}()

	// Render the stack of groups from the inside out.
	bodyStr := f.renderGroup(f.groupName, f.valuesStr, argsStr)
	for i := len(f.groups) - 1; i >= 0; i-- {
		grp := &f.groups[i]
		if grp.values == "" && bodyStr == "" {
			// no contents, so we must elide the whole group
			continue
		}
		bodyStr = f.renderGroup(grp.name, grp.values, bodyStr)
	}

	if bodyStr != "" {
		if continuing {
			buf.WriteByte(f.comma())
		}
		buf.WriteString(bodyStr)
	}

	if f.outputFormat == outputJSON {
		buf.WriteByte('}') // for the whole record
	}

	return buf.String()
}

// renderGroup returns a string representation of the named group with rendered
// values and args.  If the name is empty, this will return the values and args,
// joined.  If the name is not empty, this will return a single key-value pair,
// where the value is a grouping of the values and args.  If the values and
// args are both empty, this will return an empty string, even if the name was
// specified.
func (f Formatter) renderGroup(name string, values string, args string) string {
	buf := bytes.NewBuffer(make([]byte, 0, 1024))

	needClosingBrace := false
	if name != "" && (values != "" || args != "") {
		buf.WriteString(f.quoted(name, true)) // escape user-provided keys
		buf.WriteByte(f.colon())
		buf.WriteByte('{')
		needClosingBrace = true
	}

	continuing := false
	if values != "" {
		buf.WriteString(values)
		continuing = true
	}

	if args != "" {
		if continuing {
			buf.WriteByte(f.comma())
		}
		buf.WriteString(args)
	}

	if needClosingBrace {
		buf.WriteByte('}')
	}

	return buf.String()
}

// flatten renders a list of key-value pairs into a buffer.  If escapeKeys is
// true, the keys are assumed to have non-JSON-compatible characters in them
// and must be evaluated for escapes.
//
// This function returns a potentially modified version of kvList, which
// ensures that there is a value for every key (adding a value if needed) and
// that each key is a string (substituting a key if needed).
func (f Formatter) flatten(buf *bytes.Buffer, kvList []any, escapeKeys bool) []any {
	// This logic overlaps with sanitize() but saves one type-cast per key,
	// which can be measurable.
	if len(kvList)%2 != 0 {
		kvList = append(kvList, noValue)
	}
	copied := false
	for i := 0; i < len(kvList); i += 2 {
		k, ok := kvList[i].(string)
		if !ok {
			if !copied {
				newList := make([]any, len(kvList))
				copy(newList, kvList)
				kvList = newList
				copied = true
			}
			k = f.nonStringKey(kvList[i])
			kvList[i] = k
		}
		v := kvList[i+1]

		if i > 0 {
			if f.outputFormat == outputJSON {
				buf.WriteByte(f.comma())
			} else {
				// In theory the format could be something we don't understand.  In
				// practice, we control it, so it won't be.
				buf.WriteByte(' ')
			}
		}

		buf.WriteString(f.quoted(k, escapeKeys))
		buf.WriteByte(f.colon())
		buf.WriteString(f.pretty(v))
	}
	return kvList
}

func (f Formatter) quoted(str string, escape bool) string {
	if escape {
		return prettyString(str)
	}
	// this is faster
	return `"` + str + `"`
}

func (f Formatter) comma() byte {
	if f.outputFormat == outputJSON {
		return ','
	}
	return ' '
}

func (f Formatter) colon() byte {
	if f.outputFormat == outputJSON {
		return ':'
	}
	return '='
}

func (f Formatter) pretty(value any) string {
	return f.prettyWithFlags(value, 0, 0)
}

const (
	flagRawStruct = 0x1 // do not print braces on structs
)

// TODO: This is not fast. Most of the overhead goes here.
func (f Formatter) prettyWithFlags(value any, flags uint32, depth int) string {
	if depth > f.opts.MaxLogDepth {
		return `"<max-log-depth-exceeded>"`
	}

	// Handle types that take full control of logging.
	if v, ok := value.(logr.Marshaler); ok {
		// Replace the value with what the type wants to get logged.
		// That then gets handled below via reflection.
		value = invokeMarshaler(v)
	}

	// Handle types that want to format themselves.
	switch v := value.(type) {
	case fmt.Stringer:
		value = invokeStringer(v)
	case error:
		value = invokeError(v)
	}

	// Handling the most common types without reflect is a small perf win.
	switch v := value.(type) {
	case bool:
		return strconv.FormatBool(v)
	case string:
		return prettyString(v)
	case int:
		return strconv.FormatInt(int64(v), 10)
	case int8:
		return strconv.FormatInt(int64(v), 10)
	case int16:
		return strconv.FormatInt(int64(v), 10)
	case int32:
		return strconv.FormatInt(int64(v), 10)
	case int64:
		return strconv.FormatInt(int64(v), 10)
	case uint:
		return strconv.FormatUint(uint64(v), 10)
	case uint8:
		return strconv.FormatUint(uint64(v), 10)
	case uint16:
		return strconv.FormatUint(uint64(v), 10)
	case uint32:
		return strconv.FormatUint(uint64(v), 10)
	case uint64:
		return strconv.FormatUint(v, 10)
	case uintptr:
		return strconv.FormatUint(uint64(v), 10)
	case float32:
		return strconv.FormatFloat(float64(v), 'f', -1, 32)
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case complex64:
		return `"` + strconv.FormatComplex(complex128(v), 'f', -1, 64) + `"`
	case complex128:
		return `"` + strconv.FormatComplex(v, 'f', -1, 128) + `"`
	case PseudoStruct:
		buf := bytes.NewBuffer(make([]byte, 0, 1024))
		v = f.sanitize(v)
		if flags&flagRawStruct == 0 {
			buf.WriteByte('{')
		}
		for i := 0; i < len(v); i += 2 {
			if i > 0 {
				buf.WriteByte(f.comma())
			}
			k, _ := v[i].(string) // sanitize() above means no need to check success
			// arbitrary keys might need escaping
			buf.WriteString(prettyString(k))
			buf.WriteByte(f.colon())
			buf.WriteString(f.prettyWithFlags(v[i+1], 0, depth+1))
		}
		if flags&flagRawStruct == 0 {
			buf.WriteByte('}')
		}
		return buf.String()
	}

	buf := bytes.NewBuffer(make([]byte, 0, 256))
	t := reflect.TypeOf(value)
	if t == nil {
		return "null"
	}
	v := reflect.ValueOf(value)
	switch t.Kind() {
	case reflect.Bool:
		return strconv.FormatBool(v.Bool())
	case reflect.String:
		return prettyString(v.String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(int64(v.Int()), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(uint64(v.Uint()), 10)
	case reflect.Float32:
		return strconv.FormatFloat(float64(v.Float()), 'f', -1, 32)
	case reflect.Float64:
		return strconv.FormatFloat(v.Float(), 'f', -1, 64)
	case reflect.Complex64:
		return `"` + strconv.FormatComplex(complex128(v.Complex()), 'f', -1, 64) + `"`
	case reflect.Complex128:
		return `"` + strconv.FormatComplex(v.Complex(), 'f', -1, 128) + `"`
	case reflect.Struct:
		if flags&flagRawStruct == 0 {
			buf.WriteByte('{')
		}
		printComma := false // testing i>0 is not enough because of JSON omitted fields
		for i := 0; i < t.NumField(); i++ {
			fld := t.Field(i)
			if fld.PkgPath != "" {
				// reflect says this field is only defined for non-exported fields.
				continue
			}
			if !v.Field(i).CanInterface() {
				// reflect isn't clear exactly what this means, but we can't use it.
				continue
			}
			name := ""
			omitempty := false
			if tag, found := fld.Tag.Lookup("json"); found {
				if tag == "-" {
					continue
				}
				if comma := strings.Index(tag, ","); comma != -1 {
					if n := tag[:comma]; n != "" {
						name = n
					}
					rest := tag[comma:]
					if strings.Contains(rest, ",omitempty,") || strings.HasSuffix(rest, ",omitempty") {
						omitempty = true
					}
				} else {
					name = tag
				}
			}
			if omitempty && isEmpty(v.Field(i)) {
				continue
			}
			if printComma {
				buf.WriteByte(f.comma())
			}
			printComma = true // if we got here, we are rendering a field
			if fld.Anonymous && fld.Type.Kind() == reflect.Struct && name == "" {
				buf.WriteString(f.prettyWithFlags(v.Field(i).Interface(), flags|flagRawStruct, depth+1))
				continue
			}
			if name == "" {
				name = fld.Name
			}
			// field names can't contain characters which need escaping
			buf.WriteString(f.quoted(name, false))
			buf.WriteByte(f.colon())
			buf.WriteString(f.prettyWithFlags(v.Field(i).Interface(), 0, depth+1))
		}
		if flags&flagRawStruct == 0 {
			buf.WriteByte('}')
		}
		return buf.String()
	case reflect.Slice, reflect.Array:
		// If this is outputing as JSON make sure this isn't really a json.RawMessage.
		// If so just emit "as-is" and don't pretty it as that will just print
		// it as [X,Y,Z,...] which isn't terribly useful vs the string form you really want.
		if f.outputFormat == outputJSON {
			if rm, ok := value.(json.RawMessage); ok {
				// If it's empty make sure we emit an empty value as the array style would below.
				if len(rm) > 0 {
					buf.Write(rm)
				} else {
					buf.WriteString("null")
				}
				return buf.String()
			}
		}
		buf.WriteByte('[')
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				buf.WriteByte(f.comma())
			}
			e := v.Index(i)
			buf.WriteString(f.prettyWithFlags(e.Interface(), 0, depth+1))
		}
		buf.WriteByte(']')
		return buf.String()
	case reflect.Map:
		buf.WriteByte('{')
		// This does not sort the map keys, for best perf.
		it := v.MapRange()
		i := 0
		for it.Next() {
			if i > 0 {
				buf.WriteByte(f.comma())
			}
			// If a map key supports TextMarshaler, use it.
			keystr := ""
			if m, ok := it.Key().Interface().(encoding.TextMarshaler); ok {
				txt, err := m.MarshalText()
				if err != nil {
					keystr = fmt.Sprintf("<error-MarshalText: %s>", err.Error())
				} else {
					keystr = string(txt)
				}
				keystr = prettyString(keystr)
			} else {
				// prettyWithFlags will produce already-escaped values
				keystr = f.prettyWithFlags(it.Key().Interface(), 0, depth+1)
				if t.Key().Kind() != reflect.String {
					// JSON only does string keys.  Unlike Go's standard JSON, we'll
					// convert just about anything to a string.
					keystr = prettyString(keystr)
				}
			}
			buf.WriteString(keystr)
			buf.WriteByte(f.colon())
			buf.WriteString(f.prettyWithFlags(it.Value().Interface(), 0, depth+1))
			i++
		}
		buf.WriteByte('}')
		return buf.String()
	case reflect.Ptr, reflect.Interface:
		if v.IsNil() {
			return "null"
		}
		return f.prettyWithFlags(v.Elem().Interface(), 0, depth)
	}
	return fmt.Sprintf(`"<unhandled-%s>"`, t.Kind().String())
}

func prettyString(s string) string {
	// Avoid escaping (which does allocations) if we can.
	if needsEscape(s) {
		return strconv.Quote(s)
	}
	b := bytes.NewBuffer(make([]byte, 0, 1024))
	b.WriteByte('"')
	b.WriteString(s)
	b.WriteByte('"')
	return b.String()
}

// needsEscape determines whether the input string needs to be escaped or not,
// without doing any allocations.
func needsEscape(s string) bool {
	for _, r := range s {
		if !strconv.IsPrint(r) || r == '\\' || r == '"' {
			return true
		}
	}
	return false
}

func isEmpty(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Complex64, reflect.Complex128:
		return v.Complex() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func invokeMarshaler(m logr.Marshaler) (ret any) {
	defer func() {
		if r := recover(); r != nil {
			ret = fmt.Sprintf("<panic: %s>", r)
		}
	}()
	return m.MarshalLog()
}

func invokeStringer(s fmt.Stringer) (ret string) {
	defer func() {
		if r := recover(); r != nil {
			ret = fmt.Sprintf("<panic: %s>", r)
		}
	}()
	return s.String()
}

func invokeError(e error) (ret string) {
	defer func() {
		if r := recover(); r != nil {
			ret = fmt.Sprintf("<panic: %s>", r)
		}
	}()
	return e.Error()
}

// Caller represents the original call site for a log line, after considering
// logr.Logger.WithCallDepth and logr.Logger.WithCallStackHelper.  The File and
// Line fields will always be provided, while the Func field is optional.
// Users can set the render hook fields in Options to examine logged key-value
// pairs, one of which will be {"caller", Caller} if the Options.LogCaller
// field is enabled for the given MessageClass.
type Caller struct {
	// File is the basename of the file for this call site.
	File string `json:"file"`
	// Line is the line number in the file for this call site.
	Line int `json:"line"`
	// Func is the function name for this call site, or empty if
	// Options.LogCallerFunc is not enabled.
	Func string `json:"function,omitempty"`
}

func (f Formatter) caller() Caller {
	// +1 for this frame, +1 for Info/Error.
	pc, file, line, ok := runtime.Caller(f.depth + 2)
	if !ok {
		return Caller{"<unknown>", 0, ""}
	}
	fn := ""
	if f.opts.LogCallerFunc {
		if fp := runtime.FuncForPC(pc); fp != nil {
			fn = fp.Name()
		}
	}

	return Caller{filepath.Base(file), line, fn}
}

const noValue = "<no-value>"

func (f Formatter) nonStringKey(v any) string {
	return fmt.Sprintf("<non-string-key: %s>", f.snippet(v))
}

// snippet produces a short snippet string of an arbitrary value.
func (f Formatter) snippet(v any) string {
	const snipLen = 16

	snip := f.pretty(v)
	if len(snip) > snipLen {
		snip = snip[:snipLen]
	}
	return snip
}

// sanitize ensures that a list of key-value pairs has a value for every key
// (adding a value if needed) and that each key is a string (substituting a key
// if needed).
func (f Formatter) sanitize(kvList []any) []any {
	if len(kvList)%2 != 0 {
		kvList = append(kvList, noValue)
	}
	for i := 0; i < len(kvList); i += 2 {
		_, ok := kvList[i].(string)
		if !ok {
			kvList[i] = f.nonStringKey(kvList[i])
		}
	}
	return kvList
}

// startGroup opens a new group scope (basically a sub-struct), which locks all
// the current saved values and starts them anew.  This is needed to satisfy
// slog.
func (f *Formatter) startGroup(name string) {
	// Unnamed groups are just inlined.
	if name == "" {
		return
	}

	n := len(f.groups)
	f.groups = append(f.groups[:n:n], groupDef{f.groupName, f.valuesStr})

	// Start collecting new values.
	f.groupName = name
	f.valuesStr = ""
	f.values = nil
}

// Init configures this Formatter from runtime info, such as the call depth
// imposed by logr itself.
// Note that this receiver is a pointer, so depth can be saved.
func (f *Formatter) Init(info logr.RuntimeInfo) {
	f.depth += info.CallDepth
}

// Enabled checks whether an info message at the given level should be logged.
func (f Formatter) Enabled(level int) bool {
	return level <= f.opts.Verbosity
}

// GetDepth returns the current depth of this Formatter.  This is useful for
// implementations which do their own caller attribution.
func (f Formatter) GetDepth() int {
	return f.depth
}

// FormatInfo renders an Info log message into strings.  The prefix will be
// empty when no names were set (via AddNames), or when the output is
// configured for JSON.
func (f Formatter) FormatInfo(level int, msg string, kvList []any) (prefix, argsStr string) {
	args := make([]any, 0, 64) // using a constant here impacts perf
	prefix = f.prefix
	if f.outputFormat == outputJSON {
		args = append(args, "logger", prefix)
		prefix = ""
	}
	if f.opts.LogTimestamp {
		args = append(args, "ts", time.Now().Format(f.opts.TimestampFormat))
	}
	if policy := f.opts.LogCaller; policy == All || policy == Info {
		args = append(args, "caller", f.caller())
	}
	if key := *f.opts.LogInfoLevel; key != "" {
		args = append(args, key, level)
	}
	args = append(args, "msg", msg)
	return prefix, f.render(args, kvList)
}

// FormatError renders an Error log message into strings.  The prefix will be
// empty when no names were set (via AddNames), or when the output is
// configured for JSON.
func (f Formatter) FormatError(err error, msg string, kvList []any) (prefix, argsStr string) {
	args := make([]any, 0, 64) // using a constant here impacts perf
	prefix = f.prefix
	if f.outputFormat == outputJSON {
		args = append(args, "logger", prefix)
		prefix = ""
	}
	if f.opts.LogTimestamp {
		args = append(args, "ts", time.Now().Format(f.opts.TimestampFormat))
	}
	if policy := f.opts.LogCaller; policy == All || policy == Error {
		args = append(args, "caller", f.caller())
	}
	args = append(args, "msg", msg)
	var loggableErr any
	if err != nil {
		loggableErr = err.Error()
	}
	args = append(args, "error", loggableErr)
	return prefix, f.render(args, kvList)
}

// AddName appends the specified name.  funcr uses '/' characters to separate
// name elements.  Callers should not pass '/' in the provided name string, but
// this library does not actually enforce that.
func (f *Formatter) AddName(name string) {
	if len(f.prefix) > 0 {
		f.prefix += "/"
	}
	f.prefix += name
}

// AddValues adds key-value pairs to the set of saved values to be logged with
// each log line.
func (f *Formatter) AddValues(kvList []any) {
	// Three slice args forces a copy.
	n := len(f.values)
	f.values = append(f.values[:n:n], kvList...)

	vals := f.values
	if hook := f.opts.RenderValuesHook; hook != nil {
		vals = hook(f.sanitize(vals))
	}

	// Pre-render values, so we don't have to do it on each Info/Error call.
	buf := bytes.NewBuffer(make([]byte, 0, 1024))
	f.flatten(buf, vals, true) // escape user-provided keys
	f.valuesStr = buf.String()
}

// AddCallDepth increases the number of stack-frames to skip when attributing
// the log line to a file and line.
func (f *Formatter) AddCallDepth(depth int) {
	f.depth += depth
}
