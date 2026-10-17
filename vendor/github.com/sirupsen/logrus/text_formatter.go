package logrus

import (
	"bytes"
	"fmt"
	"maps"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

var baseTimestamp = time.Now()

// TextFormatter formats logs into text.
//
// Output is logfmt-like: key=value pairs separated by spaces. Fields from
// [Entry.Data] are included together with the standard fields derived from the
// entry. If a field conflicts with a standard field, it is prefixed with
// "fields.". Standard field names can be customized through FieldMap.
//
// Field keys are written as-is (unquoted and unescaped) in the plain
// (non-colored) format; only field values may be quoted depending on
// DisableQuote, ForceQuote, QuoteEmptyFields, and the value content.
//
// When colors are enabled, ANSI escape sequences may be added for presentation.
// For fully escaped structured output (including safe keys), use JSONFormatter.
type TextFormatter struct {
	// Set to true to bypass checking for a TTY before outputting colors.
	ForceColors bool

	// Force disabling colors.
	DisableColors bool

	// Force quoting of all values
	ForceQuote bool

	// DisableQuote disables quoting for all values.
	// DisableQuote will have a lower priority than ForceQuote.
	// If both of them are set to true, quote will be forced on all values.
	DisableQuote bool

	// Override coloring based on CLICOLOR and CLICOLOR_FORCE. - https://bixense.com/clicolors/
	EnvironmentOverrideColors bool

	// Disable timestamp logging. useful when output is redirected to logging
	// system that already adds timestamps.
	DisableTimestamp bool

	// Enable logging the full timestamp when a TTY is attached instead of just
	// the time passed since beginning of execution.
	FullTimestamp bool

	// TimestampFormat to use for display when a full timestamp is printed.
	// The format to use is the same than for time.Format or time.Parse from the standard
	// library.
	// The standard Library already provides a set of predefined format.
	TimestampFormat string

	// The fields are sorted by default for a consistent output. For applications
	// that log extremely frequently and don't use the JSON formatter this may not
	// be desired.
	DisableSorting bool

	// The keys sorting function, when uninitialized it uses slices.Sort.
	SortingFunc func([]string)

	// Disables the truncation of the level text to 4 characters.
	DisableLevelTruncation bool

	// PadLevelText Adds padding the level text so that all the levels output at the same length
	// PadLevelText is a superset of the DisableLevelTruncation option
	PadLevelText bool

	// QuoteEmptyFields will wrap empty fields in quotes if true
	QuoteEmptyFields bool

	// Whether the logger's out is to a terminal. Don't use this field
	// directly; use TextFormatter.isTerminal instead.
	terminal bool

	// FieldMap allows users to customize the names of keys for default fields.
	// Mapped keys are written as-is, so they should be safe for plain-text output.
	//
	// As an example:
	//
	// formatter := &TextFormatter{
	// 	FieldMap: FieldMap{
	// 		FieldKeyTime:  "@timestamp",
	// 		FieldKeyLevel: "@level",
	// 		FieldKeyMsg:   "@message",
	// 	},
	// }
	FieldMap FieldMap

	// CallerPrettyfier can be set by the user to modify the content
	// of the function and file keys in the data when ReportCaller is
	// activated. If any of the returned value is the empty string the
	// corresponding key will be removed from fields.
	CallerPrettyfier func(*runtime.Frame) (function string, file string)

	terminalInitOnce sync.Once
}

func (f *TextFormatter) isTerminal(entry *Entry) bool {
	if entry == nil || entry.Logger == nil {
		// Don't run the terminalInitOnce without a logger, otherwise we'd
		// cache the default (false) forever even if a logger is attached
		// later.
		return false
	}

	f.terminalInitOnce.Do(func() {
		entry.Logger.mu.Lock()
		out := entry.Logger.Out
		entry.Logger.mu.Unlock()

		f.terminal = checkIfTerminal(out)
	})

	return f.terminal
}

func (f *TextFormatter) isColored(isTerminal bool) bool {
	if f.DisableColors {
		return false
	}

	colored := f.ForceColors || isTerminal
	if !f.EnvironmentOverrideColors {
		return colored
	}
	if force, ok := os.LookupEnv("CLICOLOR_FORCE"); ok {
		return force != "0"
	}
	if os.Getenv("CLICOLOR") == "0" {
		return false
	}
	return colored
}

// Format renders a single log entry
func (f *TextFormatter) Format(entry *Entry) ([]byte, error) {
	data := make(Fields, len(entry.Data))
	maps.Copy(data, entry.Data)
	isColored := f.isColored(f.isTerminal(entry))

	caller := entry.Caller
	hasCaller := caller != nil
	prefixFieldClashes(data, f.FieldMap, hasCaller)
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}

	b := entry.Buffer
	if b == nil {
		b = new(bytes.Buffer)
	}

	if isColored {
		f.printColored(b, entry, keys, data)
	} else {
		f.printPlain(b, entry, keys, data)
	}

	return b.Bytes(), nil
}

func (f *TextFormatter) printPlain(b *bytes.Buffer, entry *Entry, keys []string, data Fields) {
	caller := entry.Caller
	hasCaller := caller != nil

	fixedKeys := make([]string, 0, len(keys)+defaultFields)
	if !f.DisableTimestamp {
		fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyTime))
	}
	fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyLevel))
	if entry.Message != "" {
		fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyMsg))
	}
	if entry.err != "" {
		fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyLogrusError))
	}

	var funcVal, fileVal string
	if caller != nil {
		if f.CallerPrettyfier != nil {
			funcVal, fileVal = f.CallerPrettyfier(caller)
		} else {
			funcVal = caller.Function
			fileVal = caller.File + ":" + strconv.FormatInt(int64(caller.Line), 10)
		}

		if funcVal != "" {
			fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyFunc))
		}
		if fileVal != "" {
			fixedKeys = append(fixedKeys, f.FieldMap.resolve(FieldKeyFile))
		}
	}

	if !f.DisableSorting {
		if f.SortingFunc == nil {
			// Default sorting does not sort the "fixed keys";
			// see https://github.com/sirupsen/logrus/commit/73bc94e60c753099e8bae902f81fbd6e7dd95f26
			slices.Sort(keys)
			fixedKeys = append(fixedKeys, keys...)
		} else {
			fixedKeys = append(fixedKeys, keys...)
			f.SortingFunc(fixedKeys)
		}
	} else {
		fixedKeys = append(fixedKeys, keys...)
	}

	for _, key := range fixedKeys {
		var value any
		switch {
		case key == f.FieldMap.resolve(FieldKeyTime):
			if f.TimestampFormat == "" {
				value = entry.Time.Format(defaultTimestampFormat)
			} else {
				value = entry.Time.Format(f.TimestampFormat)
			}
		case key == f.FieldMap.resolve(FieldKeyLevel):
			value = entry.Level.String()
		case key == f.FieldMap.resolve(FieldKeyMsg):
			value = entry.Message
		case key == f.FieldMap.resolve(FieldKeyLogrusError):
			value = entry.err
		case key == f.FieldMap.resolve(FieldKeyFunc) && hasCaller:
			value = funcVal
		case key == f.FieldMap.resolve(FieldKeyFile) && hasCaller:
			value = fileVal
		default:
			value = data[key]
		}
		f.appendKeyValue(b, key, value)
	}

	b.WriteByte('\n')
}

func (f *TextFormatter) printColored(b *bytes.Buffer, entry *Entry, keys []string, data Fields) {
	// Remove a single newline if it already exists in the message to keep
	// the behavior of logrus text_formatter the same as the stdlib log package
	entry.Message = strings.TrimSuffix(entry.Message, "\n")

	var callerText string
	if caller := entry.Caller; caller != nil {
		var funcVal, fileVal string
		if f.CallerPrettyfier != nil {
			funcVal, fileVal = f.CallerPrettyfier(caller)
		} else {
			if caller.Function != "" {
				funcVal = caller.Function + "()"
			}
			fileVal = caller.File + ":" + strconv.FormatInt(int64(caller.Line), 10)
		}

		if fileVal == "" {
			callerText = funcVal
		} else if funcVal == "" {
			callerText = fileVal
		} else {
			callerText = fileVal + " " + funcVal
		}
	}

	levelText := levelPrefix(entry.Level, f.DisableLevelTruncation, f.PadLevelText)
	switch {
	case f.DisableTimestamp:
		_, _ = fmt.Fprintf(b, "%s%s %-44s ", levelText, callerText, entry.Message)
	case !f.FullTimestamp:
		_, _ = fmt.Fprintf(b, "%s[%04d]%s %-44s ", levelText, int(entry.Time.Sub(baseTimestamp)/time.Second), callerText, entry.Message)
	default:
		timestampFormat := f.TimestampFormat
		if timestampFormat == "" {
			timestampFormat = defaultTimestampFormat
		}
		_, _ = fmt.Fprintf(b, "%s[%s]%s %-44s ", levelText, entry.Time.Format(timestampFormat), callerText, entry.Message)
	}

	if !f.DisableSorting {
		if f.SortingFunc == nil {
			slices.Sort(keys)
		} else {
			f.SortingFunc(keys)
		}
	}

	// Keys use the same color as the level-prefix.
	for _, k := range keys {
		b.WriteByte(' ')
		b.WriteString(colorize(entry.Level, k))
		b.WriteByte('=')
		f.appendValue(b, data[k])
	}

	b.WriteByte('\n')
}

// appendKeyValue writes key=value. Keys are written verbatim (unquoted/unescaped);
// values are subject to quoting/escaping.
func (f *TextFormatter) appendKeyValue(b *bytes.Buffer, key string, value any) {
	if b.Len() > 0 {
		b.WriteByte(' ')
	}
	b.WriteString(key)
	b.WriteByte('=')
	f.appendValue(b, value)
}

func (f *TextFormatter) appendValue(b *bytes.Buffer, value any) {
	// Fast paths.
	switch v := value.(type) {
	case string:
		f.appendString(b, v)
		return
	case []byte:
		f.appendBytes(b, v)
		return
	case bool:
		var raw [8]byte
		f.appendBytes(b, strconv.AppendBool(raw[:0], v))
		return
	case error:
		f.appendError(b, v)
		return
	case fmt.Stringer:
		f.appendStringer(b, v)
		return
	}

	// Handle common primitives.
	var raw [64]byte
	var num []byte

	switch v := value.(type) {
	case int:
		num = strconv.AppendInt(raw[:0], int64(v), 10)
	case int8:
		num = strconv.AppendInt(raw[:0], int64(v), 10)
	case int16:
		num = strconv.AppendInt(raw[:0], int64(v), 10)
	case int32:
		num = strconv.AppendInt(raw[:0], int64(v), 10)
	case int64:
		num = strconv.AppendInt(raw[:0], v, 10)

	case uint:
		num = strconv.AppendUint(raw[:0], uint64(v), 10)
	case uint8:
		num = strconv.AppendUint(raw[:0], uint64(v), 10)
	case uint16:
		num = strconv.AppendUint(raw[:0], uint64(v), 10)
	case uint32:
		num = strconv.AppendUint(raw[:0], uint64(v), 10)
	case uint64:
		num = strconv.AppendUint(raw[:0], v, 10)
	case uintptr:
		num = strconv.AppendUint(raw[:0], uint64(v), 10)

	case float32:
		num = strconv.AppendFloat(raw[:0], float64(v), 'g', -1, 32)
	case float64:
		num = strconv.AppendFloat(raw[:0], v, 'g', -1, 64)

	default:
		f.appendString(b, fmt.Sprint(value))
		return
	}

	f.appendNumeric(b, num)
}

func (f *TextFormatter) appendString(b *bytes.Buffer, s string) {
	quote := f.ForceQuote || (f.QuoteEmptyFields && len(s) == 0) || (!f.DisableQuote && needsQuoting(s))
	if !quote {
		b.WriteString(s)
		return
	}
	if len(s) == 0 {
		b.WriteString(`""`)
		return
	}

	var tmp [128]byte
	b.Write(strconv.AppendQuote(tmp[:0], s))
}

func (f *TextFormatter) appendBytes(b *bytes.Buffer, bs []byte) {
	quote := f.ForceQuote || (f.QuoteEmptyFields && len(bs) == 0) || (!f.DisableQuote && needsQuotingBytes(bs))
	if !quote {
		b.Write(bs)
		return
	}
	if len(bs) == 0 {
		b.WriteString(`""`)
		return
	}

	var tmp [128]byte
	b.Write(strconv.AppendQuote(tmp[:0], string(bs)))
}

func (f *TextFormatter) appendNumeric(b *bytes.Buffer, out []byte) {
	if f.ForceQuote {
		var tmp [128]byte
		b.Write(strconv.AppendQuote(tmp[:0], string(out)))
		return
	}
	b.Write(out)
}

func (f *TextFormatter) appendError(b *bytes.Buffer, v error) {
	defer f.recoverValue(b, v, "Error")

	f.appendString(b, v.Error())
}

func (f *TextFormatter) appendStringer(b *bytes.Buffer, v fmt.Stringer) {
	defer f.recoverValue(b, v, "String")

	f.appendString(b, v.String())
}

func (f *TextFormatter) recoverValue(b *bytes.Buffer, v any, method string) {
	if r := recover(); r != nil {
		rv := reflect.ValueOf(v)
		if rv.Kind() == reflect.Pointer && rv.IsNil() {
			f.appendString(b, "<nil>")
		} else {
			f.appendString(b, fmt.Sprintf("%%!v(PANIC=%s method: %v)", method, r))
		}
	}
}

// needsQuoting returns true if the string contains any byte that
// requires quoting. It returns false when every byte is "safe" according
// to isSafeByte.
func needsQuoting(s string) bool {
	// use an index loop (avoid rune decoding).
	for i := range len(s) {
		c := s[i]
		if !isSafeByte(c) {
			return true
		}
	}
	return false
}

// needsQuotingBytes returns true if the byte slice contains any byte that
// requires quoting. It returns false when every byte is "safe" according
// to isSafeByte.
func needsQuotingBytes(bs []byte) bool {
	for _, c := range bs {
		if !isSafeByte(c) {
			return true
		}
	}
	return false
}

// isSafeByte returns true if the byte is allowed unquoted (ASCII and in the allowlist).
// It purposely uses byte arithmetic (no runes) for performance.
func isSafeByte(ch byte) bool {
	ok := ch < 0x80 && ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9'))
	if ok {
		return true
	}
	switch ch {
	case '-', '.', '_', '/', '@', '^', '+':
		return true
	default:
		return false
	}
}
