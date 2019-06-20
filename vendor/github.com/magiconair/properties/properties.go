// Copyright 2018 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

// BUG(frank): Set() does not check for invalid unicode literals since this is currently handled by the lexer.
// BUG(frank): Write() does not allow to configure the newline character. Therefore, on Windows LF is used.

import (
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const maxExpansionDepth = 64

// ErrorHandlerFunc defines the type of function which handles failures
// of the MustXXX() functions. An error handler function must exit
// the application after handling the error.
type ErrorHandlerFunc func(error)

// ErrorHandler is the function which handles failures of the MustXXX()
// functions. The default is LogFatalHandler.
var ErrorHandler ErrorHandlerFunc = LogFatalHandler

// LogHandlerFunc defines the function prototype for logging errors.
type LogHandlerFunc func(fmt string, args ...interface{})

// LogPrintf defines a log handler which uses log.Printf.
var LogPrintf LogHandlerFunc = log.Printf

// LogFatalHandler handles the error by logging a fatal error and exiting.
func LogFatalHandler(err error) {
	log.Fatal(err)
}

// PanicHandler handles the error by panicking.
func PanicHandler(err error) {
	panic(err)
}

// -----------------------------------------------------------------------------

// A Properties contains the key/value pairs from the properties input.
// All values are stored in unexpanded form and are expanded at runtime
type Properties struct {
	// Pre-/Postfix for property expansion.
	Prefix  string
	Postfix string

	// DisableExpansion controls the expansion of properties on Get()
	// and the check for circular references on Set(). When set to
	// true Properties behaves like a simple key/value store and does
	// not check for circular references on Get() or on Set().
	DisableExpansion bool

	// Stores the key/value pairs
	m map[string]string

	// Stores the comments per key.
	c map[string][]string

	// Stores the keys in order of appearance.
	k []string
}

// NewProperties creates a new Properties struct with the default
// configuration for "${key}" expressions.
func NewProperties() *Properties {
	return &Properties{
		Prefix:  "${",
		Postfix: "}",
		m:       map[string]string{},
		c:       map[string][]string{},
		k:       []string{},
	}
}

// Load reads a buffer into the given Properties struct.
func (p *Properties) Load(buf []byte, enc Encoding) error {
	l := &Loader{Encoding: enc, DisableExpansion: p.DisableExpansion}
	newProperties, err := l.LoadBytes(buf)
	if err != nil {
		return err
	}
	p.Merge(newProperties)
	return nil
}

// Get returns the expanded value for the given key if exists.
// Otherwise, ok is false.
func (p *Properties) Get(key string) (value string, ok bool) {
	v, ok := p.m[key]
	if p.DisableExpansion {
		return v, ok
	}
	if !ok {
		return "", false
	}

	expanded, err := p.expand(key, v)

	// we guarantee that the expanded value is free of
	// circular references and malformed expressions
	// so we panic if we still get an error here.
	if err != nil {
		ErrorHandler(fmt.Errorf("%s in %q", err, key+" = "+v))
	}

	return expanded, true
}

// MustGet returns the expanded value for the given key if exists.
// Otherwise, it panics.
func (p *Properties) MustGet(key string) string {
	if v, ok := p.Get(key); ok {
		return v
	}
	ErrorHandler(invalidKeyError(key))
	panic("ErrorHandler should exit")
}

// ----------------------------------------------------------------------------

// ClearComments removes the comments for all keys.
func (p *Properties) ClearComments() {
	p.c = map[string][]string{}
}

// ----------------------------------------------------------------------------

// GetComment returns the last comment before the given key or an empty string.
func (p *Properties) GetComment(key string) string {
	comments, ok := p.c[key]
	if !ok || len(comments) == 0 {
		return ""
	}
	return comments[len(comments)-1]
}

// ----------------------------------------------------------------------------

// GetComments returns all comments that appeared before the given key or nil.
func (p *Properties) GetComments(key string) []string {
	if comments, ok := p.c[key]; ok {
		return comments
	}
	return nil
}

// ----------------------------------------------------------------------------

// SetComment sets the comment for the key.
func (p *Properties) SetComment(key, comment string) {
	p.c[key] = []string{comment}
}

// ----------------------------------------------------------------------------

// SetComments sets the comments for the key. If the comments are nil then
// all comments for this key are deleted.
func (p *Properties) SetComments(key string, comments []string) {
	if comments == nil {
		delete(p.c, key)
		return
	}
	p.c[key] = comments
}

// ----------------------------------------------------------------------------

// GetBool checks if the expanded value is one of '1', 'yes',
// 'true' or 'on' if the key exists. The comparison is case-insensitive.
// If the key does not exist the default value is returned.
func (p *Properties) GetBool(key string, def bool) bool {
	v, err := p.getBool(key)
	if err != nil {
		return def
	}
	return v
}

// MustGetBool checks if the expanded value is one of '1', 'yes',
// 'true' or 'on' if the key exists. The comparison is case-insensitive.
// If the key does not exist the function panics.
func (p *Properties) MustGetBool(key string) bool {
	v, err := p.getBool(key)
	if err != nil {
		ErrorHandler(err)
	}
	return v
}

func (p *Properties) getBool(key string) (value bool, err error) {
	if v, ok := p.Get(key); ok {
		return boolVal(v), nil
	}
	return false, invalidKeyError(key)
}

func boolVal(v string) bool {
	v = strings.ToLower(v)
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// ----------------------------------------------------------------------------

// GetDuration parses the expanded value as an time.Duration (in ns) if the
// key exists. If key does not exist or the value cannot be parsed the default
// value is returned. In almost all cases you want to use GetParsedDuration().
func (p *Properties) GetDuration(key string, def time.Duration) time.Duration {
	v, err := p.getInt64(key)
	if err != nil {
		return def
	}
	return time.Duration(v)
}

// MustGetDuration parses the expanded value as an time.Duration (in ns) if
// the key exists. If key does not exist or the value cannot be parsed the
// function panics. In almost all cases you want to use MustGetParsedDuration().
func (p *Properties) MustGetDuration(key string) time.Duration {
	v, err := p.getInt64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return time.Duration(v)
}

// ----------------------------------------------------------------------------

// GetParsedDuration parses the expanded value with time.ParseDuration() if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned.
func (p *Properties) GetParsedDuration(key string, def time.Duration) time.Duration {
	s, ok := p.Get(key)
	if !ok {
		return def
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return def
	}
	return v
}

// MustGetParsedDuration parses the expanded value with time.ParseDuration() if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
func (p *Properties) MustGetParsedDuration(key string) time.Duration {
	s, ok := p.Get(key)
	if !ok {
		ErrorHandler(invalidKeyError(key))
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		ErrorHandler(err)
	}
	return v
}

// ----------------------------------------------------------------------------

// GetFloat64 parses the expanded value as a float64 if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned.
func (p *Properties) GetFloat64(key string, def float64) float64 {
	v, err := p.getFloat64(key)
	if err != nil {
		return def
	}
	return v
}

// MustGetFloat64 parses the expanded value as a float64 if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
func (p *Properties) MustGetFloat64(key string) float64 {
	v, err := p.getFloat64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return v
}

func (p *Properties) getFloat64(key string) (value float64, err error) {
	if v, ok := p.Get(key); ok {
		value, err = strconv.ParseFloat(v, 64)
		if err != nil {
			return 0, err
		}
		return value, nil
	}
	return 0, invalidKeyError(key)
}

// ----------------------------------------------------------------------------

// GetInt parses the expanded value as an int if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned. If the value does not fit into an int the
// function panics with an out of range error.
func (p *Properties) GetInt(key string, def int) int {
	v, err := p.getInt64(key)
	if err != nil {
		return def
	}
	return intRangeCheck(key, v)
}

// MustGetInt parses the expanded value as an int if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
// If the value does not fit into an int the function panics with
// an out of range error.
func (p *Properties) MustGetInt(key string) int {
	v, err := p.getInt64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return intRangeCheck(key, v)
}

// ----------------------------------------------------------------------------

// GetInt64 parses the expanded value as an int64 if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned.
func (p *Properties) GetInt64(key string, def int64) int64 {
	v, err := p.getInt64(key)
	if err != nil {
		return def
	}
	return v
}

// MustGetInt64 parses the expanded value as an int if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
func (p *Properties) MustGetInt64(key string) int64 {
	v, err := p.getInt64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return v
}

func (p *Properties) getInt64(key string) (value int64, err error) {
	if v, ok := p.Get(key); ok {
		value, err = strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0, err
		}
		return value, nil
	}
	return 0, invalidKeyError(key)
}

// ----------------------------------------------------------------------------

// GetUint parses the expanded value as an uint if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned. If the value does not fit into an int the
// function panics with an out of range error.
func (p *Properties) GetUint(key string, def uint) uint {
	v, err := p.getUint64(key)
	if err != nil {
		return def
	}
	return uintRangeCheck(key, v)
}

// MustGetUint parses the expanded value as an int if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
// If the value does not fit into an int the function panics with
// an out of range error.
func (p *Properties) MustGetUint(key string) uint {
	v, err := p.getUint64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return uintRangeCheck(key, v)
}

// ----------------------------------------------------------------------------

// GetUint64 parses the expanded value as an uint64 if the key exists.
// If key does not exist or the value cannot be parsed the default
// value is returned.
func (p *Properties) GetUint64(key string, def uint64) uint64 {
	v, err := p.getUint64(key)
	if err != nil {
		return def
	}
	return v
}

// MustGetUint64 parses the expanded value as an int if the key exists.
// If key does not exist or the value cannot be parsed the function panics.
func (p *Properties) MustGetUint64(key string) uint64 {
	v, err := p.getUint64(key)
	if err != nil {
		ErrorHandler(err)
	}
	return v
}

func (p *Properties) getUint64(key string) (value uint64, err error) {
	if v, ok := p.Get(key); ok {
		value, err = strconv.ParseUint(v, 10, 64)
		if err != nil {
			return 0, err
		}
		return value, nil
	}
	return 0, invalidKeyError(key)
}

// ----------------------------------------------------------------------------

// GetString returns the expanded value for the given key if exists or
// the default value otherwise.
func (p *Properties) GetString(key, def string) string {
	if v, ok := p.Get(key); ok {
		return v
	}
	return def
}

// MustGetString returns the expanded value for the given key if exists or
// panics otherwise.
func (p *Properties) MustGetString(key string) string {
	if v, ok := p.Get(key); ok {
		return v
	}
	ErrorHandler(invalidKeyError(key))
	panic("ErrorHandler should exit")
}

// ----------------------------------------------------------------------------

// Filter returns a new properties object which contains all properties
// for which the key matches the pattern.
func (p *Properties) Filter(pattern string) (*Properties, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}

	return p.FilterRegexp(re), nil
}

// FilterRegexp returns a new properties object which contains all properties
// for which the key matches the regular expression.
func (p *Properties) FilterRegexp(re *regexp.Regexp) *Properties {
	pp := NewProperties()
	for _, k := range p.k {
		if re.MatchString(k) {
			// TODO(fs): we are ignoring the error which flags a circular reference.
			// TODO(fs): since we are just copying a subset of keys this cannot happen (fingers crossed)
			pp.Set(k, p.m[k])
		}
	}
	return pp
}

// FilterPrefix returns a new properties object with a subset of all keys
// with the given prefix.
func (p *Properties) FilterPrefix(prefix string) *Properties {
	pp := NewProperties()
	for _, k := range p.k {
		if strings.HasPrefix(k, prefix) {
			// TODO(fs): we are ignoring the error which flags a circular reference.
			// TODO(fs): since we are just copying a subset of keys this cannot happen (fingers crossed)
			pp.Set(k, p.m[k])
		}
	}
	return pp
}

// FilterStripPrefix returns a new properties object with a subset of all keys
// with the given prefix and the prefix removed from the keys.
func (p *Properties) FilterStripPrefix(prefix string) *Properties {
	pp := NewProperties()
	n := len(prefix)
	for _, k := range p.k {
		if len(k) > len(prefix) && strings.HasPrefix(k, prefix) {
			// TODO(fs): we are ignoring the error which flags a circular reference.
			// TODO(fs): since we are modifying keys I am not entirely sure whether we can create a circular reference
			// TODO(fs): this function should probably return an error but the signature is fixed
			pp.Set(k[n:], p.m[k])
		}
	}
	return pp
}

// Len returns the number of keys.
func (p *Properties) Len() int {
	return len(p.m)
}

// Keys returns all keys in the same order as in the input.
func (p *Properties) Keys() []string {
	keys := make([]string, len(p.k))
	copy(keys, p.k)
	return keys
}

// Set sets the property key to the corresponding value.
// If a value for key existed before then ok is true and prev
// contains the previous value. If the value contains a
// circular reference or a malformed expression then
// an error is returned.
// An empty key is silently ignored.
func (p *Properties) Set(key, value string) (prev string, ok bool, err error) {
	if key == "" {
		return "", false, nil
	}

	// if expansion is disabled we allow circular references
	if p.DisableExpansion {
		prev, ok = p.Get(key)
		p.m[key] = value
		if !ok {
			p.k = append(p.k, key)
		}
		return prev, ok, nil
	}

	// to check for a circular reference we temporarily need
	// to set the new value. If there is an error then revert
	// to the previous state. Only if all tests are successful
	// then we add the key to the p.k list.
	prev, ok = p.Get(key)
	p.m[key] = value

	// now check for a circular reference
	_, err = p.expand(key, value)
	if err != nil {

		// revert to the previous state
		if ok {
			p.m[key] = prev
		} else {
			delete(p.m, key)
		}

		return "", false, err
	}

	if !ok {
		p.k = append(p.k, key)
	}

	return prev, ok, nil
}

// SetValue sets property key to the default string value
// as defined by fmt.Sprintf("%v").
func (p *Properties) SetValue(key string, value interface{}) error {
	_, _, err := p.Set(key, fmt.Sprintf("%v", value))
	return err
}

// MustSet sets the property key to the corresponding value.
// If a value for key existed before then ok is true and prev
// contains the previous value. An empty key is silently ignored.
func (p *Properties) MustSet(key, value string) (prev string, ok bool) {
	prev, ok, err := p.Set(key, value)
	if err != nil {
		ErrorHandler(err)
	}
	return prev, ok
}

// String returns a string of all expanded 'key = value' pairs.
func (p *Properties) String() string {
	var s string
	for _, key := range p.k {
		value, _ := p.Get(key)
		s = fmt.Sprintf("%s%s = %s\n", s, key, value)
	}
	return s
}

// Write writes all unexpanded 'key = value' pairs to the given writer.
// Write returns the number of bytes written and any write error encountered.
func (p *Properties) Write(w io.Writer, enc Encoding) (n int, err error) {
	return p.WriteComment(w, "", enc)
}

// WriteComment writes all unexpanced 'key = value' pairs to the given writer.
// If prefix is not empty then comments are written with a blank line and the
// given prefix. The prefix should be either "# " or "! " to be compatible with
// the properties file format. Otherwise, the properties parser will not be
// able to read the file back in. It returns the number of bytes written and
// any write error encountered.
func (p *Properties) WriteComment(w io.Writer, prefix string, enc Encoding) (n int, err error) {
	var x int

	for _, key := range p.k {
		value := p.m[key]

		if prefix != "" {
			if comments, ok := p.c[key]; ok {
				// don't print comments if they are all empty
				allEmpty := true
				for _, c := range comments {
					if c != "" {
						allEmpty = false
						break
					}
				}

				if !allEmpty {
					// add a blank line between entries but not at the top
					if len(comments) > 0 && n > 0 {
						x, err = fmt.Fprintln(w)
						if err != nil {
							return
						}
						n += x
					}

					for _, c := range comments {
						x, err = fmt.Fprintf(w, "%s%s\n", prefix, encode(c, "", enc))
						if err != nil {
							return
						}
						n += x
					}
				}
			}
		}

		x, err = fmt.Fprintf(w, "%s = %s\n", encode(key, " :", enc), encode(value, "", enc))
		if err != nil {
			return
		}
		n += x
	}
	return
}

// Map returns a copy of the properties as a map.
func (p *Properties) Map() map[string]string {
	m := make(map[string]string)
	for k, v := range p.m {
		m[k] = v
	}
	return m
}

// FilterFunc returns a copy of the properties which includes the values which passed all filters.
func (p *Properties) FilterFunc(filters ...func(k, v string) bool) *Properties {
	pp := NewProperties()
outer:
	for k, v := range p.m {
		for _, f := range filters {
			if !f(k, v) {
				continue outer
			}
			pp.Set(k, v)
		}
	}
	return pp
}

// ----------------------------------------------------------------------------

// Delete removes the key and its comments.
func (p *Properties) Delete(key string) {
	delete(p.m, key)
	delete(p.c, key)
	newKeys := []string{}
	for _, k := range p.k {
		if k != key {
			newKeys = append(newKeys, k)
		}
	}
	p.k = newKeys
}

// Merge merges properties, comments and keys from other *Properties into p
func (p *Properties) Merge(other *Properties) {
	for k, v := range other.m {
		p.m[k] = v
	}
	for k, v := range other.c {
		p.c[k] = v
	}

outer:
	for _, otherKey := range other.k {
		for _, key := range p.k {
			if otherKey == key {
				continue outer
			}
		}
		p.k = append(p.k, otherKey)
	}
}

// ----------------------------------------------------------------------------

// check expands all values and returns an error if a circular reference or
// a malformed expression was found.
func (p *Properties) check() error {
	for key, value := range p.m {
		if _, err := p.expand(key, value); err != nil {
			return err
		}
	}
	return nil
}

func (p *Properties) expand(key, input string) (string, error) {
	// no pre/postfix -> nothing to expand
	if p.Prefix == "" && p.Postfix == "" {
		return input, nil
	}

	return expand(input, []string{key}, p.Prefix, p.Postfix, p.m)
}

// expand recursively expands expressions of '(prefix)key(postfix)' to their corresponding values.
// The function keeps track of the keys that were already expanded and stops if it
// detects a circular reference or a malformed expression of the form '(prefix)key'.
func expand(s string, keys []string, prefix, postfix string, values map[string]string) (string, error) {
	if len(keys) > maxExpansionDepth {
		return "", fmt.Errorf("expansion too deep")
	}

	for {
		start := strings.Index(s, prefix)
		if start == -1 {
			return s, nil
		}

		keyStart := start + len(prefix)
		keyLen := strings.Index(s[keyStart:], postfix)
		if keyLen == -1 {
			return "", fmt.Errorf("malformed expression")
		}

		end := keyStart + keyLen + len(postfix) - 1
		key := s[keyStart : keyStart+keyLen]

		// fmt.Printf("s:%q pp:%q start:%d end:%d keyStart:%d keyLen:%d key:%q\n", s, prefix + "..." + postfix, start, end, keyStart, keyLen, key)

		for _, k := range keys {
			if key == k {
				return "", fmt.Errorf("circular reference")
			}
		}

		val, ok := values[key]
		if !ok {
			val = os.Getenv(key)
		}
		new_val, err := expand(val, append(keys, key), prefix, postfix, values)
		if err != nil {
			return "", err
		}
		s = s[:start] + new_val + s[end+1:]
	}
	return s, nil
}

// encode encodes a UTF-8 string to ISO-8859-1 and escapes some characters.
func encode(s string, special string, enc Encoding) string {
	switch enc {
	case UTF8:
		return encodeUtf8(s, special)
	case ISO_8859_1:
		return encodeIso(s, special)
	default:
		panic(fmt.Sprintf("unsupported encoding %v", enc))
	}
}

func encodeUtf8(s string, special string) string {
	v := ""
	for pos := 0; pos < len(s); {
		r, w := utf8.DecodeRuneInString(s[pos:])
		pos += w
		v += escape(r, special)
	}
	return v
}

func encodeIso(s string, special string) string {
	var r rune
	var w int
	var v string
	for pos := 0; pos < len(s); {
		switch r, w = utf8.DecodeRuneInString(s[pos:]); {
		case r < 1<<8: // single byte rune -> escape special chars only
			v += escape(r, special)
		case r < 1<<16: // two byte rune -> unicode literal
			v += fmt.Sprintf("\\u%04x", r)
		default: // more than two bytes per rune -> can't encode
			v += "?"
		}
		pos += w
	}
	return v
}

func escape(r rune, special string) string {
	switch r {
	case '\f':
		return "\\f"
	case '\n':
		return "\\n"
	case '\r':
		return "\\r"
	case '\t':
		return "\\t"
	default:
		if strings.ContainsRune(special, r) {
			return "\\" + string(r)
		}
		return string(r)
	}
}

func invalidKeyError(key string) error {
	return fmt.Errorf("unknown property: %s", key)
}
