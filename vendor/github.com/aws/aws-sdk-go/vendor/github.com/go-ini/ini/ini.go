// Copyright 2014 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

// Package ini provides INI file read and write functionality in Go.
package ini

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	DEFAULT_SECTION = "DEFAULT"
	// Maximum allowed depth when recursively substituing variable names.
	_DEPTH_VALUES = 99

	_VERSION = "1.6.0"
)

func Version() string {
	return _VERSION
}

var (
	LineBreak = "\n"

	// Variable regexp pattern: %(variable)s
	varPattern = regexp.MustCompile(`%\(([^\)]+)\)s`)

	// Write spaces around "=" to look better.
	PrettyFormat = true
)

func init() {
	if runtime.GOOS == "windows" {
		LineBreak = "\r\n"
	}
}

func inSlice(str string, s []string) bool {
	for _, v := range s {
		if str == v {
			return true
		}
	}
	return false
}

// dataSource is a interface that returns file content.
type dataSource interface {
	ReadCloser() (io.ReadCloser, error)
}

type sourceFile struct {
	name string
}

func (s sourceFile) ReadCloser() (_ io.ReadCloser, err error) {
	return os.Open(s.name)
}

type bytesReadCloser struct {
	reader io.Reader
}

func (rc *bytesReadCloser) Read(p []byte) (n int, err error) {
	return rc.reader.Read(p)
}

func (rc *bytesReadCloser) Close() error {
	return nil
}

type sourceData struct {
	data []byte
}

func (s *sourceData) ReadCloser() (io.ReadCloser, error) {
	return &bytesReadCloser{bytes.NewReader(s.data)}, nil
}

//  ____  __.
// |    |/ _|____ ___.__.
// |      <_/ __ <   |  |
// |    |  \  ___/\___  |
// |____|__ \___  > ____|
//         \/   \/\/

// Key represents a key under a section.
type Key struct {
	s          *Section
	Comment    string
	name       string
	value      string
	isAutoIncr bool
}

// Name returns name of key.
func (k *Key) Name() string {
	return k.name
}

// Value returns raw value of key for performance purpose.
func (k *Key) Value() string {
	return k.value
}

// String returns string representation of value.
func (k *Key) String() string {
	val := k.value
	if strings.Index(val, "%") == -1 {
		return val
	}

	for i := 0; i < _DEPTH_VALUES; i++ {
		vr := varPattern.FindString(val)
		if len(vr) == 0 {
			break
		}

		// Take off leading '%(' and trailing ')s'.
		noption := strings.TrimLeft(vr, "%(")
		noption = strings.TrimRight(noption, ")s")

		// Search in the same section.
		nk, err := k.s.GetKey(noption)
		if err != nil {
			// Search again in default section.
			nk, _ = k.s.f.Section("").GetKey(noption)
		}

		// Substitute by new value and take off leading '%(' and trailing ')s'.
		val = strings.Replace(val, vr, nk.value, -1)
	}
	return val
}

// Validate accepts a validate function which can
// return modifed result as key value.
func (k *Key) Validate(fn func(string) string) string {
	return fn(k.String())
}

// parseBool returns the boolean value represented by the string.
//
// It accepts 1, t, T, TRUE, true, True, YES, yes, Yes, ON, on, On,
// 0, f, F, FALSE, false, False, NO, no, No, OFF, off, Off.
// Any other value returns an error.
func parseBool(str string) (value bool, err error) {
	switch str {
	case "1", "t", "T", "true", "TRUE", "True", "YES", "yes", "Yes", "ON", "on", "On":
		return true, nil
	case "0", "f", "F", "false", "FALSE", "False", "NO", "no", "No", "OFF", "off", "Off":
		return false, nil
	}
	return false, fmt.Errorf("parsing \"%s\": invalid syntax", str)
}

// Bool returns bool type value.
func (k *Key) Bool() (bool, error) {
	return parseBool(k.String())
}

// Float64 returns float64 type value.
func (k *Key) Float64() (float64, error) {
	return strconv.ParseFloat(k.String(), 64)
}

// Int returns int type value.
func (k *Key) Int() (int, error) {
	return strconv.Atoi(k.String())
}

// Int64 returns int64 type value.
func (k *Key) Int64() (int64, error) {
	return strconv.ParseInt(k.String(), 10, 64)
}

// Uint returns uint type valued.
func (k *Key) Uint() (uint, error) {
	u, e := strconv.ParseUint(k.String(), 10, 64)
	return uint(u), e
}

// Uint64 returns uint64 type value.
func (k *Key) Uint64() (uint64, error) {
	return strconv.ParseUint(k.String(), 10, 64)
}

// Duration returns time.Duration type value.
func (k *Key) Duration() (time.Duration, error) {
	return time.ParseDuration(k.String())
}

// TimeFormat parses with given format and returns time.Time type value.
func (k *Key) TimeFormat(format string) (time.Time, error) {
	return time.Parse(format, k.String())
}

// Time parses with RFC3339 format and returns time.Time type value.
func (k *Key) Time() (time.Time, error) {
	return k.TimeFormat(time.RFC3339)
}

// MustString returns default value if key value is empty.
func (k *Key) MustString(defaultVal string) string {
	val := k.String()
	if len(val) == 0 {
		return defaultVal
	}
	return val
}

// MustBool always returns value without error,
// it returns false if error occurs.
func (k *Key) MustBool(defaultVal ...bool) bool {
	val, err := k.Bool()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustFloat64 always returns value without error,
// it returns 0.0 if error occurs.
func (k *Key) MustFloat64(defaultVal ...float64) float64 {
	val, err := k.Float64()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustInt always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustInt(defaultVal ...int) int {
	val, err := k.Int()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustInt64 always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustInt64(defaultVal ...int64) int64 {
	val, err := k.Int64()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustUint always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustUint(defaultVal ...uint) uint {
	val, err := k.Uint()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustUint64 always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustUint64(defaultVal ...uint64) uint64 {
	val, err := k.Uint64()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustDuration always returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustDuration(defaultVal ...time.Duration) time.Duration {
	val, err := k.Duration()
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustTimeFormat always parses with given format and returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustTimeFormat(format string, defaultVal ...time.Time) time.Time {
	val, err := k.TimeFormat(format)
	if len(defaultVal) > 0 && err != nil {
		return defaultVal[0]
	}
	return val
}

// MustTime always parses with RFC3339 format and returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustTime(defaultVal ...time.Time) time.Time {
	return k.MustTimeFormat(time.RFC3339, defaultVal...)
}

// In always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) In(defaultVal string, candidates []string) string {
	val := k.String()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InFloat64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InFloat64(defaultVal float64, candidates []float64) float64 {
	val := k.MustFloat64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InInt always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InInt(defaultVal int, candidates []int) int {
	val := k.MustInt()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InInt64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InInt64(defaultVal int64, candidates []int64) int64 {
	val := k.MustInt64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InUint always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InUint(defaultVal uint, candidates []uint) uint {
	val := k.MustUint()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InUint64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InUint64(defaultVal uint64, candidates []uint64) uint64 {
	val := k.MustUint64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InTimeFormat always parses with given format and returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InTimeFormat(format string, defaultVal time.Time, candidates []time.Time) time.Time {
	val := k.MustTimeFormat(format)
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InTime always parses with RFC3339 format and returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InTime(defaultVal time.Time, candidates []time.Time) time.Time {
	return k.InTimeFormat(time.RFC3339, defaultVal, candidates)
}

// RangeFloat64 checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeFloat64(defaultVal, min, max float64) float64 {
	val := k.MustFloat64()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeInt checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeInt(defaultVal, min, max int) int {
	val := k.MustInt()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeInt64 checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeInt64(defaultVal, min, max int64) int64 {
	val := k.MustInt64()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeTimeFormat checks if value with given format is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeTimeFormat(format string, defaultVal, min, max time.Time) time.Time {
	val := k.MustTimeFormat(format)
	if val.Unix() < min.Unix() || val.Unix() > max.Unix() {
		return defaultVal
	}
	return val
}

// RangeTime checks if value with RFC3339 format is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeTime(defaultVal, min, max time.Time) time.Time {
	return k.RangeTimeFormat(time.RFC3339, defaultVal, min, max)
}

// Strings returns list of string devide by given delimiter.
func (k *Key) Strings(delim string) []string {
	str := k.String()
	if len(str) == 0 {
		return []string{}
	}

	vals := strings.Split(str, delim)
	for i := range vals {
		vals[i] = strings.TrimSpace(vals[i])
	}
	return vals
}

// Float64s returns list of float64 devide by given delimiter.
func (k *Key) Float64s(delim string) []float64 {
	strs := k.Strings(delim)
	vals := make([]float64, len(strs))
	for i := range strs {
		vals[i], _ = strconv.ParseFloat(strs[i], 64)
	}
	return vals
}

// Ints returns list of int devide by given delimiter.
func (k *Key) Ints(delim string) []int {
	strs := k.Strings(delim)
	vals := make([]int, len(strs))
	for i := range strs {
		vals[i], _ = strconv.Atoi(strs[i])
	}
	return vals
}

// Int64s returns list of int64 devide by given delimiter.
func (k *Key) Int64s(delim string) []int64 {
	strs := k.Strings(delim)
	vals := make([]int64, len(strs))
	for i := range strs {
		vals[i], _ = strconv.ParseInt(strs[i], 10, 64)
	}
	return vals
}

// Uints returns list of uint devide by given delimiter.
func (k *Key) Uints(delim string) []uint {
	strs := k.Strings(delim)
	vals := make([]uint, len(strs))
	for i := range strs {
		u, _ := strconv.ParseUint(strs[i], 10, 64)
		vals[i] = uint(u)
	}
	return vals
}

// Uint64s returns list of uint64 devide by given delimiter.
func (k *Key) Uint64s(delim string) []uint64 {
	strs := k.Strings(delim)
	vals := make([]uint64, len(strs))
	for i := range strs {
		vals[i], _ = strconv.ParseUint(strs[i], 10, 64)
	}
	return vals
}

// TimesFormat parses with given format and returns list of time.Time devide by given delimiter.
func (k *Key) TimesFormat(format, delim string) []time.Time {
	strs := k.Strings(delim)
	vals := make([]time.Time, len(strs))
	for i := range strs {
		vals[i], _ = time.Parse(format, strs[i])
	}
	return vals
}

// Times parses with RFC3339 format and returns list of time.Time devide by given delimiter.
func (k *Key) Times(delim string) []time.Time {
	return k.TimesFormat(time.RFC3339, delim)
}

// SetValue changes key value.
func (k *Key) SetValue(v string) {
	k.value = v
}

//   _________              __  .__
//  /   _____/ ____   _____/  |_|__| ____   ____
//  \_____  \_/ __ \_/ ___\   __\  |/  _ \ /    \
//  /        \  ___/\  \___|  | |  (  <_> )   |  \
// /_______  /\___  >\___  >__| |__|\____/|___|  /
//         \/     \/     \/                    \/

// Section represents a config section.
type Section struct {
	f        *File
	Comment  string
	name     string
	keys     map[string]*Key
	keyList  []string
	keysHash map[string]string
}

func newSection(f *File, name string) *Section {
	return &Section{f, "", name, make(map[string]*Key), make([]string, 0, 10), make(map[string]string)}
}

// Name returns name of Section.
func (s *Section) Name() string {
	return s.name
}

// NewKey creates a new key to given section.
func (s *Section) NewKey(name, val string) (*Key, error) {
	if len(name) == 0 {
		return nil, errors.New("error creating new key: empty key name")
	}

	if s.f.BlockMode {
		s.f.lock.Lock()
		defer s.f.lock.Unlock()
	}

	if inSlice(name, s.keyList) {
		s.keys[name].value = val
		return s.keys[name], nil
	}

	s.keyList = append(s.keyList, name)
	s.keys[name] = &Key{s, "", name, val, false}
	s.keysHash[name] = val
	return s.keys[name], nil
}

// GetKey returns key in section by given name.
func (s *Section) GetKey(name string) (*Key, error) {
	// FIXME: change to section level lock?
	if s.f.BlockMode {
		s.f.lock.RLock()
	}
	key := s.keys[name]
	if s.f.BlockMode {
		s.f.lock.RUnlock()
	}

	if key == nil {
		// Check if it is a child-section.
		sname := s.name
		for {
			if i := strings.LastIndex(sname, "."); i > -1 {
				sname = sname[:i]
				sec, err := s.f.GetSection(sname)
				if err != nil {
					continue
				}
				return sec.GetKey(name)
			} else {
				break
			}
		}
		return nil, fmt.Errorf("error when getting key of section '%s': key '%s' not exists", s.name, name)
	}
	return key, nil
}

// Key assumes named Key exists in section and returns a zero-value when not.
func (s *Section) Key(name string) *Key {
	key, err := s.GetKey(name)
	if err != nil {
		// It's OK here because the only possible error is empty key name,
		// but if it's empty, this piece of code won't be executed.
		key, _ = s.NewKey(name, "")
		return key
	}
	return key
}

// Keys returns list of keys of section.
func (s *Section) Keys() []*Key {
	keys := make([]*Key, len(s.keyList))
	for i := range s.keyList {
		keys[i] = s.Key(s.keyList[i])
	}
	return keys
}

// KeyStrings returns list of key names of section.
func (s *Section) KeyStrings() []string {
	list := make([]string, len(s.keyList))
	copy(list, s.keyList)
	return list
}

// KeysHash returns keys hash consisting of names and values.
func (s *Section) KeysHash() map[string]string {
	if s.f.BlockMode {
		s.f.lock.RLock()
		defer s.f.lock.RUnlock()
	}

	hash := map[string]string{}
	for key, value := range s.keysHash {
		hash[key] = value
	}
	return hash
}

// DeleteKey deletes a key from section.
func (s *Section) DeleteKey(name string) {
	if s.f.BlockMode {
		s.f.lock.Lock()
		defer s.f.lock.Unlock()
	}

	for i, k := range s.keyList {
		if k == name {
			s.keyList = append(s.keyList[:i], s.keyList[i+1:]...)
			delete(s.keys, name)
			return
		}
	}
}

// ___________.__.__
// \_   _____/|__|  |   ____
//  |    __)  |  |  | _/ __ \
//  |     \   |  |  |_\  ___/
//  \___  /   |__|____/\___  >
//      \/                 \/

// File represents a combination of a or more INI file(s) in memory.
type File struct {
	// Should make things safe, but sometimes doesn't matter.
	BlockMode bool
	// Make sure data is safe in multiple goroutines.
	lock sync.RWMutex

	// Allow combination of multiple data sources.
	dataSources []dataSource
	// Actual data is stored here.
	sections map[string]*Section

	// To keep data in order.
	sectionList []string

	NameMapper
}

// newFile initializes File object with given data sources.
func newFile(dataSources []dataSource) *File {
	return &File{
		BlockMode:   true,
		dataSources: dataSources,
		sections:    make(map[string]*Section),
		sectionList: make([]string, 0, 10),
	}
}

func parseDataSource(source interface{}) (dataSource, error) {
	switch s := source.(type) {
	case string:
		return sourceFile{s}, nil
	case []byte:
		return &sourceData{s}, nil
	default:
		return nil, fmt.Errorf("error parsing data source: unknown type '%s'", s)
	}
}

// Load loads and parses from INI data sources.
// Arguments can be mixed of file name with string type, or raw data in []byte.
func Load(source interface{}, others ...interface{}) (_ *File, err error) {
	sources := make([]dataSource, len(others)+1)
	sources[0], err = parseDataSource(source)
	if err != nil {
		return nil, err
	}
	for i := range others {
		sources[i+1], err = parseDataSource(others[i])
		if err != nil {
			return nil, err
		}
	}
	f := newFile(sources)
	return f, f.Reload()
}

// Empty returns an empty file object.
func Empty() *File {
	// Ignore error here, we sure our data is good.
	f, _ := Load([]byte(""))
	return f
}

// NewSection creates a new section.
func (f *File) NewSection(name string) (*Section, error) {
	if len(name) == 0 {
		return nil, errors.New("error creating new section: empty section name")
	}

	if f.BlockMode {
		f.lock.Lock()
		defer f.lock.Unlock()
	}

	if inSlice(name, f.sectionList) {
		return f.sections[name], nil
	}

	f.sectionList = append(f.sectionList, name)
	f.sections[name] = newSection(f, name)
	return f.sections[name], nil
}

// NewSections creates a list of sections.
func (f *File) NewSections(names ...string) (err error) {
	for _, name := range names {
		if _, err = f.NewSection(name); err != nil {
			return err
		}
	}
	return nil
}

// GetSection returns section by given name.
func (f *File) GetSection(name string) (*Section, error) {
	if len(name) == 0 {
		name = DEFAULT_SECTION
	}

	if f.BlockMode {
		f.lock.RLock()
		defer f.lock.RUnlock()
	}

	sec := f.sections[name]
	if sec == nil {
		return nil, fmt.Errorf("error when getting section: section '%s' not exists", name)
	}
	return sec, nil
}

// Section assumes named section exists and returns a zero-value when not.
func (f *File) Section(name string) *Section {
	sec, err := f.GetSection(name)
	if err != nil {
		// Note: It's OK here because the only possible error is empty section name,
		// but if it's empty, this piece of code won't be executed.
		sec, _ = f.NewSection(name)
		return sec
	}
	return sec
}

// Section returns list of Section.
func (f *File) Sections() []*Section {
	sections := make([]*Section, len(f.sectionList))
	for i := range f.sectionList {
		sections[i] = f.Section(f.sectionList[i])
	}
	return sections
}

// SectionStrings returns list of section names.
func (f *File) SectionStrings() []string {
	list := make([]string, len(f.sectionList))
	copy(list, f.sectionList)
	return list
}

// DeleteSection deletes a section.
func (f *File) DeleteSection(name string) {
	if f.BlockMode {
		f.lock.Lock()
		defer f.lock.Unlock()
	}

	if len(name) == 0 {
		name = DEFAULT_SECTION
	}

	for i, s := range f.sectionList {
		if s == name {
			f.sectionList = append(f.sectionList[:i], f.sectionList[i+1:]...)
			delete(f.sections, name)
			return
		}
	}
}

func cutComment(str string) string {
	i := strings.Index(str, "#")
	if i == -1 {
		return str
	}
	return str[:i]
}

func checkMultipleLines(buf *bufio.Reader, line, val, valQuote string) (string, error) {
	isEnd := false
	for {
		next, err := buf.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return "", err
			}
			isEnd = true
		}
		pos := strings.LastIndex(next, valQuote)
		if pos > -1 {
			val += next[:pos]
			break
		}
		val += next
		if isEnd {
			return "", fmt.Errorf("error parsing line: missing closing key quote from '%s' to '%s'", line, next)
		}
	}
	return val, nil
}

func checkContinuationLines(buf *bufio.Reader, val string) (string, bool, error) {
	isEnd := false
	for {
		valLen := len(val)
		if valLen == 0 || val[valLen-1] != '\\' {
			break
		}
		val = val[:valLen-1]

		next, err := buf.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return "", isEnd, err
			}
			isEnd = true
		}

		next = strings.TrimSpace(next)
		if len(next) == 0 {
			break
		}
		val += next
	}
	return val, isEnd, nil
}

// parse parses data through an io.Reader.
func (f *File) parse(reader io.Reader) error {
	buf := bufio.NewReader(reader)

	// Handle BOM-UTF8.
	// http://en.wikipedia.org/wiki/Byte_order_mark#Representations_of_byte_order_marks_by_encoding
	mask, err := buf.Peek(3)
	if err == nil && len(mask) >= 3 && mask[0] == 239 && mask[1] == 187 && mask[2] == 191 {
		buf.Read(mask)
	}

	count := 1
	comments := ""
	isEnd := false

	section, err := f.NewSection(DEFAULT_SECTION)
	if err != nil {
		return err
	}

	for {
		line, err := buf.ReadString('\n')
		line = strings.TrimSpace(line)
		length := len(line)

		// Check error and ignore io.EOF just for a moment.
		if err != nil {
			if err != io.EOF {
				return fmt.Errorf("error reading next line: %v", err)
			}
			// The last line of file could be an empty line.
			if length == 0 {
				break
			}
			isEnd = true
		}

		// Skip empty lines.
		if length == 0 {
			continue
		}

		switch {
		case line[0] == '#' || line[0] == ';': // Comments.
			if len(comments) == 0 {
				comments = line
			} else {
				comments += LineBreak + line
			}
			continue
		case line[0] == '[' && line[length-1] == ']': // New sction.
			section, err = f.NewSection(strings.TrimSpace(line[1 : length-1]))
			if err != nil {
				return err
			}

			if len(comments) > 0 {
				section.Comment = comments
				comments = ""
			}
			// Reset counter.
			count = 1
			continue
		}

		// Other possibilities.
		var (
			i        int
			keyQuote string
			kname    string
			valQuote string
			val      string
		)

		// Key name surrounded by quotes.
		if line[0] == '"' {
			if length > 6 && line[0:3] == `"""` {
				keyQuote = `"""`
			} else {
				keyQuote = `"`
			}
		} else if line[0] == '`' {
			keyQuote = "`"
		}
		if len(keyQuote) > 0 {
			qLen := len(keyQuote)
			pos := strings.Index(line[qLen:], keyQuote)
			if pos == -1 {
				return fmt.Errorf("error parsing line: missing closing key quote: %s", line)
			}
			pos = pos + qLen
			i = strings.IndexAny(line[pos:], "=:")
			if i < 0 {
				return fmt.Errorf("error parsing line: key-value delimiter not found: %s", line)
			} else if i == pos {
				return fmt.Errorf("error parsing line: key is empty: %s", line)
			}
			i = i + pos
			kname = line[qLen:pos] // Just keep spaces inside quotes.
		} else {
			i = strings.IndexAny(line, "=:")
			if i < 0 {
				return fmt.Errorf("error parsing line: key-value delimiter not found: %s", line)
			} else if i == 0 {
				return fmt.Errorf("error parsing line: key is empty: %s", line)
			}
			kname = strings.TrimSpace(line[0:i])
		}

		isAutoIncr := false
		// Auto increment.
		if kname == "-" {
			isAutoIncr = true
			kname = "#" + fmt.Sprint(count)
			count++
		}

		lineRight := strings.TrimSpace(line[i+1:])
		lineRightLength := len(lineRight)
		firstChar := ""
		if lineRightLength >= 2 {
			firstChar = lineRight[0:1]
		}
		if firstChar == "`" {
			valQuote = "`"
		} else if firstChar == `"` {
			if lineRightLength >= 3 && lineRight[0:3] == `"""` {
				valQuote = `"""`
			} else {
				valQuote = `"`
			}
		} else if firstChar == `'` {
			valQuote = `'`
		}

		if len(valQuote) > 0 {
			qLen := len(valQuote)
			pos := strings.LastIndex(lineRight[qLen:], valQuote)
			// For multiple-line value check.
			if pos == -1 {
				if valQuote == `"` || valQuote == `'` {
					return fmt.Errorf("error parsing line: single quote does not allow multiple-line value: %s", line)
				}

				val = lineRight[qLen:] + "\n"
				val, err = checkMultipleLines(buf, line, val, valQuote)
				if err != nil {
					return err
				}
			} else {
				val = lineRight[qLen : pos+qLen]
			}
		} else {
			val = strings.TrimSpace(cutComment(lineRight))
			val, isEnd, err = checkContinuationLines(buf, val)
			if err != nil {
				return err
			}
		}

		k, err := section.NewKey(kname, val)
		if err != nil {
			return err
		}
		k.isAutoIncr = isAutoIncr
		if len(comments) > 0 {
			k.Comment = comments
			comments = ""
		}

		if isEnd {
			break
		}
	}
	return nil
}

func (f *File) reload(s dataSource) error {
	r, err := s.ReadCloser()
	if err != nil {
		return err
	}
	defer r.Close()

	return f.parse(r)
}

// Reload reloads and parses all data sources.
func (f *File) Reload() (err error) {
	for _, s := range f.dataSources {
		if err = f.reload(s); err != nil {
			return err
		}
	}
	return nil
}

// Append appends one or more data sources and reloads automatically.
func (f *File) Append(source interface{}, others ...interface{}) error {
	ds, err := parseDataSource(source)
	if err != nil {
		return err
	}
	f.dataSources = append(f.dataSources, ds)
	for _, s := range others {
		ds, err = parseDataSource(s)
		if err != nil {
			return err
		}
		f.dataSources = append(f.dataSources, ds)
	}
	return f.Reload()
}

// WriteToIndent writes file content into io.Writer with given value indention.
func (f *File) WriteToIndent(w io.Writer, indent string) (n int64, err error) {
	equalSign := "="
	if PrettyFormat {
		equalSign = " = "
	}

	// Use buffer to make sure target is safe until finish encoding.
	buf := bytes.NewBuffer(nil)
	for i, sname := range f.sectionList {
		sec := f.Section(sname)
		if len(sec.Comment) > 0 {
			if sec.Comment[0] != '#' && sec.Comment[0] != ';' {
				sec.Comment = "; " + sec.Comment
			}
			if _, err = buf.WriteString(sec.Comment + LineBreak); err != nil {
				return 0, err
			}
		}

		if i > 0 {
			if _, err = buf.WriteString("[" + sname + "]" + LineBreak); err != nil {
				return 0, err
			}
		} else {
			// Write nothing if default section is empty.
			if len(sec.keyList) == 0 {
				continue
			}
		}

		for _, kname := range sec.keyList {
			key := sec.Key(kname)
			if len(key.Comment) > 0 {
				if len(indent) > 0 && sname != DEFAULT_SECTION {
					buf.WriteString(indent)
				}
				if key.Comment[0] != '#' && key.Comment[0] != ';' {
					key.Comment = "; " + key.Comment
				}
				if _, err = buf.WriteString(key.Comment + LineBreak); err != nil {
					return 0, err
				}
			}

			if len(indent) > 0 && sname != DEFAULT_SECTION {
				buf.WriteString(indent)
			}

			switch {
			case key.isAutoIncr:
				kname = "-"
			case strings.Contains(kname, "`") || strings.Contains(kname, `"`):
				kname = `"""` + kname + `"""`
			case strings.Contains(kname, `=`) || strings.Contains(kname, `:`):
				kname = "`" + kname + "`"
			}

			val := key.value
			// In case key value contains "\n", "`" or "\"".
			if strings.Contains(val, "\n") || strings.Contains(val, "`") || strings.Contains(val, `"`) ||
				strings.Contains(val, "#") {
				val = `"""` + val + `"""`
			}
			if _, err = buf.WriteString(kname + equalSign + val + LineBreak); err != nil {
				return 0, err
			}
		}

		// Put a line between sections.
		if _, err = buf.WriteString(LineBreak); err != nil {
			return 0, err
		}
	}

	return buf.WriteTo(w)
}

// WriteTo writes file content into io.Writer.
func (f *File) WriteTo(w io.Writer) (int64, error) {
	return f.WriteToIndent(w, "")
}

// SaveToIndent writes content to file system with given value indention.
func (f *File) SaveToIndent(filename, indent string) error {
	// Note: Because we are truncating with os.Create,
	// 	so it's safer to save to a temporary file location and rename afte done.
	tmpPath := filename + "." + strconv.Itoa(time.Now().Nanosecond()) + ".tmp"
	defer os.Remove(tmpPath)

	fw, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	if _, err = f.WriteToIndent(fw, indent); err != nil {
		fw.Close()
		return err
	}
	fw.Close()

	// Remove old file and rename the new one.
	os.Remove(filename)
	return os.Rename(tmpPath, filename)
}

// SaveTo writes content to file system.
func (f *File) SaveTo(filename string) error {
	return f.SaveToIndent(filename, "")
}
