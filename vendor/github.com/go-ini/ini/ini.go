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
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// Name for default section. You can use this constant or the string literal.
	// In most of cases, an empty string is all you need to access the section.
	DEFAULT_SECTION = "DEFAULT"

	// Maximum allowed depth when recursively substituing variable names.
	_DEPTH_VALUES = 99
	_VERSION      = "1.25.4"
)

// Version returns current package version literal.
func Version() string {
	return _VERSION
}

var (
	// Delimiter to determine or compose a new line.
	// This variable will be changed to "\r\n" automatically on Windows
	// at package init time.
	LineBreak = "\n"

	// Variable regexp pattern: %(variable)s
	varPattern = regexp.MustCompile(`%\(([^\)]+)\)s`)

	// Indicate whether to align "=" sign with spaces to produce pretty output
	// or reduce all possible spaces for compact format.
	PrettyFormat = true

	// Explicitly write DEFAULT section header
	DefaultHeader = false
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

// dataSource is an interface that returns object which can be read and closed.
type dataSource interface {
	ReadCloser() (io.ReadCloser, error)
}

// sourceFile represents an object that contains content on the local file system.
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

// sourceData represents an object that contains content in memory.
type sourceData struct {
	data []byte
}

func (s *sourceData) ReadCloser() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewReader(s.data)), nil
}

// sourceReadCloser represents an input stream with Close method.
type sourceReadCloser struct {
	reader io.ReadCloser
}

func (s *sourceReadCloser) ReadCloser() (io.ReadCloser, error) {
	return s.reader, nil
}

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

	options LoadOptions

	NameMapper
	ValueMapper
}

// newFile initializes File object with given data sources.
func newFile(dataSources []dataSource, opts LoadOptions) *File {
	return &File{
		BlockMode:   true,
		dataSources: dataSources,
		sections:    make(map[string]*Section),
		sectionList: make([]string, 0, 10),
		options:     opts,
	}
}

func parseDataSource(source interface{}) (dataSource, error) {
	switch s := source.(type) {
	case string:
		return sourceFile{s}, nil
	case []byte:
		return &sourceData{s}, nil
	case io.ReadCloser:
		return &sourceReadCloser{s}, nil
	default:
		return nil, fmt.Errorf("error parsing data source: unknown type '%s'", s)
	}
}

type LoadOptions struct {
	// Loose indicates whether the parser should ignore nonexistent files or return error.
	Loose bool
	// Insensitive indicates whether the parser forces all section and key names to lowercase.
	Insensitive bool
	// IgnoreContinuation indicates whether to ignore continuation lines while parsing.
	IgnoreContinuation bool
	// AllowBooleanKeys indicates whether to allow boolean type keys or treat as value is missing.
	// This type of keys are mostly used in my.cnf.
	AllowBooleanKeys bool
	// AllowShadows indicates whether to keep track of keys with same name under same section.
	AllowShadows bool
	// Some INI formats allow group blocks that store a block of raw content that doesn't otherwise
	// conform to key/value pairs. Specify the names of those blocks here.
	UnparseableSections []string
}

func LoadSources(opts LoadOptions, source interface{}, others ...interface{}) (_ *File, err error) {
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
	f := newFile(sources, opts)
	if err = f.Reload(); err != nil {
		return nil, err
	}
	return f, nil
}

// Load loads and parses from INI data sources.
// Arguments can be mixed of file name with string type, or raw data in []byte.
// It will return error if list contains nonexistent files.
func Load(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{}, source, others...)
}

// LooseLoad has exactly same functionality as Load function
// except it ignores nonexistent files instead of returning error.
func LooseLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{Loose: true}, source, others...)
}

// InsensitiveLoad has exactly same functionality as Load function
// except it forces all section and key names to be lowercased.
func InsensitiveLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{Insensitive: true}, source, others...)
}

// InsensitiveLoad has exactly same functionality as Load function
// except it allows have shadow keys.
func ShadowLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{AllowShadows: true}, source, others...)
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
	} else if f.options.Insensitive && name != DEFAULT_SECTION {
		name = strings.ToLower(name)
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

// NewRawSection creates a new section with an unparseable body.
func (f *File) NewRawSection(name, body string) (*Section, error) {
	section, err := f.NewSection(name)
	if err != nil {
		return nil, err
	}

	section.isRawSection = true
	section.rawBody = body
	return section, nil
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
	} else if f.options.Insensitive {
		name = strings.ToLower(name)
	}

	if f.BlockMode {
		f.lock.RLock()
		defer f.lock.RUnlock()
	}

	sec := f.sections[name]
	if sec == nil {
		return nil, fmt.Errorf("section '%s' does not exist", name)
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
			// In loose mode, we create an empty default section for nonexistent files.
			if os.IsNotExist(err) && f.options.Loose {
				f.parse(bytes.NewBuffer(nil))
				continue
			}
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

// WriteToIndent writes content into io.Writer with given indention.
// If PrettyFormat has been set to be true,
// it will align "=" sign with spaces under each section.
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

		if i > 0 || DefaultHeader {
			if _, err = buf.WriteString("[" + sname + "]" + LineBreak); err != nil {
				return 0, err
			}
		} else {
			// Write nothing if default section is empty
			if len(sec.keyList) == 0 {
				continue
			}
		}

		if sec.isRawSection {
			if _, err = buf.WriteString(sec.rawBody); err != nil {
				return 0, err
			}
			continue
		}

		// Count and generate alignment length and buffer spaces using the
		// longest key. Keys may be modifed if they contain certain characters so
		// we need to take that into account in our calculation.
		alignLength := 0
		if PrettyFormat {
			for _, kname := range sec.keyList {
				keyLength := len(kname)
				// First case will surround key by ` and second by """
				if strings.ContainsAny(kname, "\"=:") {
					keyLength += 2
				} else if strings.Contains(kname, "`") {
					keyLength += 6
				}

				if keyLength > alignLength {
					alignLength = keyLength
				}
			}
		}
		alignSpaces := bytes.Repeat([]byte(" "), alignLength)

	KEY_LIST:
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
			case key.isAutoIncrement:
				kname = "-"
			case strings.ContainsAny(kname, "\"=:"):
				kname = "`" + kname + "`"
			case strings.Contains(kname, "`"):
				kname = `"""` + kname + `"""`
			}

			for _, val := range key.ValueWithShadows() {
				if _, err = buf.WriteString(kname); err != nil {
					return 0, err
				}

				if key.isBooleanType {
					if kname != sec.keyList[len(sec.keyList)-1] {
						buf.WriteString(LineBreak)
					}
					continue KEY_LIST
				}

				// Write out alignment spaces before "=" sign
				if PrettyFormat {
					buf.Write(alignSpaces[:alignLength-len(kname)])
				}

				// In case key value contains "\n", "`", "\"", "#" or ";"
				if strings.ContainsAny(val, "\n`") {
					val = `"""` + val + `"""`
				} else if strings.ContainsAny(val, "#;") {
					val = "`" + val + "`"
				}
				if _, err = buf.WriteString(equalSign + val + LineBreak); err != nil {
					return 0, err
				}
			}
		}

		// Put a line between sections
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
