// Copyright 2017 Unknwon
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

package ini

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"sync"
)

// File represents a combination of one or more INI files in memory.
type File struct {
	options     LoadOptions
	dataSources []dataSource

	// Should make things safe, but sometimes doesn't matter.
	BlockMode bool
	lock      sync.RWMutex

	// To keep data in order.
	sectionList []string
	// To keep track of the index of a section with same name.
	// This meta list is only used with non-unique section names are allowed.
	sectionIndexes []int

	// Actual data is stored here.
	sections map[string][]*Section

	NameMapper
	ValueMapper
}

// newFile initializes File object with given data sources.
func newFile(dataSources []dataSource, opts LoadOptions) *File {
	if len(opts.KeyValueDelimiters) == 0 {
		opts.KeyValueDelimiters = "=:"
	}
	if len(opts.KeyValueDelimiterOnWrite) == 0 {
		opts.KeyValueDelimiterOnWrite = "="
	}
	if len(opts.ChildSectionDelimiter) == 0 {
		opts.ChildSectionDelimiter = "."
	}

	return &File{
		BlockMode:   true,
		dataSources: dataSources,
		sections:    make(map[string][]*Section),
		options:     opts,
	}
}

// Empty returns an empty file object.
func Empty(opts ...LoadOptions) *File {
	var opt LoadOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	// Ignore error here, we are sure our data is good.
	f, _ := LoadSources(opt, []byte(""))
	return f
}

// NewSection creates a new section.
func (f *File) NewSection(name string) (*Section, error) {
	if len(name) == 0 {
		return nil, errors.New("empty section name")
	}

	if (f.options.Insensitive || f.options.InsensitiveSections) && name != DefaultSection {
		name = strings.ToLower(name)
	}

	if f.BlockMode {
		f.lock.Lock()
		defer f.lock.Unlock()
	}

	if !f.options.AllowNonUniqueSections && inSlice(name, f.sectionList) {
		return f.sections[name][0], nil
	}

	f.sectionList = append(f.sectionList, name)

	// NOTE: Append to indexes must happen before appending to sections,
	// otherwise index will have off-by-one problem.
	f.sectionIndexes = append(f.sectionIndexes, len(f.sections[name]))

	sec := newSection(f, name)
	f.sections[name] = append(f.sections[name], sec)

	return sec, nil
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
	secs, err := f.SectionsByName(name)
	if err != nil {
		return nil, err
	}

	return secs[0], err
}

// HasSection returns true if the file contains a section with given name.
func (f *File) HasSection(name string) bool {
	section, _ := f.GetSection(name)
	return section != nil
}

// SectionsByName returns all sections with given name.
func (f *File) SectionsByName(name string) ([]*Section, error) {
	if len(name) == 0 {
		name = DefaultSection
	}
	if f.options.Insensitive || f.options.InsensitiveSections {
		name = strings.ToLower(name)
	}

	if f.BlockMode {
		f.lock.RLock()
		defer f.lock.RUnlock()
	}

	secs := f.sections[name]
	if len(secs) == 0 {
		return nil, fmt.Errorf("section %q does not exist", name)
	}

	return secs, nil
}

// Section assumes named section exists and returns a zero-value when not.
func (f *File) Section(name string) *Section {
	sec, err := f.GetSection(name)
	if err != nil {
		if name == "" {
			name = DefaultSection
		}
		sec, _ = f.NewSection(name)
		return sec
	}
	return sec
}

// SectionWithIndex assumes named section exists and returns a new section when not.
func (f *File) SectionWithIndex(name string, index int) *Section {
	secs, err := f.SectionsByName(name)
	if err != nil || len(secs) <= index {
		// NOTE: It's OK here because the only possible error is empty section name,
		// but if it's empty, this piece of code won't be executed.
		newSec, _ := f.NewSection(name)
		return newSec
	}

	return secs[index]
}

// Sections returns a list of Section stored in the current instance.
func (f *File) Sections() []*Section {
	if f.BlockMode {
		f.lock.RLock()
		defer f.lock.RUnlock()
	}

	sections := make([]*Section, len(f.sectionList))
	for i, name := range f.sectionList {
		sections[i] = f.sections[name][f.sectionIndexes[i]]
	}
	return sections
}

// ChildSections returns a list of child sections of given section name.
func (f *File) ChildSections(name string) []*Section {
	return f.Section(name).ChildSections()
}

// SectionStrings returns list of section names.
func (f *File) SectionStrings() []string {
	list := make([]string, len(f.sectionList))
	copy(list, f.sectionList)
	return list
}

// DeleteSection deletes a section or all sections with given name.
func (f *File) DeleteSection(name string) {
	secs, err := f.SectionsByName(name)
	if err != nil {
		return
	}

	for i := 0; i < len(secs); i++ {
		// For non-unique sections, it is always needed to remove the first one so
		// in the next iteration, the subsequent section continue having index 0.
		// Ignoring the error as index 0 never returns an error.
		_ = f.DeleteSectionWithIndex(name, 0)
	}
}

// DeleteSectionWithIndex deletes a section with given name and index.
func (f *File) DeleteSectionWithIndex(name string, index int) error {
	if !f.options.AllowNonUniqueSections && index != 0 {
		return fmt.Errorf("delete section with non-zero index is only allowed when non-unique sections is enabled")
	}

	if len(name) == 0 {
		name = DefaultSection
	}
	if f.options.Insensitive || f.options.InsensitiveSections {
		name = strings.ToLower(name)
	}

	if f.BlockMode {
		f.lock.Lock()
		defer f.lock.Unlock()
	}

	// Count occurrences of the sections
	occurrences := 0

	sectionListCopy := make([]string, len(f.sectionList))
	copy(sectionListCopy, f.sectionList)

	for i, s := range sectionListCopy {
		if s != name {
			continue
		}

		if occurrences == index {
			if len(f.sections[name]) <= 1 {
				delete(f.sections, name) // The last one in the map
			} else {
				f.sections[name] = append(f.sections[name][:index], f.sections[name][index+1:]...)
			}

			// Fix section lists
			f.sectionList = append(f.sectionList[:i], f.sectionList[i+1:]...)
			f.sectionIndexes = append(f.sectionIndexes[:i], f.sectionIndexes[i+1:]...)

		} else if occurrences > index {
			// Fix the indices of all following sections with this name.
			f.sectionIndexes[i-1]--
		}

		occurrences++
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
			// In loose mode, we create an empty default section for nonexistent files.
			if os.IsNotExist(err) && f.options.Loose {
				_ = f.parse(bytes.NewBuffer(nil))
				continue
			}
			return err
		}
		if f.options.ShortCircuit {
			return nil
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

func (f *File) writeToBuffer(indent string) (*bytes.Buffer, error) {
	equalSign := DefaultFormatLeft + f.options.KeyValueDelimiterOnWrite + DefaultFormatRight

	if PrettyFormat || PrettyEqual {
		equalSign = fmt.Sprintf(" %s ", f.options.KeyValueDelimiterOnWrite)
	}

	// Use buffer to make sure target is safe until finish encoding.
	buf := bytes.NewBuffer(nil)
	for i, sname := range f.sectionList {
		sec := f.SectionWithIndex(sname, f.sectionIndexes[i])
		if len(sec.Comment) > 0 {
			// Support multiline comments
			lines := strings.Split(sec.Comment, LineBreak)
			for i := range lines {
				if lines[i][0] != '#' && lines[i][0] != ';' {
					lines[i] = "; " + lines[i]
				} else {
					lines[i] = lines[i][:1] + " " + strings.TrimSpace(lines[i][1:])
				}

				if _, err := buf.WriteString(lines[i] + LineBreak); err != nil {
					return nil, err
				}
			}
		}

		if i > 0 || DefaultHeader || (i == 0 && strings.ToUpper(sec.name) != DefaultSection) {
			if _, err := buf.WriteString("[" + sname + "]" + LineBreak); err != nil {
				return nil, err
			}
		} else {
			// Write nothing if default section is empty
			if len(sec.keyList) == 0 {
				continue
			}
		}

		if sec.isRawSection {
			if _, err := buf.WriteString(sec.rawBody); err != nil {
				return nil, err
			}

			if PrettySection {
				// Put a line between sections
				if _, err := buf.WriteString(LineBreak); err != nil {
					return nil, err
				}
			}
			continue
		}

		// Count and generate alignment length and buffer spaces using the
		// longest key. Keys may be modified if they contain certain characters so
		// we need to take that into account in our calculation.
		alignLength := 0
		if PrettyFormat {
			for _, kname := range sec.keyList {
				keyLength := len(kname)
				// First case will surround key by ` and second by """
				if strings.Contains(kname, "\"") || strings.ContainsAny(kname, f.options.KeyValueDelimiters) {
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

	KeyList:
		for _, kname := range sec.keyList {
			key := sec.Key(kname)
			if len(key.Comment) > 0 {
				if len(indent) > 0 && sname != DefaultSection {
					buf.WriteString(indent)
				}

				// Support multiline comments
				lines := strings.Split(key.Comment, LineBreak)
				for i := range lines {
					if lines[i][0] != '#' && lines[i][0] != ';' {
						lines[i] = "; " + strings.TrimSpace(lines[i])
					} else {
						lines[i] = lines[i][:1] + " " + strings.TrimSpace(lines[i][1:])
					}

					if _, err := buf.WriteString(lines[i] + LineBreak); err != nil {
						return nil, err
					}
				}
			}

			if len(indent) > 0 && sname != DefaultSection {
				buf.WriteString(indent)
			}

			switch {
			case key.isAutoIncrement:
				kname = "-"
			case strings.Contains(kname, "\"") || strings.ContainsAny(kname, f.options.KeyValueDelimiters):
				kname = "`" + kname + "`"
			case strings.Contains(kname, "`"):
				kname = `"""` + kname + `"""`
			}

			for _, val := range key.ValueWithShadows() {
				if _, err := buf.WriteString(kname); err != nil {
					return nil, err
				}

				if key.isBooleanType {
					if kname != sec.keyList[len(sec.keyList)-1] {
						buf.WriteString(LineBreak)
					}
					continue KeyList
				}

				// Write out alignment spaces before "=" sign
				if PrettyFormat {
					buf.Write(alignSpaces[:alignLength-len(kname)])
				}

				// In case key value contains "\n", "`", "\"", "#" or ";"
				if strings.ContainsAny(val, "\n`") {
					val = `"""` + val + `"""`
				} else if !f.options.IgnoreInlineComment && strings.ContainsAny(val, "#;") {
					val = "`" + val + "`"
				} else if len(strings.TrimSpace(val)) != len(val) {
					val = `"` + val + `"`
				}
				if _, err := buf.WriteString(equalSign + val + LineBreak); err != nil {
					return nil, err
				}
			}

			for _, val := range key.nestedValues {
				if _, err := buf.WriteString(indent + "  " + val + LineBreak); err != nil {
					return nil, err
				}
			}
		}

		if PrettySection {
			// Put a line between sections
			if _, err := buf.WriteString(LineBreak); err != nil {
				return nil, err
			}
		}
	}

	return buf, nil
}

// WriteToIndent writes content into io.Writer with given indention.
// If PrettyFormat has been set to be true,
// it will align "=" sign with spaces under each section.
func (f *File) WriteToIndent(w io.Writer, indent string) (int64, error) {
	buf, err := f.writeToBuffer(indent)
	if err != nil {
		return 0, err
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
	// 	so it's safer to save to a temporary file location and rename after done.
	buf, err := f.writeToBuffer(indent)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filename, buf.Bytes(), 0666)
}

// SaveTo writes content to file system.
func (f *File) SaveTo(filename string) error {
	return f.SaveToIndent(filename, "")
}
