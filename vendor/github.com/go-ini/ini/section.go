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

package ini

import (
	"errors"
	"fmt"
	"strings"
)

// Section represents a config section.
type Section struct {
	f        *File
	Comment  string
	name     string
	keys     map[string]*Key
	keyList  []string
	keysHash map[string]string

	isRawSection bool
	rawBody      string
}

func newSection(f *File, name string) *Section {
	return &Section{
		f:        f,
		name:     name,
		keys:     make(map[string]*Key),
		keyList:  make([]string, 0, 10),
		keysHash: make(map[string]string),
	}
}

// Name returns name of Section.
func (s *Section) Name() string {
	return s.name
}

// Body returns rawBody of Section if the section was marked as unparseable.
// It still follows the other rules of the INI format surrounding leading/trailing whitespace.
func (s *Section) Body() string {
	return strings.TrimSpace(s.rawBody)
}

// NewKey creates a new key to given section.
func (s *Section) NewKey(name, val string) (*Key, error) {
	if len(name) == 0 {
		return nil, errors.New("error creating new key: empty key name")
	} else if s.f.options.Insensitive {
		name = strings.ToLower(name)
	}

	if s.f.BlockMode {
		s.f.lock.Lock()
		defer s.f.lock.Unlock()
	}

	if inSlice(name, s.keyList) {
		if s.f.options.AllowShadows {
			if err := s.keys[name].addShadow(val); err != nil {
				return nil, err
			}
		} else {
			s.keys[name].value = val
		}
		return s.keys[name], nil
	}

	s.keyList = append(s.keyList, name)
	s.keys[name] = newKey(s, name, val)
	s.keysHash[name] = val
	return s.keys[name], nil
}

// NewBooleanKey creates a new boolean type key to given section.
func (s *Section) NewBooleanKey(name string) (*Key, error) {
	key, err := s.NewKey(name, "true")
	if err != nil {
		return nil, err
	}

	key.isBooleanType = true
	return key, nil
}

// GetKey returns key in section by given name.
func (s *Section) GetKey(name string) (*Key, error) {
	// FIXME: change to section level lock?
	if s.f.BlockMode {
		s.f.lock.RLock()
	}
	if s.f.options.Insensitive {
		name = strings.ToLower(name)
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

// HasKey returns true if section contains a key with given name.
func (s *Section) HasKey(name string) bool {
	key, _ := s.GetKey(name)
	return key != nil
}

// Haskey is a backwards-compatible name for HasKey.
func (s *Section) Haskey(name string) bool {
	return s.HasKey(name)
}

// HasValue returns true if section contains given raw value.
func (s *Section) HasValue(value string) bool {
	if s.f.BlockMode {
		s.f.lock.RLock()
		defer s.f.lock.RUnlock()
	}

	for _, k := range s.keys {
		if value == k.value {
			return true
		}
	}
	return false
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

// ParentKeys returns list of keys of parent section.
func (s *Section) ParentKeys() []*Key {
	var parentKeys []*Key
	sname := s.name
	for {
		if i := strings.LastIndex(sname, "."); i > -1 {
			sname = sname[:i]
			sec, err := s.f.GetSection(sname)
			if err != nil {
				continue
			}
			parentKeys = append(parentKeys, sec.Keys()...)
		} else {
			break
		}

	}
	return parentKeys
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

// ChildSections returns a list of child sections of current section.
// For example, "[parent.child1]" and "[parent.child12]" are child sections
// of section "[parent]".
func (s *Section) ChildSections() []*Section {
	prefix := s.name + "."
	children := make([]*Section, 0, 3)
	for _, name := range s.f.sectionList {
		if strings.HasPrefix(name, prefix) {
			children = append(children, s.f.sections[name])
		}
	}
	return children
}
