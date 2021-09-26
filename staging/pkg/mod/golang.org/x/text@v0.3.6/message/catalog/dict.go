// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catalog

import (
	"sync"

	"golang.org/x/text/internal"
	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
)

// TODO:
// Dictionary returns a Dictionary that returns the first Message, using the
// given language tag, that matches:
//   1. the last one registered by one of the Set methods
//   2. returned by one of the Loaders
//   3. repeat from 1. using the parent language
// This approach allows messages to be underspecified.
// func (c *Catalog) Dictionary(tag language.Tag) (Dictionary, error) {
// 	// TODO: verify dictionary exists.
// 	return &dict{&c.index, tag}, nil
// }

type dict struct {
	s   *store
	tag language.Tag // TODO: make compact tag.
}

func (d *dict) Lookup(key string) (data string, ok bool) {
	return d.s.lookup(d.tag, key)
}

func (b *Builder) lookup(tag language.Tag, key string) (data string, ok bool) {
	return b.index.lookup(tag, key)
}

func (c *Builder) set(tag language.Tag, key string, s *store, msg ...Message) error {
	data, err := catmsg.Compile(tag, &dict{&c.macros, tag}, firstInSequence(msg))

	s.mutex.Lock()
	defer s.mutex.Unlock()

	m := s.index[tag]
	if m == nil {
		m = msgMap{}
		if s.index == nil {
			s.index = map[language.Tag]msgMap{}
		}
		c.matcher = nil
		s.index[tag] = m
	}

	m[key] = data
	return err
}

func (c *Builder) Matcher() language.Matcher {
	c.index.mutex.RLock()
	m := c.matcher
	c.index.mutex.RUnlock()
	if m != nil {
		return m
	}

	c.index.mutex.Lock()
	if c.matcher == nil {
		c.matcher = language.NewMatcher(c.unlockedLanguages())
	}
	m = c.matcher
	c.index.mutex.Unlock()
	return m
}

type store struct {
	mutex sync.RWMutex
	index map[language.Tag]msgMap
}

type msgMap map[string]string

func (s *store) lookup(tag language.Tag, key string) (data string, ok bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	for ; ; tag = tag.Parent() {
		if msgs, ok := s.index[tag]; ok {
			if msg, ok := msgs[key]; ok {
				return msg, true
			}
		}
		if tag == language.Und {
			break
		}
	}
	return "", false
}

// Languages returns all languages for which the Catalog contains variants.
func (b *Builder) Languages() []language.Tag {
	s := &b.index
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return b.unlockedLanguages()
}

func (b *Builder) unlockedLanguages() []language.Tag {
	s := &b.index
	if len(s.index) == 0 {
		return nil
	}
	tags := make([]language.Tag, 0, len(s.index))
	_, hasFallback := s.index[b.options.fallback]
	offset := 0
	if hasFallback {
		tags = append(tags, b.options.fallback)
		offset = 1
	}
	for t := range s.index {
		if t != b.options.fallback {
			tags = append(tags, t)
		}
	}
	internal.SortTags(tags[offset:])
	return tags
}
