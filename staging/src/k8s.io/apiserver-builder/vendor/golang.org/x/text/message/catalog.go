// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

// TODO: some types in this file will need to be made public at some time.
// Documentation and method names will reflect this by using the exported name.

import (
	"sync"

	"golang.org/x/text/internal"
	"golang.org/x/text/internal/format"
	"golang.org/x/text/language"
)

// DefaultCatalog is used by SetString.
var DefaultCatalog *Catalog = newCatalog()

// SetString calls SetString on the default Catalog.
func SetString(tag language.Tag, key string, msg string) error {
	return DefaultCatalog.SetString(tag, key, msg)
}

// TODO:
// // SetSelect is a shorthand for DefaultCatalog.SetSelect.
// func SetSelect(tag language.Tag, key string, s ...format.Statement) error {
// 	return DefaultCatalog.SetSelect(tag, key, s...)
// }

type msgMap map[string]format.Statement

// A Catalog holds translations for messages for supported languages.
type Catalog struct {
	index map[language.Tag]msgMap

	mutex sync.Mutex // For locking all operations.
}

// Printer creates a Printer that uses c.
func (c *Catalog) Printer(tag language.Tag) *Printer {
	// TODO: pre-create indexes for tag lookup.
	return &Printer{
		tag: tag,
		cat: c,
	}
}

// NewCatalog returns a new Catalog. If a message is not present in a Catalog,
// the fallback Catalogs will be used in order as an alternative source.
func newCatalog(fallback ...*Catalog) *Catalog {
	// TODO: implement fallback.
	return &Catalog{
		index: map[language.Tag]msgMap{},
	}
}

// Languages returns a slice of all languages for which the Catalog contains
// variants.
func (c *Catalog) Languages() []language.Tag {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	tags := []language.Tag{}
	for t, _ := range c.index {
		tags = append(tags, t)
	}
	internal.SortTags(tags)
	return tags
}

// SetString sets the translation for the given language and key.
func (c *Catalog) SetString(tag language.Tag, key string, msg string) error {
	return c.set(tag, key, format.String(msg))
}

func (c *Catalog) get(tag language.Tag, key string) (msg string, ok bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	for ; ; tag = tag.Parent() {
		if msgs, ok := c.index[tag]; ok {
			if statement, ok := msgs[key]; ok {
				// TODO: use type switches when we implement selecting.
				msg := string(statement.(format.String))
				return msg, true
			}
		}
		if tag == language.Und {
			break
		}
	}
	return "", false
}

func (c *Catalog) set(tag language.Tag, key string, s ...format.Statement) error {
	if len(s) != 1 {
		// TODO: handle errors properly when we process statement sequences.
		panic("statement sequence should be of length 1")
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	m := c.index[tag]
	if m == nil {
		m = map[string]format.Statement{}
		c.index[tag] = m
	}
	m[key] = s[0]
	return nil
}
