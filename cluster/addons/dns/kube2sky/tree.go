/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"fmt"
	pathutil "path"
	"sort"
	"strings"
)

type directory struct {
	dirs  map[string]*directory
	files map[string]string
}

func newDirectory() *directory {
	return &directory{
		dirs:  make(map[string]*directory),
		files: make(map[string]string),
	}
}

type (
	isDirErr  string
	isFileErr string
)

func (e isDirErr) Error() string  { return string(e) }
func (e isFileErr) Error() string { return string(e) }

func newIsDirErr(name string) isDirErr {
	return isDirErr(fmt.Sprintf("%s is a directory", name))
}

func newIsFileErr(name string) isFileErr {
	return isFileErr(fmt.Sprintf("%s is a file", name))
}

func (d *directory) set(path []string, data string) error {
	switch len(path) {
	case 1:
		return newIsDirErr(path[0])
	case 2:
		if _, ok := d.dirs[path[1]]; ok {
			return newIsDirErr(path[1])
		}
		d.files[path[1]] = data
		return nil
	}
	if _, ok := d.files[path[1]]; ok {
		return newIsFileErr(path[1])
	}
	child, ok := d.dirs[path[1]]
	if !ok {
		child = newDirectory()
		d.dirs[path[1]] = child
	}
	return child.set(path[1:], data)
}

func (d *directory) walk(c chan<- flatRecord, path string) {
	if d == nil {
		return
	}
	// root directory is read-only in etcd, so don't send it as
	// it's not something that we can modify.
	if path != "/" {
		c <- flatRecord{path, nil}
	}
	for name := range d.files {
		s := d.files[name]
		c <- flatRecord{pathutil.Join(path, name), &s}
	}
	for name, child := range d.dirs {
		child.walk(c, pathutil.Join(path, name))
	}
}

type recordTree struct {
	head *directory
}

// Set sets the value of the key to be val. Valid keys start with "/".
func (tr *recordTree) Set(key string, val string) error {
	if key[len(key)-1] == '/' {
		return fmt.Errorf("invalid key %s, keys must not end with a /")
	}
	if tr.head == nil {
		tr.head = newDirectory()
	}

	return tr.head.set(strings.Split(key, "/"), val)
}

func (tr *recordTree) walk(c chan<- flatRecord) {
	tr.head.walk(c, "/")
	close(c)
}

func (tr *recordTree) flatten() flatRecords {
	var flat []flatRecord

	c := make(chan flatRecord)
	go tr.walk(c)

	// make sure to drain channel
	for r := range c {
		flat = append(flat, r)
	}
	return flat
}

type flatRecord struct {
	key string
	val *string
}

func (r flatRecord) IsDir() bool {
	return r.val == nil
}

func (r flatRecord) String() string {
	if r.IsDir() {
		return fmt.Sprintf("(D %q)", r.key)
	}
	return fmt.Sprintf("(F %q:%q)", r.key, *r.val)
}

type byKey []flatRecord

func (rs byKey) Len() int           { return len(rs) }
func (rs byKey) Less(i, j int) bool { return rs[i].key < rs[j].key }
func (rs byKey) Swap(i, j int)      { rs[i], rs[j] = rs[j], rs[i] }

type flatRecords []flatRecord

// SetEtcd copies the values of the records to etcd. This does not handle any file to
// directory conversions or vice-versa.
func (rs flatRecords) SetEtcd(c etcdClient) error {
	for _, r := range rs {
		if r.IsDir() {
			continue
		}
		if _, err := c.Set(r.key, *r.val, 0); err != nil {
			return err
		}
	}
	return nil
}

// RemoveEtcd removes the records (both files and directories) from etcd. It assumes
// that the records have been sorted such that files come before the directories that
// contain them.
func (rs flatRecords) RemoveEtcd(c etcdClient) error {
	for _, r := range rs {
		if r.IsDir() {
			if _, err := c.Delete(r.key, true); err != nil {
				return err
			}
		} else {
			if _, err := c.Delete(r.key, false); err != nil {
				return err
			}
		}
	}
	return nil
}

// recordDiff creates two slices of flatRecords to move an etcd client from have
// the old records to the new ones. toRemove will contain the records that should be
// removed before the records in toSet should be set. Make sure to remove the records
// using toRemove before setting them.
func recordDiff(old, new *recordTree) (toSet, toRemove flatRecords) {
	oldRs := old.flatten()
	newRs := new.flatten()

	sort.Sort(sort.Reverse(byKey(oldRs)))
	sort.Sort(sort.Reverse(byKey(newRs)))

	i, j := 0, 0
	for i < len(oldRs) && j < len(newRs) {
		if oldRs[i].key == newRs[j].key {
			if !(oldRs[i].IsDir() && newRs[j].IsDir()) {
				if oldRs[i].IsDir() || newRs[j].IsDir() {
					toRemove = append(toRemove, oldRs[i])
					toSet = append(toSet, newRs[j])
				} else if *oldRs[i].val != *newRs[j].val {
					toSet = append(toSet, newRs[j])
				}
			}
			i++
			j++
		} else if oldRs[i].key > newRs[j].key {
			toRemove = append(toRemove, oldRs[i])
			i++
		} else {
			if !newRs[j].IsDir() {
				toSet = append(toSet, newRs[j])
			}
			j++
		}
	}
	for ; i < len(oldRs); i++ {
		toRemove = append(toRemove, oldRs[i])
	}
	for ; j < len(newRs); j++ {
		if !newRs[j].IsDir() {
			toSet = append(toSet, newRs[j])
		}
	}
	return toSet, toRemove
}
