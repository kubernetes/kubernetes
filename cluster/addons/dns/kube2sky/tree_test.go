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
	"strings"
	"testing"
)

func makeRecordTree(kvs []kvPair) (*recordTree, error) {
	var records recordTree
	for _, kv := range kvs {
		if err := records.Set(kv.k, kv.v); err != nil {
			return nil, fmt.Errorf("error while setting %s to %s: %v", kv.k, kv.v, err)
		}
	}
	return &records, nil
}

type kvPair struct {
	k string
	v string
}

func newKVPairs(kv ...string) []kvPair {
	var ps []kvPair
	for i := 0; i < len(kv); i += 2 {
		ps = append(ps, kvPair{kv[i], kv[i+1]})
	}
	return ps
}

func TestRecordTree(t *testing.T) {
	cases := []struct {
		kv  []kvPair
		err bool
		msg string
	}{
		{newKVPairs("/a", "`", "/b", "1", "/c", "2"), false, ""},
		{newKVPairs("/a/b", "2", "/a/c", "][p"), false, ""},
		{newKVPairs("/a/b/c", "3512123491234", "/a/r", "2"), false, ""},
		{newKVPairs("/skydns/a", "x", "/skydns/b/c/d", "y", "/skydns/b/c/e", "z", "/skydns/b/c/f", "2"), false, ""},
		{newKVPairs("/a/b", "2", "/a/b", "3"), false, ""},
		{newKVPairs("/a/", "2"), true, "must not end in \"\", \"\" is not a valid filename"},
		{newKVPairs("/skydns/a", "3", "/skydns/b", "$", "/skydns/a/f", "1"), true, "cannot turn a file into a directory"},
		{newKVPairs("/skydns/a/c", "3", "/skydns/b", "$", "/skydns/a", "1"), true, "cannot turn a directory into a file"},
	}
	for i, test := range cases {
		_, err := makeRecordTree(test.kv)
		if test.err {
			if err == nil {
				t.Errorf("case %d: expected error, reason: %s", i, test.msg)
			}
		} else if err != nil {
			t.Errorf("case %d: unexpected error %v", i, err)
		}
	}
}

func addMissingKV(kvs []kvPair) []kvPair {
	newMap := make(map[string]string)
	for _, kv := range kvs {
		// Paths must start with /, and we don't want / in the pairs since etcd treats
		// / as read-only (i.e. we cannot modify it so it shouldn't be in any
		// flatRecords or here).
		split := strings.Split(kv.k, "/")
		for i := 2; i < len(split); i++ {
			newMap["/"+pathutil.Join(split[:i]...)] = ""
		}
		newMap[kv.k] = kv.v
	}
	var ps []kvPair
	for k, v := range newMap {
		ps = append(ps, kvPair{k, v})
	}
	return ps
}

func (rs flatRecords) sameAsKV(kvs []kvPair) error {
	if len(rs) != len(kvs) {
		return fmt.Errorf("different lengths")
	}
	for _, r := range rs {
		var found bool
		for _, kv := range kvs {
			if r.key == kv.k {
				if r.val == nil {
					if kv.v != "" {
						return fmt.Errorf("%s should not be a directory", r.key)
					}
				} else if *r.val != kv.v {
					return fmt.Errorf("%s expected: %s got: %s", r.key, kv.v, *r.val)
				}
				found = true
			}
		}
		if !found {
			return fmt.Errorf("non-matching records, expected: %v, got: %v", kvs, rs)
		}
	}
	return nil
}

func TestRecordTreeFlatten(t *testing.T) {
	cases := [][]kvPair{
		newKVPairs("/skydns/a", "x", "/skydns/b/c/d", "y", "/skydns/b/c/e", "z", "/skydns/b/c/f", "2"),
	}
	for i, test := range cases {
		records, err := makeRecordTree(test)
		if err != nil {
			t.Errorf("case %d: unexpected error making tree %v", i, err)
			continue
		}
		flatRec := records.flatten()
		err = flatRec.sameAsKV(addMissingKV(test))
		if err != nil {
			t.Errorf("case %d, flattened records not equivalent to original: %v", i, err)
		}
	}
}

func TestRecordDiff(t *testing.T) {
	// In the following tests, removals are slightly different from creations because
	// while directories are implicitly created, they are explicitly removed.
	cases := []struct {
		old    []kvPair
		new    []kvPair
		setLen int
		rmLen  int
		desc   string
	}{
		{newKVPairs("/y/a", "x"), newKVPairs(), 0, 2, "completely remove of a file and dir"},
		{newKVPairs(), newKVPairs("/x/b", "y"), 1, 0, "create a new file and dir"},
		{newKVPairs("/y/a", "x"), newKVPairs("/x/b", "y"), 1, 2, "both removal and create a new file and dir"},
		{newKVPairs("/a", "1"), newKVPairs("/a", "2"), 1, 0, "updating a top-level file"},
		{newKVPairs("/a/b", "2"), newKVPairs("/a", "0", "/b", "2"), 2, 2, "change a dir to a file and move its contents"},
		{newKVPairs("/a", "."), newKVPairs("/a", ".", "/b", "*"), 1, 0, "add a second top-level file"},
		{newKVPairs("/8", "@", "/5", "1"), newKVPairs("/5", "1"), 0, 1, "remove one top-level file"},
	}
	for i, test := range cases {
		t.Logf("case %d: %s", i, test.desc)
		oldRec, err := makeRecordTree(test.old)
		if err != nil {
			t.Errorf("case %d: unexpected error making old tree %v", i, err)
		}
		newRec, err := makeRecordTree(test.new)
		if err != nil {
			t.Errorf("case %d: unexpected error making new tree %v", i, err)
		}
		toSet, toRemove := recordDiff(oldRec, newRec)
		if len(toSet) != test.setLen {
			t.Errorf("case %d: expected to set %d paths, instead setting %v", i, test.setLen, toSet)
		}
		if len(toRemove) != test.rmLen {
			t.Errorf("case %d: expected to remove %d paths, instead removing %v", i, test.rmLen, toRemove)
		}
	}
}
