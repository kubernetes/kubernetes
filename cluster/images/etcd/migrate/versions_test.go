/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"github.com/blang/semver/v4"
)

func TestSerializeEtcdVersionPair(t *testing.T) {
	cases := []struct {
		versionTxt string
		version    *EtcdVersionPair
		match      bool
	}{
		{"3.1.2/etcd3", &EtcdVersionPair{&EtcdVersion{semver.MustParse("3.1.2")}, storageEtcd3}, true},
		{"1.1.1-rc.0/etcd3", &EtcdVersionPair{&EtcdVersion{semver.MustParse("1.1.1-rc.0")}, storageEtcd3}, true},
		{"10.100.1000/etcd3", &EtcdVersionPair{&EtcdVersion{semver.MustParse("10.100.1000")}, storageEtcd3}, true},
	}

	for _, c := range cases {
		vp, err := ParseEtcdVersionPair(c.versionTxt)
		if err != nil {
			t.Errorf("Failed to parse '%s': %v", c.versionTxt, err)
		}
		if vp.Equals(c.version) != c.match {
			t.Errorf("Expected '%s' to be parsed as '%+v', got '%+v'", c.versionTxt, c.version, vp)
		}
		if vp.String() != c.versionTxt {
			t.Errorf("Expected round trip serialization back to '%s', got '%s'", c.versionTxt, vp.String())
		}
	}

	unparsables := []string{
		"1.1/etcd3",
		"1.1.1.1/etcd3",
		"1.1.1/etcd4",
	}
	for _, unparsable := range unparsables {
		vp, err := ParseEtcdVersionPair(unparsable)
		if err == nil {
			t.Errorf("Should have failed to parse '%s' but got '%s'", unparsable, vp)
		}
	}
}

func TestMajorMinorEquals(t *testing.T) {
	cases := []struct {
		first  *EtcdVersion
		second *EtcdVersion
		match  bool
	}{
		{&EtcdVersion{semver.Version{Major: 3, Minor: 1, Patch: 2}}, &EtcdVersion{semver.Version{Major: 3, Minor: 1, Patch: 0}}, true},
		{&EtcdVersion{semver.Version{Major: 3, Minor: 1, Patch: 2}}, &EtcdVersion{semver.Version{Major: 3, Minor: 1, Patch: 2}}, true},

		{&EtcdVersion{semver.Version{Major: 3, Minor: 0, Patch: 0}}, &EtcdVersion{semver.Version{Major: 3, Minor: 1, Patch: 0}}, false},
		{&EtcdVersion{semver.Version{Major: 2, Minor: 0, Patch: 0}}, &EtcdVersion{semver.Version{Major: 3, Minor: 0, Patch: 0}}, false},
	}

	for _, c := range cases {
		if c.first.MajorMinorEquals(c.second) != c.match {
			t.Errorf("Expected (%+v == %+v) == %t, got %t", c.first, c.second, c.match, !c.match)
		}
	}
}
