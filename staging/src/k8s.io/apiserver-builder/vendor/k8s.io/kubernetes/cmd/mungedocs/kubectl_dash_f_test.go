/*
Copyright 2015 The Kubernetes Authors.

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

import "testing"

func TestKubectlDashF(t *testing.T) {
	var cases = []struct {
		in string
		ok bool
	}{
		// No match
		{"", true},
		{
			"Foo\nBar\n",
			true,
		},
		{
			"Foo\nkubectl blah blech\nBar",
			true,
		},
		{
			"Foo\n```shell\nkubectl blah blech\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create blech\n```\nBar",
			true,
		},
		// Special cases
		{
			"Foo\n```\nkubectl -blah create -f -\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f-\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f FILENAME\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -fFILENAME\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f http://google.com/foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -fhttp://google.com/foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f ./foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f./foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f /foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f/foobar\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -f~/foobar\n```\nBar",
			true,
		},
		// Real checks
		{
			"Foo\n```\nkubectl -blah create -f mungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah create -fmungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah update -f mungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah update -fmungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah replace -f mungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah replace -fmungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah delete -f mungedocs.go\n```\nBar",
			true,
		},
		{
			"Foo\n```\nkubectl -blah delete -fmungedocs.go\n```\nBar",
			true,
		},
		// Failures
		{
			"Foo\n```\nkubectl -blah delete -f does_not_exist\n```\nBar",
			false,
		},
		{
			"Foo\n```\nkubectl -blah delete -fdoes_not_exist\n```\nBar",
			false,
		},
	}
	for i, c := range cases {
		repoRoot = ""
		in := getMungeLines(c.in)
		_, err := updateKubectlFileTargets("filename.md", in)
		if err != nil && c.ok {
			t.Errorf("case[%d]: expected success, got %v", i, err)
		}
		if err == nil && !c.ok {
			t.Errorf("case[%d]: unexpected success", i)
		}
	}
}
