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

package jq

import "testing"

func TestPlainText(t *testing.T) {
	text := "hello jq"
	l := lex("hello", text, "'", "'")
	item := l.nextItem()
	if item.typ != itemText {
		t.Errorf("expect to get itemText, got %v", item)
	}
	if item.val != "hello jq" {
		t.Errorf("expect to get %v, got %v", text, item.val)
	}
	item = l.nextItem()
	if item.typ != itemEOF {
		t.Errorf("expect to get itemEOF, got %v", item)
	}
}

func TestVariable(t *testing.T) {
	text := "hello '.foo'"
	l := lex("hello", text, "'", "'")
	expect := []itemType{itemText, itemLeftDelim, itemField, itemRightDelim, itemEOF}

	for i := 0; i < 5; i++ {
		item := l.nextItem()
		if item.typ != expect[i] {
			t.Logf("expect to get %v, got %v", expect[i], item)
		}
	}
}
