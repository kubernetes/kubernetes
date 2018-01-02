// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mo

import (
	"reflect"
	"sort"
	"testing"
)

func TestFile_Data(t *testing.T) {
	f, err := LoadData(testMoFile.Data())
	if err != nil {
		t.Fatal(err)
	}
	if a, b := len(f.Messages), len(testMoFile.Messages); a != b {
		t.Logf("size not equal: expect = %d, got = %d", b, a)
	}
	for i, v := range f.Messages {
		if !reflect.DeepEqual(&v, &testMoFile.Messages[i]) {
			t.Fatalf("%d: expect = %v, got = %v", i, testMoFile.Messages[i], v)
		}
	}
}

func init() {
	sort.Sort(byMessages(testMoFile.Messages))
}

var testMoFile = &File{
	Messages: []Message{
		Message{
			MsgContext: "main.init",
			MsgId:      "Gettext in init.",
			MsgStr:     "Init函数中的Gettext.",
		},
		Message{
			MsgContext: "main.main",
			MsgId:      "Hello, world!",
			MsgStr:     "你好, 世界!",
		},
		Message{
			MsgContext: "main.func",
			MsgId:      "Gettext in func.",
			MsgStr:     "闭包函数中的Gettext.",
		},
		Message{
			MsgContext: "code.google.com/p/gettext-go/examples/hi.SayHi",
			MsgId:      "pkg hi: Hello, world!",
			MsgStr:     "来自\"Hi\"包的问候: 你好, 世界!",
		},
	},
}
