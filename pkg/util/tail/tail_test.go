/*
Copyright 2016 The Kubernetes Authors.

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

package tail

import (
	"bytes"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

func TestReadAtMost(t *testing.T) {
	fakeFile, _ := ioutil.TempFile("", "")
	defer os.Remove(fakeFile.Name())
	fakeData := []byte("this is fake data")

	if err := ioutil.WriteFile(fakeFile.Name(), fakeData, 0600); err != nil {
		t.Fatalf("WriteFile %s: %v", fakeFile.Name(), err)
	}

	var readTests = []struct {
		readLen    int
		out        []byte
		moreUnread bool
	}{
		{5, []byte(" data"), true},
		{len(fakeData), fakeData, false},
		{len(fakeData) + 1, fakeData, false},
	}

	for c, tt := range readTests {
		t.Logf("TestCase #%d: %+v", c, tt)
		s, moreUnread, err := ReadAtMost(fakeFile.Name(), int64(tt.readLen))

		if err != nil {
			t.Errorf("#%d: ReadAtMost error: %v", c, err)
		}
		if bytes.Compare(s, tt.out) != 0 {
			t.Errorf("#%d: expected data %s got %s", tt.out, s)
		}
		if moreUnread != tt.moreUnread {
			t.Errorf("#%d: expected more unread %t got %t", moreUnread, tt.moreUnread)
		}
	}
}

func TestTail(t *testing.T) {
	line := strings.Repeat("a", blockSize)
	testBytes := []byte(line + "\n" +
		line + "\n" +
		line + "\n" +
		line + "\n" +
		line[blockSize/2:]) // incomplete line

	for c, test := range []struct {
		n     int64
		start int64
	}{
		{n: -1, start: 0},
		{n: 0, start: int64(len(line)+1) * 4},
		{n: 1, start: int64(len(line)+1) * 3},
		{n: 9999, start: 0},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		r := bytes.NewReader(testBytes)
		s, err := FindTailLineStartIndex(r, test.n)
		if err != nil {
			t.Error(err)
		}
		if s != test.start {
			t.Errorf("%d != %d", s, test.start)
		}
	}
}
