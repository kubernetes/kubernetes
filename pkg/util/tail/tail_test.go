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
	"k8s.io/apimachinery/pkg/util/diff"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type ReadAtMostReturn struct {
	data                []byte
	IsOffsetGreaterZero bool
	err                 error
}

func TestReadAtMost(t *testing.T) {
	// Append sentenceStrB to the back of sentenceStrA
	// There will be three results, the file size is enough, the file size is just right, the file size is not enough
	// Return the expected size data form the end of the file
	fileDir, err := ioutil.TempDir("", "file-")
	if err != nil {
		t.Fatal(err)
	}
	sentenceStrA := "this is a sentence"
	sentenceStrB := "for test"
	defer os.RemoveAll(fileDir)
	filePath := filepath.Join(fileDir, "/testFileA")
	if err := ioutil.WriteFile(filePath, []byte(sentenceStrA+sentenceStrB), os.FileMode(0600)); err != nil {
		t.Fatalf("Failed to create file %s: %v", fileDir+"/testFileA", err)
	}
	var tests = []struct {
		name     string
		path     string
		max      int64
		expected *ReadAtMostReturn
	}{{
		name: "File size is enough",
		path: filePath,
		max:  int64(len(sentenceStrB)),
		expected: &ReadAtMostReturn{
			data:                []byte(sentenceStrB),
			IsOffsetGreaterZero: false,
			err:                 nil,
		},
	},
		{
			name: "File size is just right",
			path: filePath,
			max:  int64(len(sentenceStrA) + len(sentenceStrB)),
			expected: &ReadAtMostReturn{
				data:                []byte(sentenceStrA + sentenceStrB),
				IsOffsetGreaterZero: true,
				err:                 nil,
			},
		},
		{
			name: "File size is not enough",
			path: filePath,
			max:  int64(len(sentenceStrA) + len(sentenceStrB) + 10),
			expected: &ReadAtMostReturn{
				data:                []byte(sentenceStrA + sentenceStrB),
				IsOffsetGreaterZero: true,
				err:                 nil,
			},
		},
	}

	for _, test := range tests {
		data, IsOffsetGreaterZero, _ := ReadAtMost(test.path, test.max)
		if !reflect.DeepEqual(data, test.expected.data) {
			t.Errorf("Test case: %s failed Unexpected data: %s", test.name, diff.ObjectDiff(string(data), string(test.expected.data)))
		}
		if IsOffsetGreaterZero == test.expected.IsOffsetGreaterZero {
			t.Errorf("Test case: %s failed Unexpected OffsetState", test.name)
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
