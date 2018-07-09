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

package iptables

import (
	"testing"
)

func TestReadLinesFromByteBuffer(t *testing.T) {
	testFn := func(byteArray []byte, expected []string) {
		index := 0
		readIndex := 0
		for ; readIndex < len(byteArray); index++ {
			line, n := readLine(readIndex, byteArray)
			readIndex = n
			if expected[index] != string(line) {
				t.Errorf("expected:%q, actual:%q", expected[index], line)
			}
		} // for
		if readIndex < len(byteArray) {
			t.Errorf("Byte buffer was only partially read. Buffer length is:%d, readIndex is:%d", len(byteArray), readIndex)
		}
		if index < len(expected) {
			t.Errorf("All expected strings were not compared. expected arr length:%d, matched count:%d", len(expected), index-1)
		}
	}

	byteArray1 := []byte("\n  Line 1  \n\n\n L ine4  \nLine 5 \n \n")
	expected1 := []string{"", "Line 1", "", "", "L ine4", "Line 5", ""}
	testFn(byteArray1, expected1)

	byteArray1 = []byte("")
	expected1 = []string{}
	testFn(byteArray1, expected1)

	byteArray1 = []byte("\n\n")
	expected1 = []string{"", ""}
	testFn(byteArray1, expected1)
}
