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

import "bytes"

// Blocks of ``` need to have blank lines on both sides or they don't look
// right in HTML.
func checkPreformatted(filePath string, fileBytes []byte) ([]byte, error) {
	f := splitByPreformatted(fileBytes)
	f = append(fileBlocks{{false, []byte{}}}, f...)
	f = append(f, fileBlock{false, []byte{}})

	output := []byte(nil)
	for i := 1; i < len(f)-1; i++ {
		prev := &f[i-1]
		block := &f[i]
		next := &f[i+1]
		if !block.preformatted {
			continue
		}
		neededSuffix := []byte("\n\n")
		for !bytes.HasSuffix(prev.data, neededSuffix) {
			prev.data = append(prev.data, '\n')
		}
		for !bytes.HasSuffix(block.data, neededSuffix) {
			block.data = append(block.data, '\n')
			if bytes.HasPrefix(next.data, []byte("\n")) {
				// don't change the number of newlines unless needed.
				next.data = next.data[1:]
				if len(next.data) == 0 {
					f = append(f[:i+1], f[i+2:]...)
				}
			}
		}
	}
	for _, block := range f {
		output = append(output, block.data...)
	}
	return output, nil
}
