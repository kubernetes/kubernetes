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

import "fmt"

// Blocks of ``` need to have blank lines on both sides or they don't look
// right in HTML.
func updatePreformatted(filePath string, mlines mungeLines) (mungeLines, error) {
	var out mungeLines
	inpreformat := false
	for i, mline := range mlines {
		if !inpreformat && mline.preformatted {
			if i == 0 || out[len(out)-1].data != "" {
				out = append(out, blankMungeLine)
			}
			// start of a preformat block
			inpreformat = true
		}
		out = append(out, mline)
		if inpreformat && !mline.preformatted {
			if i >= len(mlines)-2 || mlines[i+1].data != "" {
				out = append(out, blankMungeLine)
			}
			inpreformat = false
		}
	}
	return out, nil
}

// If the file ends on a preformatted line, there must have been an imbalance.
func checkPreformatBalance(filePath string, mlines mungeLines) (mungeLines, error) {
	if len(mlines) > 0 && mlines[len(mlines)-1].preformatted {
		return mlines, fmt.Errorf("unbalanced triple backtick delimiters")
	}
	return mlines, nil
}
