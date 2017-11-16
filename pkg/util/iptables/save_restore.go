/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"strings"
)

// MakeChainLine return an iptables-save/restore formatted chain line given a Chain
func MakeChainLine(chain Chain) string {
	return fmt.Sprintf(":%s - [0:0]", chain)
}

// GetChainLines parses a table's iptables-save data to find chains in the table.
// It returns a map of iptables.Chain to string where the string is the chain line from the save (with counters etc).
func GetChainLines(table Table, save []byte) map[Chain]string {
	chainsMap := make(map[Chain]string)
	tablePrefix := "*" + string(table)
	readIndex := 0
	// find beginning of table
	for readIndex < len(save) {
		line, n := ReadLine(readIndex, save)
		readIndex = n
		if strings.HasPrefix(line, tablePrefix) {
			break
		}
	}
	// parse table lines
	for readIndex < len(save) {
		line, n := ReadLine(readIndex, save)
		readIndex = n
		if len(line) == 0 {
			continue
		}
		if strings.HasPrefix(line, "COMMIT") || strings.HasPrefix(line, "*") {
			break
		} else if strings.HasPrefix(line, "#") {
			continue
		} else if strings.HasPrefix(line, ":") && len(line) > 1 {
			// We assume that the <line> contains space - chain lines have 3 fields,
			// space delimited. If there is no space, this line will panic.
			chain := Chain(line[1:strings.Index(line, " ")])
			chainsMap[chain] = line
		}
	}
	return chainsMap
}

func ReadLine(readIndex int, byteArray []byte) (string, int) {
	currentReadIndex := readIndex

	// consume left spaces
	for currentReadIndex < len(byteArray) {
		if byteArray[currentReadIndex] == ' ' {
			currentReadIndex++
		} else {
			break
		}
	}

	// leftTrimIndex stores the left index of the line after the line is left-trimmed
	leftTrimIndex := currentReadIndex

	// rightTrimIndex stores the right index of the line after the line is right-trimmed
	// it is set to -1 since the correct value has not yet been determined.
	rightTrimIndex := -1

	for ; currentReadIndex < len(byteArray); currentReadIndex++ {
		if byteArray[currentReadIndex] == ' ' {
			// set rightTrimIndex
			if rightTrimIndex == -1 {
				rightTrimIndex = currentReadIndex
			}
		} else if (byteArray[currentReadIndex] == '\n') || (currentReadIndex == (len(byteArray) - 1)) {
			// end of line or byte buffer is reached
			if currentReadIndex <= leftTrimIndex {
				return "", currentReadIndex + 1
			}
			// set the rightTrimIndex
			if rightTrimIndex == -1 {
				rightTrimIndex = currentReadIndex
				if currentReadIndex == (len(byteArray)-1) && (byteArray[currentReadIndex] != '\n') {
					// ensure that the last character is part of the returned string,
					// unless the last character is '\n'
					rightTrimIndex = currentReadIndex + 1
				}
			}
			return string(byteArray[leftTrimIndex:rightTrimIndex]), currentReadIndex + 1
		} else {
			// unset rightTrimIndex
			rightTrimIndex = -1
		}
	}
	return "", currentReadIndex
}
