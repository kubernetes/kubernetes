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
	"bytes"
	"fmt"
	"k8s.io/klog/v2"
)

var (
	commitBytes = []byte("COMMIT")
	spaceBytes  = []byte(" ")
)

// MakeChainLine return an iptables-save/restore formatted chain line given a Chain
func MakeChainLine(chain Chain) string {
	return fmt.Sprintf(":%s - [0:0]", chain)
}

// GetChainLines parses a table's iptables-save data to find chains in the table.
// It returns a map of iptables.Chain to []byte where the []byte is the chain line
// from save (with counters etc.).
// Note that to avoid allocations memory is SHARED with save.
func GetChainLines(table Table, save []byte) map[Chain][]byte {
	chainsMap := make(map[Chain][]byte)
	tablePrefix := []byte("*" + string(table))
	readIndex := 0
	// find beginning of table
	for readIndex < len(save) {
		line, n := readLine(readIndex, save)
		readIndex = n
		if bytes.HasPrefix(line, tablePrefix) {
			break
		}
	}
	// parse table lines
	for readIndex < len(save) {
		line, n := readLine(readIndex, save)
		readIndex = n
		if len(line) == 0 {
			continue
		}
		if bytes.HasPrefix(line, commitBytes) || line[0] == '*' {
			break
		} else if line[0] == '#' {
			continue
		} else if line[0] == ':' && len(line) > 1 {
			// We assume that the <line> contains space - chain lines have 3 fields,
			// space delimited. If there is no space, this line will panic.
			spaceIndex := bytes.Index(line, spaceBytes)
			if spaceIndex == -1 {
				panic(fmt.Sprintf("Unexpected chain line in iptables-save output: %v", string(line)))
			}
			chain := Chain(line[1:spaceIndex])
			chainsMap[chain] = line
		}
	}
	return chainsMap
}

// LoadIptablesSaveDataToBuffer load iptables-save output to buffer.
// buffer can used exist, if no, will allocate  one
func LoadIPtablesSaveDatasToBuffer(iptables Interface, table Table, buffer *bytes.Buffer) ([]byte, error) {
	if buffer == nil {
		buffer = bytes.NewBuffer(nil)
	}
	buffer.Reset()
	err := iptables.SaveInto(table, buffer)
	return buffer.Bytes(), err
}

// GetExistingChains get iptables-save output so we can check for existing chains and rules.
// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
// Result may SHARE memory with contents of buffer.
func GetExistingChains(iptables Interface, table Table, buffer *bytes.Buffer) map[Chain][]byte {
	saveDatas, err := LoadIPtablesSaveDatasToBuffer(iptables, table, buffer)
	if err != nil {
		klog.ErrorS(err, "Failed to execute iptables-save, syncing all rules", "table", table)
	}
	return GetChainLines(table, saveDatas)
}

func readLine(readIndex int, byteArray []byte) ([]byte, int) {
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
				return nil, currentReadIndex + 1
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
			// Avoid unnecessary allocation.
			return byteArray[leftTrimIndex:rightTrimIndex], currentReadIndex + 1
		} else {
			// unset rightTrimIndex
			rightTrimIndex = -1
		}
	}
	return nil, currentReadIndex
}
