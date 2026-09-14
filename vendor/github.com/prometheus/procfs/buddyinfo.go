// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// A BuddyInfo is the details parsed from /proc/buddyinfo.
// The data is comprised of an array of free fragments of each size.
// The sizes are 2^n*PAGE_SIZE, where n is the array index.
type BuddyInfo struct {
	Node  string
	Zone  string
	Sizes []float64
}

// BuddyInfo reads the buddyinfo statistics from the specified `proc` filesystem.
func (fs FS) BuddyInfo() ([]BuddyInfo, error) {
	file, err := os.Open(fs.proc.Path("buddyinfo"))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return parseBuddyInfo(file)
}

func parseBuddyInfo(r io.Reader) ([]BuddyInfo, error) {
	var (
		buddyInfo   = []BuddyInfo{}
		scanner     = bufio.NewScanner(r)
		bucketCount = -1
	)

	for scanner.Scan() {
		var err error
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 4 {
			return nil, fmt.Errorf("%w: Invalid number of fields, found: %v", ErrFileParse, parts)
		}

		node := strings.TrimSuffix(parts[1], ",")
		zone := strings.TrimSuffix(parts[3], ",")
		arraySize := len(parts[4:])

		if bucketCount == -1 {
			bucketCount = arraySize
		} else if bucketCount != arraySize {
			return nil, fmt.Errorf("%w: mismatch in number of buddyinfo buckets, previous count %d, new count %d", ErrFileParse, bucketCount, arraySize)
		}

		sizes := make([]float64, arraySize)
		for i := range arraySize {
			sizes[i], err = strconv.ParseFloat(parts[i+4], 64)
			if err != nil {
				return nil, fmt.Errorf("%w: Invalid valid in buddyinfo: %f: %w", ErrFileParse, sizes[i], err)
			}
		}

		buddyInfo = append(buddyInfo, BuddyInfo{node, zone, sizes})
	}

	return buddyInfo, scanner.Err()
}
