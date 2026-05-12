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
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Swap represents an entry in /proc/swaps.
type Swap struct {
	Filename string
	Type     string
	Size     int
	Used     int
	Priority int
}

// Swaps returns a slice of all configured swap devices on the system.
func (fs FS) Swaps() ([]*Swap, error) {
	data, err := util.ReadFileNoStat(fs.proc.Path("swaps"))
	if err != nil {
		return nil, err
	}
	return parseSwaps(data)
}

func parseSwaps(info []byte) ([]*Swap, error) {
	swaps := []*Swap{}
	scanner := bufio.NewScanner(bytes.NewReader(info))
	scanner.Scan() // ignore header line
	for scanner.Scan() {
		swapString := scanner.Text()
		parsedSwap, err := parseSwapString(swapString)
		if err != nil {
			return nil, err
		}
		swaps = append(swaps, parsedSwap)
	}

	err := scanner.Err()
	return swaps, err
}

func parseSwapString(swapString string) (*Swap, error) {
	var err error

	swapFields := strings.Fields(swapString)
	swapLength := len(swapFields)
	if swapLength < 5 {
		return nil, fmt.Errorf("%w: too few fields in swap string: %s", ErrFileParse, swapString)
	}

	swap := &Swap{
		Filename: swapFields[0],
		Type:     swapFields[1],
	}

	swap.Size, err = strconv.Atoi(swapFields[2])
	if err != nil {
		return nil, fmt.Errorf("%w: invalid swap size: %s: %w", ErrFileParse, swapFields[2], err)
	}
	swap.Used, err = strconv.Atoi(swapFields[3])
	if err != nil {
		return nil, fmt.Errorf("%w: invalid swap used: %s: %w", ErrFileParse, swapFields[3], err)
	}
	swap.Priority, err = strconv.Atoi(swapFields[4])
	if err != nil {
		return nil, fmt.Errorf("%w: invalid swap priority: %s: %w", ErrFileParse, swapFields[4], err)
	}

	return swap, nil
}
