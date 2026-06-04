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
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

const (
	blackholeRepresentation string = "*"
	blackholeIfaceName      string = "blackhole"
	routeLineColumns        int    = 11
)

// A NetRouteLine represents one line from net/route.
type NetRouteLine struct {
	Iface       string
	Destination uint32
	Gateway     uint32
	Flags       uint32
	RefCnt      uint32
	Use         uint32
	Metric      uint32
	Mask        uint32
	MTU         uint32
	Window      uint32
	IRTT        uint32
}

func (fs FS) NetRoute() ([]NetRouteLine, error) {
	return readNetRoute(fs.proc.Path("net", "route"))
}

func readNetRoute(path string) ([]NetRouteLine, error) {
	b, err := util.ReadFileNoStat(path)
	if err != nil {
		return nil, err
	}

	routelines, err := parseNetRoute(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("failed to read net route from %s: %w", path, err)
	}
	return routelines, nil
}

func parseNetRoute(r io.Reader) ([]NetRouteLine, error) {
	var routelines []NetRouteLine

	scanner := bufio.NewScanner(r)
	scanner.Scan()
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		routeline, err := parseNetRouteLine(fields)
		if err != nil {
			return nil, err
		}
		routelines = append(routelines, *routeline)
	}
	return routelines, nil
}

func parseNetRouteLine(fields []string) (*NetRouteLine, error) {
	if len(fields) != routeLineColumns {
		return nil, fmt.Errorf("invalid routeline, num of digits: %d", len(fields))
	}
	iface := fields[0]
	if iface == blackholeRepresentation {
		iface = blackholeIfaceName
	}
	destination, err := strconv.ParseUint(fields[1], 16, 32)
	if err != nil {
		return nil, err
	}
	gateway, err := strconv.ParseUint(fields[2], 16, 32)
	if err != nil {
		return nil, err
	}
	flags, err := strconv.ParseUint(fields[3], 10, 32)
	if err != nil {
		return nil, err
	}
	refcnt, err := strconv.ParseUint(fields[4], 10, 32)
	if err != nil {
		return nil, err
	}
	use, err := strconv.ParseUint(fields[5], 10, 32)
	if err != nil {
		return nil, err
	}
	metric, err := strconv.ParseUint(fields[6], 10, 32)
	if err != nil {
		return nil, err
	}
	mask, err := strconv.ParseUint(fields[7], 16, 32)
	if err != nil {
		return nil, err
	}
	mtu, err := strconv.ParseUint(fields[8], 10, 32)
	if err != nil {
		return nil, err
	}
	window, err := strconv.ParseUint(fields[9], 10, 32)
	if err != nil {
		return nil, err
	}
	irtt, err := strconv.ParseUint(fields[10], 10, 32)
	if err != nil {
		return nil, err
	}
	routeline := &NetRouteLine{
		Iface:       iface,
		Destination: uint32(destination),
		Gateway:     uint32(gateway),
		Flags:       uint32(flags),
		RefCnt:      uint32(refcnt),
		Use:         uint32(use),
		Metric:      uint32(metric),
		Mask:        uint32(mask),
		MTU:         uint32(mtu),
		Window:      uint32(window),
		IRTT:        uint32(irtt),
	}
	return routeline, nil
}
