// Copyright 2023 The Prometheus Authors
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

// Wireless models the content of /proc/net/wireless.
type Wireless struct {
	Name string

	// Status is the current 4-digit hex value status of the interface.
	Status uint64

	// QualityLink is the link quality.
	QualityLink int

	// QualityLevel is the signal gain (dBm).
	QualityLevel int

	// QualityNoise is the signal noise baseline (dBm).
	QualityNoise int

	// DiscardedNwid is the number of discarded packets with wrong nwid/essid.
	DiscardedNwid int

	// DiscardedCrypt is the number of discarded packets with wrong code/decode (WEP).
	DiscardedCrypt int

	// DiscardedFrag is the number of discarded packets that can't perform MAC reassembly.
	DiscardedFrag int

	// DiscardedRetry is the number of discarded packets that reached max MAC retries.
	DiscardedRetry int

	// DiscardedMisc is the number of discarded packets for other reasons.
	DiscardedMisc int

	// MissedBeacon is the number of missed beacons/superframe.
	MissedBeacon int
}

// Wireless returns kernel wireless statistics.
func (fs FS) Wireless() ([]*Wireless, error) {
	b, err := util.ReadFileNoStat(fs.proc.Path("net/wireless"))
	if err != nil {
		return nil, err
	}

	m, err := parseWireless(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("failed to parse wireless: %w", err)
	}

	return m, nil
}

// parseWireless parses the contents of /proc/net/wireless.
/*
Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
 eth1: 0000    5.  -256.  -10.       0      1      0     3      0        0
 eth2: 0000    5.  -256.  -20.       0      2      0     4      0        0
*/
func parseWireless(r io.Reader) ([]*Wireless, error) {
	var (
		interfaces []*Wireless
		scanner    = bufio.NewScanner(r)
	)

	for n := 0; scanner.Scan(); n++ {
		// Skip the 2 header lines.
		if n < 2 {
			continue
		}

		line := scanner.Text()

		parts := strings.Split(line, ":")
		if len(parts) != 2 {
			return nil, fmt.Errorf("expected 2 parts after splitting line by ':', got %d for line %q", len(parts), line)
		}

		name := strings.TrimSpace(parts[0])
		stats := strings.Fields(parts[1])

		if len(stats) < 10 {
			return nil, fmt.Errorf("invalid number of fields in line %d, expected at least 10, got %d: %q", n, len(stats), line)
		}

		status, err := strconv.ParseUint(stats[0], 16, 16)
		if err != nil {
			return nil, fmt.Errorf("invalid status in line %d: %q", n, line)
		}

		qlink, err := strconv.Atoi(strings.TrimSuffix(stats[1], "."))
		if err != nil {
			return nil, fmt.Errorf("failed to parse Quality:link as integer %q: %w", qlink, err)
		}

		qlevel, err := strconv.Atoi(strings.TrimSuffix(stats[2], "."))
		if err != nil {
			return nil, fmt.Errorf("failed to parse Quality:level as integer %q: %w", qlevel, err)
		}

		qnoise, err := strconv.Atoi(strings.TrimSuffix(stats[3], "."))
		if err != nil {
			return nil, fmt.Errorf("failed to parse Quality:noise as integer %q: %w", qnoise, err)
		}

		dnwid, err := strconv.Atoi(stats[4])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Discarded:nwid as integer %q: %w", dnwid, err)
		}

		dcrypt, err := strconv.Atoi(stats[5])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Discarded:crypt as integer %q: %w", dcrypt, err)
		}

		dfrag, err := strconv.Atoi(stats[6])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Discarded:frag as integer %q: %w", dfrag, err)
		}

		dretry, err := strconv.Atoi(stats[7])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Discarded:retry as integer %q: %w", dretry, err)
		}

		dmisc, err := strconv.Atoi(stats[8])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Discarded:misc as integer %q: %w", dmisc, err)
		}

		mbeacon, err := strconv.Atoi(stats[9])
		if err != nil {
			return nil, fmt.Errorf("failed to parse Missed:beacon as integer %q: %w", mbeacon, err)
		}

		w := &Wireless{
			Name:           name,
			Status:         status,
			QualityLink:    qlink,
			QualityLevel:   qlevel,
			QualityNoise:   qnoise,
			DiscardedNwid:  dnwid,
			DiscardedCrypt: dcrypt,
			DiscardedFrag:  dfrag,
			DiscardedRetry: dretry,
			DiscardedMisc:  dmisc,
			MissedBeacon:   mbeacon,
		}

		interfaces = append(interfaces, w)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan /proc/net/wireless: %w", err)
	}

	return interfaces, nil
}
