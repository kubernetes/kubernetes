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
		return nil, fmt.Errorf("%w: wireless: %w", ErrFileParse, err)
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
			return nil, fmt.Errorf("%w: expected 2 parts after splitting line by ':', got %d for line %q", ErrFileParse, len(parts), line)
		}

		name := strings.TrimSpace(parts[0])
		stats := strings.Fields(parts[1])

		if len(stats) < 10 {
			return nil, fmt.Errorf("%w: invalid number of fields in line %d, expected 10+, got %d: %q", ErrFileParse, n, len(stats), line)
		}

		status, err := strconv.ParseUint(stats[0], 16, 16)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid status in line %d: %q", ErrFileParse, n, line)
		}

		qlink, err := strconv.Atoi(strings.TrimSuffix(stats[1], "."))
		if err != nil {
			return nil, fmt.Errorf("%w: parse Quality:link as integer %q: %w", ErrFileParse, qlink, err)
		}

		qlevel, err := strconv.Atoi(strings.TrimSuffix(stats[2], "."))
		if err != nil {
			return nil, fmt.Errorf("%w: Quality:level as integer %q: %w", ErrFileParse, qlevel, err)
		}

		qnoise, err := strconv.Atoi(strings.TrimSuffix(stats[3], "."))
		if err != nil {
			return nil, fmt.Errorf("%w: Quality:noise as integer %q: %w", ErrFileParse, qnoise, err)
		}

		dnwid, err := strconv.Atoi(stats[4])
		if err != nil {
			return nil, fmt.Errorf("%w: Discarded:nwid as integer %q: %w", ErrFileParse, dnwid, err)
		}

		dcrypt, err := strconv.Atoi(stats[5])
		if err != nil {
			return nil, fmt.Errorf("%w: Discarded:crypt as integer %q: %w", ErrFileParse, dcrypt, err)
		}

		dfrag, err := strconv.Atoi(stats[6])
		if err != nil {
			return nil, fmt.Errorf("%w: Discarded:frag as integer %q: %w", ErrFileParse, dfrag, err)
		}

		dretry, err := strconv.Atoi(stats[7])
		if err != nil {
			return nil, fmt.Errorf("%w: Discarded:retry as integer %q: %w", ErrFileParse, dretry, err)
		}

		dmisc, err := strconv.Atoi(stats[8])
		if err != nil {
			return nil, fmt.Errorf("%w: Discarded:misc as integer %q: %w", ErrFileParse, dmisc, err)
		}

		mbeacon, err := strconv.Atoi(stats[9])
		if err != nil {
			return nil, fmt.Errorf("%w: Missed:beacon as integer %q: %w", ErrFileParse, mbeacon, err)
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
		return nil, fmt.Errorf("%w: Failed to scan /proc/net/wireless: %w", ErrFileRead, err)
	}

	return interfaces, nil
}
