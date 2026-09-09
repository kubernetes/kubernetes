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

//go:build !windows

package procfs

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Zoneinfo holds info parsed from /proc/zoneinfo.
type Zoneinfo struct {
	Node                       string
	Zone                       string
	NrFreePages                *int64
	Min                        *int64
	Low                        *int64
	High                       *int64
	Scanned                    *int64
	Spanned                    *int64
	Present                    *int64
	Managed                    *int64
	NrActiveAnon               *int64
	NrInactiveAnon             *int64
	NrIsolatedAnon             *int64
	NrAnonPages                *int64
	NrAnonTransparentHugepages *int64
	NrActiveFile               *int64
	NrInactiveFile             *int64
	NrIsolatedFile             *int64
	NrFilePages                *int64
	NrSlabReclaimable          *int64
	NrSlabUnreclaimable        *int64
	NrMlockStack               *int64
	NrKernelStack              *int64
	NrMapped                   *int64
	NrDirty                    *int64
	NrWriteback                *int64
	NrUnevictable              *int64
	NrShmem                    *int64
	NrDirtied                  *int64
	NrWritten                  *int64
	NumaHit                    *int64
	NumaMiss                   *int64
	NumaForeign                *int64
	NumaInterleave             *int64
	NumaLocal                  *int64
	NumaOther                  *int64
	Protection                 []*int64
}

var nodeZoneRE = regexp.MustCompile(`(\d+), zone\s+(\w+)`)

// Zoneinfo parses an zoneinfo-file (/proc/zoneinfo) and returns a slice of
// structs containing the relevant info.  More information available here:
// https://www.kernel.org/doc/Documentation/sysctl/vm.txt
func (fs FS) Zoneinfo() ([]Zoneinfo, error) {
	data, err := os.ReadFile(fs.proc.Path("zoneinfo"))
	if err != nil {
		return nil, fmt.Errorf("%w: error reading zoneinfo %q: %w", ErrFileRead, fs.proc.Path("zoneinfo"), err)
	}
	zoneinfo, err := parseZoneinfo(data)
	if err != nil {
		return nil, fmt.Errorf("%w: error parsing zoneinfo %q: %w", ErrFileParse, fs.proc.Path("zoneinfo"), err)
	}
	return zoneinfo, nil
}

func parseZoneinfo(zoneinfoData []byte) ([]Zoneinfo, error) {

	zoneinfo := []Zoneinfo{}

	for block := range bytes.SplitSeq(zoneinfoData, []byte("\nNode")) {
		var zoneinfoElement Zoneinfo
		for line := range strings.SplitSeq(string(block), "\n") {

			if nodeZone := nodeZoneRE.FindStringSubmatch(line); nodeZone != nil {
				zoneinfoElement.Node = nodeZone[1]
				zoneinfoElement.Zone = nodeZone[2]
				continue
			}
			if strings.HasPrefix(strings.TrimSpace(line), "per-node stats") {
				continue
			}
			parts := strings.Fields(strings.TrimSpace(line))
			if len(parts) < 2 {
				continue
			}
			vp := util.NewValueParser(parts[1])
			switch parts[0] {
			case "nr_free_pages":
				zoneinfoElement.NrFreePages = vp.PInt64()
			case "min":
				zoneinfoElement.Min = vp.PInt64()
			case "low":
				zoneinfoElement.Low = vp.PInt64()
			case "high":
				zoneinfoElement.High = vp.PInt64()
			case "scanned":
				zoneinfoElement.Scanned = vp.PInt64()
			case "spanned":
				zoneinfoElement.Spanned = vp.PInt64()
			case "present":
				zoneinfoElement.Present = vp.PInt64()
			case "managed":
				zoneinfoElement.Managed = vp.PInt64()
			case "nr_active_anon":
				zoneinfoElement.NrActiveAnon = vp.PInt64()
			case "nr_inactive_anon":
				zoneinfoElement.NrInactiveAnon = vp.PInt64()
			case "nr_isolated_anon":
				zoneinfoElement.NrIsolatedAnon = vp.PInt64()
			case "nr_anon_pages":
				zoneinfoElement.NrAnonPages = vp.PInt64()
			case "nr_anon_transparent_hugepages":
				zoneinfoElement.NrAnonTransparentHugepages = vp.PInt64()
			case "nr_active_file":
				zoneinfoElement.NrActiveFile = vp.PInt64()
			case "nr_inactive_file":
				zoneinfoElement.NrInactiveFile = vp.PInt64()
			case "nr_isolated_file":
				zoneinfoElement.NrIsolatedFile = vp.PInt64()
			case "nr_file_pages":
				zoneinfoElement.NrFilePages = vp.PInt64()
			case "nr_slab_reclaimable":
				zoneinfoElement.NrSlabReclaimable = vp.PInt64()
			case "nr_slab_unreclaimable":
				zoneinfoElement.NrSlabUnreclaimable = vp.PInt64()
			case "nr_mlock_stack":
				zoneinfoElement.NrMlockStack = vp.PInt64()
			case "nr_kernel_stack":
				zoneinfoElement.NrKernelStack = vp.PInt64()
			case "nr_mapped":
				zoneinfoElement.NrMapped = vp.PInt64()
			case "nr_dirty":
				zoneinfoElement.NrDirty = vp.PInt64()
			case "nr_writeback":
				zoneinfoElement.NrWriteback = vp.PInt64()
			case "nr_unevictable":
				zoneinfoElement.NrUnevictable = vp.PInt64()
			case "nr_shmem":
				zoneinfoElement.NrShmem = vp.PInt64()
			case "nr_dirtied":
				zoneinfoElement.NrDirtied = vp.PInt64()
			case "nr_written":
				zoneinfoElement.NrWritten = vp.PInt64()
			case "numa_hit":
				zoneinfoElement.NumaHit = vp.PInt64()
			case "numa_miss":
				zoneinfoElement.NumaMiss = vp.PInt64()
			case "numa_foreign":
				zoneinfoElement.NumaForeign = vp.PInt64()
			case "numa_interleave":
				zoneinfoElement.NumaInterleave = vp.PInt64()
			case "numa_local":
				zoneinfoElement.NumaLocal = vp.PInt64()
			case "numa_other":
				zoneinfoElement.NumaOther = vp.PInt64()
			case "protection:":
				protectionParts := strings.Split(line, ":")
				protectionValues := strings.Replace(protectionParts[1], "(", "", 1)
				protectionValues = strings.Replace(protectionValues, ")", "", 1)
				protectionValues = strings.TrimSpace(protectionValues)
				protectionStringMap := strings.Split(protectionValues, ", ")
				val, err := util.ParsePInt64s(protectionStringMap)
				if err == nil {
					zoneinfoElement.Protection = val
				}
			}

		}

		zoneinfo = append(zoneinfo, zoneinfoElement)
	}
	return zoneinfo, nil
}
