// Copyright 2019 The Prometheus Authors
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

// +build !windows

package sysfs

import (
	"errors"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// RaplZone stores the information for one RAPL power zone
type RaplZone struct {
	Name           string // name of RAPL zone from file "name"
	Index          int    // index (different value for duplicate names)
	Path           string // filesystem path of RaplZone
	MaxMicrojoules uint64 // max RAPL microjoule value
}

// GetRaplZones returns a slice of RaplZones
// When RAPL files are not present, returns nil with error
// https://www.kernel.org/doc/Documentation/power/powercap/powercap.txt
func GetRaplZones(fs FS) ([]RaplZone, error) {
	raplDir := fs.sys.Path("class/powercap")

	files, err := ioutil.ReadDir(raplDir)
	if err != nil {
		return nil, errors.New(
			"no sysfs powercap / RAPL power metrics files found")
	}

	var zones []RaplZone

	// count name usages to avoid duplicates (label them with an index)
	countNameUsages := make(map[string]int)

	// loop through directory files searching for file "name" from subdirs
	for _, f := range files {
		nameFile := filepath.Join(raplDir, f.Name(), "/name")
		nameBytes, err := ioutil.ReadFile(nameFile)
		if err == nil {
			// add new rapl zone since name file was found
			name := strings.TrimSpace(string(nameBytes))

			// get a pair of index and final name
			index, name := getIndexAndName(countNameUsages,
				name)

			maxMicrojouleFilename := filepath.Join(raplDir, f.Name(),
				"/max_energy_range_uj")
			maxMicrojoules, err := util.ReadUintFromFile(maxMicrojouleFilename)
			if err != nil {
				return nil, err
			}

			zone := RaplZone{
				Name:           name,
				Index:          index,
				Path:           filepath.Join(raplDir, f.Name()),
				MaxMicrojoules: maxMicrojoules,
			}

			zones = append(zones, zone)

			// Store into map how many times this name has been used. There can
			// be e.g. multiple "dram" instances without any index postfix. The
			// count is then used for indexing
			countNameUsages[name] = index + 1
		}
	}

	return zones, nil
}

// GetEnergyMicrojoules returns the current microjoule value from the zone energy counter
// https://www.kernel.org/doc/Documentation/power/powercap/powercap.txt
func (rz RaplZone) GetEnergyMicrojoules() (uint64, error) {
	return util.ReadUintFromFile(filepath.Join(rz.Path, "/energy_uj"))
}

// getIndexAndName returns a pair of (index, name) for a given name and name
// counting map. Some RAPL-names have an index at the end, some have duplicates
// without an index at the end. When the index is embedded in the name, it is
// provided back as an integer, and stripped from the returned name. Usage
// count is used when the index value is absent from the name.
func getIndexAndName(countNameUsages map[string]int, name string) (int, string) {
	s := strings.Split(name, "-")
	if len(s) == 2 {
		index, err := strconv.Atoi(s[1])
		if err == nil {
			return index, s[0]
		}
	}
	// return count as the index, since name didn't have an index at the end
	return countNameUsages[name], name
}
