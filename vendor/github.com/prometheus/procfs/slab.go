// Copyright 2020 The Prometheus Authors
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
	"regexp"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

var (
	slabSpace  = regexp.MustCompile(`\s+`)
	slabVer    = regexp.MustCompile(`slabinfo -`)
	slabHeader = regexp.MustCompile(`# name`)
)

// Slab represents a slab pool in the kernel.
type Slab struct {
	Name         string
	ObjActive    int64
	ObjNum       int64
	ObjSize      int64
	ObjPerSlab   int64
	PagesPerSlab int64
	// tunables
	Limit        int64
	Batch        int64
	SharedFactor int64
	SlabActive   int64
	SlabNum      int64
	SharedAvail  int64
}

// SlabInfo represents info for all slabs.
type SlabInfo struct {
	Slabs []*Slab
}

func shouldParseSlab(line string) bool {
	if slabVer.MatchString(line) {
		return false
	}
	if slabHeader.MatchString(line) {
		return false
	}
	return true
}

// parseV21SlabEntry is used to parse a line from /proc/slabinfo version 2.1.
func parseV21SlabEntry(line string) (*Slab, error) {
	// First cleanup whitespace.
	l := slabSpace.ReplaceAllString(line, " ")
	s := strings.Split(l, " ")
	if len(s) != 16 {
		return nil, fmt.Errorf("%w: unable to parse: %q", ErrFileParse, line)
	}
	var err error
	i := &Slab{Name: s[0]}
	i.ObjActive, err = strconv.ParseInt(s[1], 10, 64)
	if err != nil {
		return nil, err
	}
	i.ObjNum, err = strconv.ParseInt(s[2], 10, 64)
	if err != nil {
		return nil, err
	}
	i.ObjSize, err = strconv.ParseInt(s[3], 10, 64)
	if err != nil {
		return nil, err
	}
	i.ObjPerSlab, err = strconv.ParseInt(s[4], 10, 64)
	if err != nil {
		return nil, err
	}
	i.PagesPerSlab, err = strconv.ParseInt(s[5], 10, 64)
	if err != nil {
		return nil, err
	}
	i.Limit, err = strconv.ParseInt(s[8], 10, 64)
	if err != nil {
		return nil, err
	}
	i.Batch, err = strconv.ParseInt(s[9], 10, 64)
	if err != nil {
		return nil, err
	}
	i.SharedFactor, err = strconv.ParseInt(s[10], 10, 64)
	if err != nil {
		return nil, err
	}
	i.SlabActive, err = strconv.ParseInt(s[13], 10, 64)
	if err != nil {
		return nil, err
	}
	i.SlabNum, err = strconv.ParseInt(s[14], 10, 64)
	if err != nil {
		return nil, err
	}
	i.SharedAvail, err = strconv.ParseInt(s[15], 10, 64)
	if err != nil {
		return nil, err
	}
	return i, nil
}

// parseSlabInfo21 is used to parse a slabinfo 2.1 file.
func parseSlabInfo21(r *bytes.Reader) (SlabInfo, error) {
	scanner := bufio.NewScanner(r)
	s := SlabInfo{Slabs: []*Slab{}}
	for scanner.Scan() {
		line := scanner.Text()
		if !shouldParseSlab(line) {
			continue
		}
		slab, err := parseV21SlabEntry(line)
		if err != nil {
			return s, err
		}
		s.Slabs = append(s.Slabs, slab)
	}
	return s, nil
}

// SlabInfo reads data from `/proc/slabinfo`.
func (fs FS) SlabInfo() (SlabInfo, error) {
	// TODO: Consider passing options to allow for parsing different
	// slabinfo versions. However, slabinfo 2.1 has been stable since
	// kernel 2.6.10 and later.
	data, err := util.ReadFileNoStat(fs.proc.Path("slabinfo"))
	if err != nil {
		return SlabInfo{}, err
	}

	return parseSlabInfo21(bytes.NewReader(data))
}
