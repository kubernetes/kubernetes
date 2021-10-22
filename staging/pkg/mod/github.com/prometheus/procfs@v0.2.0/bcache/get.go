// Copyright 2017 The Prometheus Authors
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

package bcache

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/fs"
)

// FS represents the pseudo-filesystem proc, which provides an interface to
// kernel data structures.
type FS struct {
	sys *fs.FS
}

// NewDefaultFS returns a new Bcache using the default sys fs mount point. It will error
// if the mount point can't be read.
func NewDefaultFS() (FS, error) {
	return NewFS(fs.DefaultSysMountPoint)
}

// NewFS returns a new Bcache using the given sys fs mount point. It will error
// if the mount point can't be read.
func NewFS(mountPoint string) (FS, error) {
	if strings.TrimSpace(mountPoint) == "" {
		mountPoint = fs.DefaultSysMountPoint
	}
	fs, err := fs.NewFS(mountPoint)
	if err != nil {
		return FS{}, err
	}
	return FS{&fs}, nil
}

// Stats is a wrapper around stats()
// It returns full available statistics
func (fs FS) Stats() ([]*Stats, error) {
	return fs.stats(true)
}

// StatsWithoutPriority is a wrapper around stats().
// It ignores priority_stats file, because it is expensive to read.
func (fs FS) StatsWithoutPriority() ([]*Stats, error) {
	return fs.stats(false)
}

// stats() retrieves bcache runtime statistics for each bcache.
// priorityStats flag controls if we need to read priority_stats.
func (fs FS) stats(priorityStats bool) ([]*Stats, error) {
	matches, err := filepath.Glob(fs.sys.Path("fs/bcache/*-*"))
	if err != nil {
		return nil, err
	}

	stats := make([]*Stats, 0, len(matches))
	for _, uuidPath := range matches {
		// "*-*" in glob above indicates the name of the bcache.
		name := filepath.Base(uuidPath)

		// stats
		s, err := GetStats(uuidPath, priorityStats)
		if err != nil {
			return nil, err
		}

		s.Name = name
		stats = append(stats, s)
	}

	return stats, nil
}

// ParsePseudoFloat parses the peculiar format produced by bcache's bch_hprint.
func parsePseudoFloat(str string) (float64, error) {
	ss := strings.Split(str, ".")

	intPart, err := strconv.ParseFloat(ss[0], 64)
	if err != nil {
		return 0, err
	}

	if len(ss) == 1 {
		// Pure integers are fine.
		return intPart, nil
	}
	fracPart, err := strconv.ParseFloat(ss[1], 64)
	if err != nil {
		return 0, err
	}
	// fracPart is a number between 0 and 1023 divided by 100; it is off
	// by a small amount. Unexpected bumps in time lines may occur because
	// for bch_hprint .1 != .10 and .10 > .9 (at least up to Linux
	// v4.12-rc3).

	// Restore the proper order:
	fracPart = fracPart / 10.24
	return intPart + fracPart, nil
}

// Dehumanize converts a human-readable byte slice into a uint64.
func dehumanize(hbytes []byte) (uint64, error) {
	ll := len(hbytes)
	if ll == 0 {
		return 0, fmt.Errorf("zero-length reply")
	}
	lastByte := hbytes[ll-1]
	mul := float64(1)
	var (
		mant float64
		err  error
	)
	// If lastByte is beyond the range of ASCII digits, it must be a
	// multiplier.
	if lastByte > 57 {
		// Remove multiplier from slice.
		hbytes = hbytes[:len(hbytes)-1]

		const (
			_ = 1 << (10 * iota)
			KiB
			MiB
			GiB
			TiB
			PiB
			EiB
			ZiB
			YiB
		)

		multipliers := map[rune]float64{
			// Source for conversion rules:
			// linux-kernel/drivers/md/bcache/util.c:bch_hprint()
			'k': KiB,
			'M': MiB,
			'G': GiB,
			'T': TiB,
			'P': PiB,
			'E': EiB,
			'Z': ZiB,
			'Y': YiB,
		}
		mul = multipliers[rune(lastByte)]
		mant, err = parsePseudoFloat(string(hbytes))
		if err != nil {
			return 0, err
		}
	} else {
		// Not humanized by bch_hprint
		mant, err = strconv.ParseFloat(string(hbytes), 64)
		if err != nil {
			return 0, err
		}
	}
	res := uint64(mant * mul)
	return res, nil
}

func dehumanizeSigned(str string) (int64, error) {
	value, err := dehumanize([]byte(strings.TrimPrefix(str, "-")))
	if err != nil {
		return 0, err
	}
	if strings.HasPrefix(str, "-") {
		return int64(-value), nil
	}
	return int64(value), nil
}

type parser struct {
	uuidPath   string
	subDir     string
	currentDir string
	err        error
}

func (p *parser) setSubDir(pathElements ...string) {
	p.subDir = path.Join(pathElements...)
	p.currentDir = path.Join(p.uuidPath, p.subDir)
}

func (p *parser) readValue(fileName string) uint64 {
	if p.err != nil {
		return 0
	}
	path := path.Join(p.currentDir, fileName)
	byt, err := ioutil.ReadFile(path)
	if err != nil {
		p.err = fmt.Errorf("failed to read: %s", path)
		return 0
	}
	// Remove trailing newline.
	byt = byt[:len(byt)-1]
	res, err := dehumanize(byt)
	p.err = err
	return res
}

// ParsePriorityStats parses lines from the priority_stats file.
func parsePriorityStats(line string, ps *PriorityStats) error {
	var (
		value uint64
		err   error
	)
	switch {
	case strings.HasPrefix(line, "Unused:"):
		fields := strings.Fields(line)
		rawValue := fields[len(fields)-1]
		valueStr := strings.TrimSuffix(rawValue, "%")
		value, err = strconv.ParseUint(valueStr, 10, 64)
		if err != nil {
			return err
		}
		ps.UnusedPercent = value
	case strings.HasPrefix(line, "Metadata:"):
		fields := strings.Fields(line)
		rawValue := fields[len(fields)-1]
		valueStr := strings.TrimSuffix(rawValue, "%")
		value, err = strconv.ParseUint(valueStr, 10, 64)
		if err != nil {
			return err
		}
		ps.MetadataPercent = value
	}
	return nil
}

// ParseWritebackRateDebug parses lines from the writeback_rate_debug file.
func parseWritebackRateDebug(line string, wrd *WritebackRateDebugStats) error {
	switch {
	case strings.HasPrefix(line, "rate:"):
		fields := strings.Fields(line)
		rawValue := fields[len(fields)-1]
		valueStr := strings.TrimSuffix(rawValue, "/sec")
		value, err := dehumanize([]byte(valueStr))
		if err != nil {
			return err
		}
		wrd.Rate = value
	case strings.HasPrefix(line, "dirty:"):
		fields := strings.Fields(line)
		valueStr := fields[len(fields)-1]
		value, err := dehumanize([]byte(valueStr))
		if err != nil {
			return err
		}
		wrd.Dirty = value
	case strings.HasPrefix(line, "target:"):
		fields := strings.Fields(line)
		valueStr := fields[len(fields)-1]
		value, err := dehumanize([]byte(valueStr))
		if err != nil {
			return err
		}
		wrd.Target = value
	case strings.HasPrefix(line, "proportional:"):
		fields := strings.Fields(line)
		valueStr := fields[len(fields)-1]
		value, err := dehumanizeSigned(valueStr)
		if err != nil {
			return err
		}
		wrd.Proportional = value
	case strings.HasPrefix(line, "integral:"):
		fields := strings.Fields(line)
		valueStr := fields[len(fields)-1]
		value, err := dehumanizeSigned(valueStr)
		if err != nil {
			return err
		}
		wrd.Integral = value
	case strings.HasPrefix(line, "change:"):
		fields := strings.Fields(line)
		rawValue := fields[len(fields)-1]
		valueStr := strings.TrimSuffix(rawValue, "/sec")
		value, err := dehumanizeSigned(valueStr)
		if err != nil {
			return err
		}
		wrd.Change = value
	case strings.HasPrefix(line, "next io:"):
		fields := strings.Fields(line)
		rawValue := fields[len(fields)-1]
		valueStr := strings.TrimSuffix(rawValue, "ms")
		value, err := strconv.ParseInt(valueStr, 10, 64)
		if err != nil {
			return err
		}
		wrd.NextIO = value
	}
	return nil
}

func (p *parser) getPriorityStats() PriorityStats {
	var res PriorityStats

	if p.err != nil {
		return res
	}

	path := path.Join(p.currentDir, "priority_stats")

	file, err := os.Open(path)
	if err != nil {
		p.err = fmt.Errorf("failed to read: %s", path)
		return res
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		err = parsePriorityStats(scanner.Text(), &res)
		if err != nil {
			p.err = fmt.Errorf("failed to parse: %s (%s)", path, err)
			return res
		}
	}
	if err := scanner.Err(); err != nil {
		p.err = fmt.Errorf("failed to parse: %s (%s)", path, err)
		return res
	}
	return res
}

func (p *parser) getWritebackRateDebug() WritebackRateDebugStats {
	var res WritebackRateDebugStats

	if p.err != nil {
		return res
	}
	path := path.Join(p.currentDir, "writeback_rate_debug")
	file, err := os.Open(path)
	if err != nil {
		p.err = fmt.Errorf("failed to read: %s", path)
		return res
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		err = parseWritebackRateDebug(scanner.Text(), &res)
		if err != nil {
			p.err = fmt.Errorf("failed to parse: %s (%s)", path, err)
			return res
		}
	}
	if err := scanner.Err(); err != nil {
		p.err = fmt.Errorf("failed to parse: %s (%s)", path, err)
		return res
	}
	return res
}

// GetStats collects from sysfs files data tied to one bcache ID.
func GetStats(uuidPath string, priorityStats bool) (*Stats, error) {
	var bs Stats

	par := parser{uuidPath: uuidPath}

	// bcache stats

	// dir <uuidPath>
	par.setSubDir("")
	bs.Bcache.AverageKeySize = par.readValue("average_key_size")
	bs.Bcache.BtreeCacheSize = par.readValue("btree_cache_size")
	bs.Bcache.CacheAvailablePercent = par.readValue("cache_available_percent")
	bs.Bcache.Congested = par.readValue("congested")
	bs.Bcache.RootUsagePercent = par.readValue("root_usage_percent")
	bs.Bcache.TreeDepth = par.readValue("tree_depth")

	// bcache stats (internal)

	// dir <uuidPath>/internal
	par.setSubDir("internal")
	bs.Bcache.Internal.ActiveJournalEntries = par.readValue("active_journal_entries")
	bs.Bcache.Internal.BtreeNodes = par.readValue("btree_nodes")
	bs.Bcache.Internal.BtreeReadAverageDurationNanoSeconds = par.readValue("btree_read_average_duration_us")
	bs.Bcache.Internal.CacheReadRaces = par.readValue("cache_read_races")

	// bcache stats (period)

	// dir <uuidPath>/stats_five_minute
	par.setSubDir("stats_five_minute")
	bs.Bcache.FiveMin.Bypassed = par.readValue("bypassed")
	bs.Bcache.FiveMin.CacheHits = par.readValue("cache_hits")

	bs.Bcache.FiveMin.Bypassed = par.readValue("bypassed")
	bs.Bcache.FiveMin.CacheBypassHits = par.readValue("cache_bypass_hits")
	bs.Bcache.FiveMin.CacheBypassMisses = par.readValue("cache_bypass_misses")
	bs.Bcache.FiveMin.CacheHits = par.readValue("cache_hits")
	bs.Bcache.FiveMin.CacheMissCollisions = par.readValue("cache_miss_collisions")
	bs.Bcache.FiveMin.CacheMisses = par.readValue("cache_misses")
	bs.Bcache.FiveMin.CacheReadaheads = par.readValue("cache_readaheads")

	// dir <uuidPath>/stats_total
	par.setSubDir("stats_total")
	bs.Bcache.Total.Bypassed = par.readValue("bypassed")
	bs.Bcache.Total.CacheHits = par.readValue("cache_hits")

	bs.Bcache.Total.Bypassed = par.readValue("bypassed")
	bs.Bcache.Total.CacheBypassHits = par.readValue("cache_bypass_hits")
	bs.Bcache.Total.CacheBypassMisses = par.readValue("cache_bypass_misses")
	bs.Bcache.Total.CacheHits = par.readValue("cache_hits")
	bs.Bcache.Total.CacheMissCollisions = par.readValue("cache_miss_collisions")
	bs.Bcache.Total.CacheMisses = par.readValue("cache_misses")
	bs.Bcache.Total.CacheReadaheads = par.readValue("cache_readaheads")

	if par.err != nil {
		return nil, par.err
	}

	// bdev stats

	reg := path.Join(uuidPath, "bdev[0-9]*")
	bdevDirs, err := filepath.Glob(reg)
	if err != nil {
		return nil, err
	}

	bs.Bdevs = make([]BdevStats, len(bdevDirs))

	for ii, bdevDir := range bdevDirs {
		var bds = &bs.Bdevs[ii]

		bds.Name = filepath.Base(bdevDir)

		par.setSubDir(bds.Name)
		bds.DirtyData = par.readValue("dirty_data")

		wrd := par.getWritebackRateDebug()
		bds.WritebackRateDebug = wrd

		// dir <uuidPath>/<bds.Name>/stats_five_minute
		par.setSubDir(bds.Name, "stats_five_minute")
		bds.FiveMin.Bypassed = par.readValue("bypassed")
		bds.FiveMin.CacheBypassHits = par.readValue("cache_bypass_hits")
		bds.FiveMin.CacheBypassMisses = par.readValue("cache_bypass_misses")
		bds.FiveMin.CacheHits = par.readValue("cache_hits")
		bds.FiveMin.CacheMissCollisions = par.readValue("cache_miss_collisions")
		bds.FiveMin.CacheMisses = par.readValue("cache_misses")
		bds.FiveMin.CacheReadaheads = par.readValue("cache_readaheads")

		// dir <uuidPath>/<bds.Name>/stats_total
		par.setSubDir("stats_total")
		bds.Total.Bypassed = par.readValue("bypassed")
		bds.Total.CacheBypassHits = par.readValue("cache_bypass_hits")
		bds.Total.CacheBypassMisses = par.readValue("cache_bypass_misses")
		bds.Total.CacheHits = par.readValue("cache_hits")
		bds.Total.CacheMissCollisions = par.readValue("cache_miss_collisions")
		bds.Total.CacheMisses = par.readValue("cache_misses")
		bds.Total.CacheReadaheads = par.readValue("cache_readaheads")
	}

	if par.err != nil {
		return nil, par.err
	}

	// cache stats

	reg = path.Join(uuidPath, "cache[0-9]*")
	cacheDirs, err := filepath.Glob(reg)
	if err != nil {
		return nil, err
	}
	bs.Caches = make([]CacheStats, len(cacheDirs))

	for ii, cacheDir := range cacheDirs {
		var cs = &bs.Caches[ii]
		cs.Name = filepath.Base(cacheDir)

		// dir is <uuidPath>/<cs.Name>
		par.setSubDir(cs.Name)
		cs.IOErrors = par.readValue("io_errors")
		cs.MetadataWritten = par.readValue("metadata_written")
		cs.Written = par.readValue("written")

		if priorityStats {
			ps := par.getPriorityStats()
			cs.Priority = ps
		}
	}

	if par.err != nil {
		return nil, par.err
	}

	return &bs, nil
}
