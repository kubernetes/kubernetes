package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/golang/snappy"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/tsdb/engine/tsm1"
)

// these consts are for the old tsm format. They can be removed once we remove
// the inspection for the original tsm1 files.
const (
	//IDsFileExtension is the extension for the file that keeps the compressed map
	// of keys to uint64 IDs.
	IDsFileExtension = "ids"

	// FieldsFileExtension is the extension for the file that stores compressed field
	// encoding data for this db
	FieldsFileExtension = "fields"

	// SeriesFileExtension is the extension for the file that stores the compressed
	// series metadata for series in this db
	SeriesFileExtension = "series"
)

type tsdmDumpOpts struct {
	dumpIndex  bool
	dumpBlocks bool
	filterKey  string
	path       string
}

type tsmIndex struct {
	series  int
	offset  int64
	minTime time.Time
	maxTime time.Time
	blocks  []*block
}

type block struct {
	id     uint64
	offset int64
}

type blockStats struct {
	min, max int
	counts   [][]int
}

func (b *blockStats) inc(typ int, enc byte) {
	for len(b.counts) <= typ {
		b.counts = append(b.counts, []int{})
	}
	for len(b.counts[typ]) <= int(enc) {
		b.counts[typ] = append(b.counts[typ], 0)
	}
	b.counts[typ][enc]++
}

func (b *blockStats) size(sz int) {
	if b.min == 0 || sz < b.min {
		b.min = sz
	}
	if b.min == 0 || sz > b.max {
		b.max = sz
	}
}

var (
	fieldType = []string{
		"timestamp", "float", "int", "bool", "string",
	}
	blockTypes = []string{
		"float64", "int64", "bool", "string",
	}
	timeEnc = []string{
		"none", "s8b", "rle",
	}
	floatEnc = []string{
		"none", "gor",
	}
	intEnc = []string{
		"none", "s8b", "rle",
	}
	boolEnc = []string{
		"none", "bp",
	}
	stringEnc = []string{
		"none", "snpy",
	}
	encDescs = [][]string{
		timeEnc, floatEnc, intEnc, boolEnc, stringEnc,
	}
)

func readFields(path string) (map[string]*tsdb.MeasurementFields, error) {
	fields := make(map[string]*tsdb.MeasurementFields)

	f, err := os.OpenFile(filepath.Join(path, FieldsFileExtension), os.O_RDONLY, 0666)
	if os.IsNotExist(err) {
		return fields, nil
	} else if err != nil {
		return nil, err
	}
	b, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	data, err := snappy.Decode(nil, b)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	return fields, nil
}

func readSeries(path string) (map[string]*tsdb.Series, error) {
	series := make(map[string]*tsdb.Series)

	f, err := os.OpenFile(filepath.Join(path, SeriesFileExtension), os.O_RDONLY, 0666)
	if os.IsNotExist(err) {
		return series, nil
	} else if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	data, err := snappy.Decode(nil, b)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(data, &series); err != nil {
		return nil, err
	}

	return series, nil
}

func readIds(path string) (map[string]uint64, error) {
	f, err := os.OpenFile(filepath.Join(path, IDsFileExtension), os.O_RDONLY, 0666)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	b, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	b, err = snappy.Decode(nil, b)
	if err != nil {
		return nil, err
	}

	ids := make(map[string]uint64)
	if b != nil {
		if err := json.Unmarshal(b, &ids); err != nil {
			return nil, err
		}
	}
	return ids, err
}
func readIndex(f *os.File) (*tsmIndex, error) {
	// Get the file size
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	// Seek to the series count
	f.Seek(-4, os.SEEK_END)
	b := make([]byte, 8)
	_, err = f.Read(b[:4])
	if err != nil {
		return nil, err
	}

	seriesCount := binary.BigEndian.Uint32(b)

	// Get the min time
	f.Seek(-20, os.SEEK_END)
	f.Read(b)
	minTime := time.Unix(0, int64(btou64(b)))

	// Get max time
	f.Seek(-12, os.SEEK_END)
	f.Read(b)
	maxTime := time.Unix(0, int64(btou64(b)))

	// Figure out where the index starts
	indexStart := stat.Size() - int64(seriesCount*12+20)

	// Seek to the start of the index
	f.Seek(indexStart, os.SEEK_SET)
	count := int(seriesCount)
	index := &tsmIndex{
		offset:  indexStart,
		minTime: minTime,
		maxTime: maxTime,
		series:  count,
	}

	if indexStart < 0 {
		return nil, fmt.Errorf("index corrupt: offset=%d", indexStart)
	}

	// Read the index entries
	for i := 0; i < count; i++ {
		f.Read(b)
		id := binary.BigEndian.Uint64(b)
		f.Read(b[:4])
		pos := binary.BigEndian.Uint32(b[:4])
		index.blocks = append(index.blocks, &block{id: id, offset: int64(pos)})
	}

	return index, nil
}

func cmdDumpTsm1(opts *tsdmDumpOpts) {
	var errors []error

	f, err := os.Open(opts.path)
	if err != nil {
		println(err.Error())
		os.Exit(1)
	}

	// Get the file size
	stat, err := f.Stat()
	if err != nil {
		println(err.Error())
		os.Exit(1)
	}

	b := make([]byte, 8)
	f.Read(b[:4])

	// Verify magic number
	if binary.BigEndian.Uint32(b[:4]) != 0x16D116D1 {
		println("Not a tsm1 file.")
		os.Exit(1)
	}

	ids, err := readIds(filepath.Dir(opts.path))
	if err != nil {
		println("Failed to read series:", err.Error())
		os.Exit(1)
	}

	invIds := map[uint64]string{}
	for k, v := range ids {
		invIds[v] = k
	}

	index, err := readIndex(f)
	if err != nil {
		println("Failed to readIndex:", err.Error())

		// Create a stubbed out index so we can still try and read the block data directly
		// w/o panicing ourselves.
		index = &tsmIndex{
			minTime: time.Unix(0, 0),
			maxTime: time.Unix(0, 0),
			offset:  stat.Size(),
		}
	}

	blockStats := &blockStats{}

	println("Summary:")
	fmt.Printf("  File: %s\n", opts.path)
	fmt.Printf("  Time Range: %s - %s\n",
		index.minTime.UTC().Format(time.RFC3339Nano),
		index.maxTime.UTC().Format(time.RFC3339Nano),
	)
	fmt.Printf("  Duration: %s ", index.maxTime.Sub(index.minTime))
	fmt.Printf("  Series: %d ", index.series)
	fmt.Printf("  File Size: %d\n", stat.Size())
	println()

	tw := tabwriter.NewWriter(os.Stdout, 8, 8, 1, '\t', 0)
	fmt.Fprintln(tw, "  "+strings.Join([]string{"Pos", "ID", "Ofs", "Key", "Field"}, "\t"))
	for i, block := range index.blocks {
		key := invIds[block.id]
		split := strings.Split(key, "#!~#")

		// We dont' know know if we have fields so use an informative default
		var measurement, field string = "UNKNOWN", "UNKNOWN"

		// We read some IDs from the ids file
		if len(invIds) > 0 {
			// Change the default to error until we know we have a valid key
			measurement = "ERR"
			field = "ERR"

			// Possible corruption? Try to read as much as we can and point to the problem.
			if key == "" {
				errors = append(errors, fmt.Errorf("index pos %d, field id: %d, missing key for id", i, block.id))
			} else if len(split) < 2 {
				errors = append(errors, fmt.Errorf("index pos %d, field id: %d, key corrupt: got '%v'", i, block.id, key))
			} else {
				measurement = split[0]
				field = split[1]
			}
		}

		if opts.filterKey != "" && !strings.Contains(key, opts.filterKey) {
			continue
		}
		fmt.Fprintln(tw, "  "+strings.Join([]string{
			strconv.FormatInt(int64(i), 10),
			strconv.FormatUint(block.id, 10),
			strconv.FormatInt(int64(block.offset), 10),
			measurement,
			field,
		}, "\t"))
	}

	if opts.dumpIndex {
		println("Index:")
		tw.Flush()
		println()
	}

	tw = tabwriter.NewWriter(os.Stdout, 8, 8, 1, '\t', 0)
	fmt.Fprintln(tw, "  "+strings.Join([]string{"Blk", "Ofs", "Len", "ID", "Type", "Min Time", "Points", "Enc [T/V]", "Len [T/V]"}, "\t"))

	// Staring at 4 because the magic number is 4 bytes
	i := int64(4)
	var blockCount, pointCount, blockSize int64
	indexSize := stat.Size() - index.offset

	// Start at the beginning and read every block
	for i < index.offset {
		f.Seek(int64(i), 0)

		f.Read(b)
		id := btou64(b)
		f.Read(b[:4])
		length := binary.BigEndian.Uint32(b[:4])
		buf := make([]byte, length)
		f.Read(buf)

		blockSize += int64(len(buf)) + 12

		startTime := time.Unix(0, int64(btou64(buf[:8])))
		blockType := buf[8]

		encoded := buf[9:]

		var v []tsm1.Value
		v, err := tsm1.DecodeBlock(buf, v)
		if err != nil {
			fmt.Printf("error: %v\n", err.Error())
			os.Exit(1)
		}

		pointCount += int64(len(v))

		// Length of the timestamp block
		tsLen, j := binary.Uvarint(encoded)

		// Unpack the timestamp bytes
		ts := encoded[int(j) : int(j)+int(tsLen)]

		// Unpack the value bytes
		values := encoded[int(j)+int(tsLen):]

		tsEncoding := timeEnc[int(ts[0]>>4)]
		vEncoding := encDescs[int(blockType+1)][values[0]>>4]

		typeDesc := blockTypes[blockType]

		blockStats.inc(0, ts[0]>>4)
		blockStats.inc(int(blockType+1), values[0]>>4)
		blockStats.size(len(buf))

		if opts.filterKey != "" && !strings.Contains(invIds[id], opts.filterKey) {
			i += (12 + int64(length))
			blockCount++
			continue
		}

		fmt.Fprintln(tw, "  "+strings.Join([]string{
			strconv.FormatInt(blockCount, 10),
			strconv.FormatInt(i, 10),
			strconv.FormatInt(int64(len(buf)), 10),
			strconv.FormatUint(id, 10),
			typeDesc,
			startTime.UTC().Format(time.RFC3339Nano),
			strconv.FormatInt(int64(len(v)), 10),
			fmt.Sprintf("%s/%s", tsEncoding, vEncoding),
			fmt.Sprintf("%d/%d", len(ts), len(values)),
		}, "\t"))

		i += (12 + int64(length))
		blockCount++
	}
	if opts.dumpBlocks {
		println("Blocks:")
		tw.Flush()
		println()
	}

	fmt.Printf("Statistics\n")
	fmt.Printf("  Blocks:\n")
	fmt.Printf("    Total: %d Size: %d Min: %d Max: %d Avg: %d\n",
		blockCount, blockSize, blockStats.min, blockStats.max, blockSize/blockCount)
	fmt.Printf("  Index:\n")
	fmt.Printf("    Total: %d Size: %d\n", len(index.blocks), indexSize)
	fmt.Printf("  Points:\n")
	fmt.Printf("    Total: %d", pointCount)
	println()

	println("  Encoding:")
	for i, counts := range blockStats.counts {
		if len(counts) == 0 {
			continue
		}
		fmt.Printf("    %s: ", strings.Title(fieldType[i]))
		for j, v := range counts {
			fmt.Printf("\t%s: %d (%d%%) ", encDescs[i][j], v, int(float64(v)/float64(blockCount)*100))
		}
		println()
	}
	fmt.Printf("  Compression:\n")
	fmt.Printf("    Per block: %0.2f bytes/point\n", float64(blockSize)/float64(pointCount))
	fmt.Printf("    Total: %0.2f bytes/point\n", float64(stat.Size())/float64(pointCount))

	if len(errors) > 0 {
		println()
		fmt.Printf("Errors (%d):\n", len(errors))
		for _, err := range errors {
			fmt.Printf("  * %v\n", err)
		}
		println()
	}
}

func cmdDumpTsm1dev(opts *tsdmDumpOpts) {
	var errors []error

	f, err := os.Open(opts.path)
	if err != nil {
		println(err.Error())
		os.Exit(1)
	}

	// Get the file size
	stat, err := f.Stat()
	if err != nil {
		println(err.Error())
		os.Exit(1)
	}
	b := make([]byte, 8)

	r, err := tsm1.NewTSMReaderWithOptions(tsm1.TSMReaderOptions{
		MMAPFile: f,
	})
	if err != nil {
		println("Error opening TSM files: ", err.Error())
	}
	defer r.Close()

	minTime, maxTime := r.TimeRange()
	keys := r.Keys()

	blockStats := &blockStats{}

	println("Summary:")
	fmt.Printf("  File: %s\n", opts.path)
	fmt.Printf("  Time Range: %s - %s\n",
		minTime.UTC().Format(time.RFC3339Nano),
		maxTime.UTC().Format(time.RFC3339Nano),
	)
	fmt.Printf("  Duration: %s ", maxTime.Sub(minTime))
	fmt.Printf("  Series: %d ", len(keys))
	fmt.Printf("  File Size: %d\n", stat.Size())
	println()

	tw := tabwriter.NewWriter(os.Stdout, 8, 8, 1, '\t', 0)
	fmt.Fprintln(tw, "  "+strings.Join([]string{"Pos", "Min Time", "Max Time", "Ofs", "Size", "Key", "Field"}, "\t"))
	var pos int
	for _, key := range keys {
		for _, e := range r.Entries(key) {
			pos++
			split := strings.Split(key, "#!~#")

			// We dont' know know if we have fields so use an informative default
			var measurement, field string = "UNKNOWN", "UNKNOWN"

			// Possible corruption? Try to read as much as we can and point to the problem.
			measurement = split[0]
			field = split[1]

			if opts.filterKey != "" && !strings.Contains(key, opts.filterKey) {
				continue
			}
			fmt.Fprintln(tw, "  "+strings.Join([]string{
				strconv.FormatInt(int64(pos), 10),
				e.MinTime.UTC().Format(time.RFC3339Nano),
				e.MaxTime.UTC().Format(time.RFC3339Nano),
				strconv.FormatInt(int64(e.Offset), 10),
				strconv.FormatInt(int64(e.Size), 10),
				measurement,
				field,
			}, "\t"))
		}
	}

	if opts.dumpIndex {
		println("Index:")
		tw.Flush()
		println()
	}

	tw = tabwriter.NewWriter(os.Stdout, 8, 8, 1, '\t', 0)
	fmt.Fprintln(tw, "  "+strings.Join([]string{"Blk", "Chk", "Ofs", "Len", "Type", "Min Time", "Points", "Enc [T/V]", "Len [T/V]"}, "\t"))

	// Starting at 5 because the magic number is 4 bytes + 1 byte version
	i := int64(5)
	var blockCount, pointCount, blockSize int64
	indexSize := r.IndexSize()

	// Start at the beginning and read every block
	for _, key := range keys {
		for _, e := range r.Entries(key) {

			f.Seek(int64(e.Offset), 0)
			f.Read(b[:4])

			chksum := btou32(b)

			buf := make([]byte, e.Size)
			f.Read(buf)

			blockSize += int64(len(buf)) + 4

			startTime := time.Unix(0, int64(btou64(buf[:8])))
			blockType := buf[0]

			encoded := buf[1:]

			var v []tsm1.Value
			v, err := tsm1.DecodeBlock(buf, v)
			if err != nil {
				fmt.Printf("error: %v\n", err.Error())
				os.Exit(1)
			}

			pointCount += int64(len(v))

			// Length of the timestamp block
			tsLen, j := binary.Uvarint(encoded)

			// Unpack the timestamp bytes
			ts := encoded[int(j) : int(j)+int(tsLen)]

			// Unpack the value bytes
			values := encoded[int(j)+int(tsLen):]

			tsEncoding := timeEnc[int(ts[0]>>4)]
			vEncoding := encDescs[int(blockType+1)][values[0]>>4]

			typeDesc := blockTypes[blockType]

			blockStats.inc(0, ts[0]>>4)
			blockStats.inc(int(blockType+1), values[0]>>4)
			blockStats.size(len(buf))

			if opts.filterKey != "" && !strings.Contains(key, opts.filterKey) {
				i += (4 + int64(e.Size))
				blockCount++
				continue
			}

			fmt.Fprintln(tw, "  "+strings.Join([]string{
				strconv.FormatInt(blockCount, 10),
				strconv.FormatUint(uint64(chksum), 10),
				strconv.FormatInt(i, 10),
				strconv.FormatInt(int64(len(buf)), 10),
				typeDesc,
				startTime.UTC().Format(time.RFC3339Nano),
				strconv.FormatInt(int64(len(v)), 10),
				fmt.Sprintf("%s/%s", tsEncoding, vEncoding),
				fmt.Sprintf("%d/%d", len(ts), len(values)),
			}, "\t"))

			i += (4 + int64(e.Size))
			blockCount++
		}
	}

	if opts.dumpBlocks {
		println("Blocks:")
		tw.Flush()
		println()
	}

	var blockSizeAvg int64
	if blockCount > 0 {
		blockSizeAvg = blockSize / blockCount
	}
	fmt.Printf("Statistics\n")
	fmt.Printf("  Blocks:\n")
	fmt.Printf("    Total: %d Size: %d Min: %d Max: %d Avg: %d\n",
		blockCount, blockSize, blockStats.min, blockStats.max, blockSizeAvg)
	fmt.Printf("  Index:\n")
	fmt.Printf("    Total: %d Size: %d\n", blockCount, indexSize)
	fmt.Printf("  Points:\n")
	fmt.Printf("    Total: %d", pointCount)
	println()

	println("  Encoding:")
	for i, counts := range blockStats.counts {
		if len(counts) == 0 {
			continue
		}
		fmt.Printf("    %s: ", strings.Title(fieldType[i]))
		for j, v := range counts {
			fmt.Printf("\t%s: %d (%d%%) ", encDescs[i][j], v, int(float64(v)/float64(blockCount)*100))
		}
		println()
	}
	fmt.Printf("  Compression:\n")
	fmt.Printf("    Per block: %0.2f bytes/point\n", float64(blockSize)/float64(pointCount))
	fmt.Printf("    Total: %0.2f bytes/point\n", float64(stat.Size())/float64(pointCount))

	if len(errors) > 0 {
		println()
		fmt.Printf("Errors (%d):\n", len(errors))
		for _, err := range errors {
			fmt.Printf("  * %v\n", err)
		}
		println()
	}
}
