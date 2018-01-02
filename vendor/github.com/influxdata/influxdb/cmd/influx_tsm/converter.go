package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/influxdata/influxdb/cmd/influx_tsm/stats"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

const (
	maxBlocksPerKey = 65535
)

// KeyIterator is used to iterate over b* keys for conversion to tsm keys
type KeyIterator interface {
	Next() bool
	Read() (string, []tsm1.Value, error)
}

// Converter encapsulates the logic for converting b*1 shards to tsm1 shards.
type Converter struct {
	path           string
	maxTSMFileSize uint32
	sequence       int
	stats          *stats.Stats
}

// NewConverter returns a new instance of the Converter.
func NewConverter(path string, sz uint32, stats *stats.Stats) *Converter {
	return &Converter{
		path:           path,
		maxTSMFileSize: sz,
		stats:          stats,
	}
}

// Process writes the data provided by iter to a tsm1 shard.
func (c *Converter) Process(iter KeyIterator) error {
	// Ensure the tsm1 directory exists.
	if err := os.MkdirAll(c.path, 0777); err != nil {
		return err
	}

	// Iterate until no more data remains.
	var w tsm1.TSMWriter
	var keyCount map[string]int

	for iter.Next() {
		k, v, err := iter.Read()
		if err != nil {
			return err
		}

		if w == nil {
			w, err = c.nextTSMWriter()
			if err != nil {
				return err
			}
			keyCount = map[string]int{}
		}
		if err := w.Write(k, v); err != nil {
			return err
		}
		keyCount[k]++

		c.stats.AddPointsRead(len(v))
		c.stats.AddPointsWritten(len(v))

		// If we have a max file size configured and we're over it, start a new TSM file.
		if w.Size() > c.maxTSMFileSize || keyCount[k] == maxBlocksPerKey {
			if err := w.WriteIndex(); err != nil && err != tsm1.ErrNoValues {
				return err
			}

			c.stats.AddTSMBytes(w.Size())

			if err := w.Close(); err != nil {
				return err
			}
			w = nil
		}
	}

	if w != nil {
		if err := w.WriteIndex(); err != nil && err != tsm1.ErrNoValues {
			return err
		}
		c.stats.AddTSMBytes(w.Size())

		if err := w.Close(); err != nil {
			return err
		}
	}

	return nil
}

// nextTSMWriter returns the next TSMWriter for the Converter.
func (c *Converter) nextTSMWriter() (tsm1.TSMWriter, error) {
	c.sequence++
	fileName := filepath.Join(c.path, fmt.Sprintf("%09d-%09d.%s", 1, c.sequence, tsm1.TSMFileExtension))

	fd, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return nil, err
	}

	// Create the writer for the new TSM file.
	w, err := tsm1.NewTSMWriter(fd)
	if err != nil {
		return nil, err
	}

	c.stats.IncrTSMFileCount()
	return w, nil
}
