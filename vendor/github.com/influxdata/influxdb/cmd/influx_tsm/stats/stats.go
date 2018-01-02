package stats

import (
	"sync/atomic"
	"time"
)

// Stats are the statistics captured while converting non-TSM shards to TSM
type Stats struct {
	NanFiltered     uint64
	InfFiltered     uint64
	FieldsFiltered  uint64
	PointsWritten   uint64
	PointsRead      uint64
	TsmFilesCreated uint64
	TsmBytesWritten uint64
	CompletedShards uint64
	TotalTime       time.Duration
}

// AddPointsRead increments the number of read points.
func (s *Stats) AddPointsRead(n int) {
	atomic.AddUint64(&s.PointsRead, uint64(n))
}

// AddPointsWritten increments the number of written points.
func (s *Stats) AddPointsWritten(n int) {
	atomic.AddUint64(&s.PointsWritten, uint64(n))
}

// AddTSMBytes increments the number of TSM Bytes.
func (s *Stats) AddTSMBytes(n uint32) {
	atomic.AddUint64(&s.TsmBytesWritten, uint64(n))
}

// IncrTSMFileCount increments the number of TSM files created.
func (s *Stats) IncrTSMFileCount() {
	atomic.AddUint64(&s.TsmFilesCreated, 1)
}

// IncrNaN increments the number of NaNs filtered.
func (s *Stats) IncrNaN() {
	atomic.AddUint64(&s.NanFiltered, 1)
}

// IncrInf increments the number of Infs filtered.
func (s *Stats) IncrInf() {
	atomic.AddUint64(&s.InfFiltered, 1)
}

// IncrFiltered increments the number of fields filtered.
func (s *Stats) IncrFiltered() {
	atomic.AddUint64(&s.FieldsFiltered, 1)
}
