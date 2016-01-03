package storage

import (
	"encoding/binary"
	"time"
)

func (s *store) scheduleCompaction(compactMainRev int64, keep map[revision]struct{}) {
	defer s.wg.Done()

	totalStart := time.Now()
	defer dbCompactionTotalDurations.Observe(float64(time.Now().Sub(totalStart) / time.Millisecond))

	end := make([]byte, 8)
	binary.BigEndian.PutUint64(end, uint64(compactMainRev+1))

	batchsize := int64(10000)
	last := make([]byte, 8+1+8)
	for {
		var rev revision

		start := time.Now()
		tx := s.b.BatchTx()
		tx.Lock()

		keys, _ := tx.UnsafeRange(keyBucketName, last, end, batchsize)
		for _, key := range keys {
			rev = bytesToRev(key)
			if _, ok := keep[rev]; !ok {
				tx.UnsafeDelete(keyBucketName, key)
			}
		}

		if len(keys) < int(batchsize) {
			rbytes := make([]byte, 8+1+8)
			revToBytes(revision{main: compactMainRev}, rbytes)
			tx.UnsafePut(metaBucketName, finishedCompactKeyName, rbytes)
			tx.Unlock()
			return
		}

		// update last
		revToBytes(revision{main: rev.main, sub: rev.sub + 1}, last)
		tx.Unlock()
		dbCompactionPauseDurations.Observe(float64(time.Now().Sub(start) / time.Millisecond))

		select {
		case <-time.After(100 * time.Millisecond):
		case <-s.stopc:
			return
		}
	}
}
