// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mvcc

import (
	"encoding/binary"
	"time"

	"go.uber.org/zap"
)

func (s *store) scheduleCompaction(compactMainRev int64, keep map[revision]struct{}) bool {
	totalStart := time.Now()
	defer func() { dbCompactionTotalMs.Observe(float64(time.Since(totalStart) / time.Millisecond)) }()
	keyCompactions := 0
	defer func() { dbCompactionKeysCounter.Add(float64(keyCompactions)) }()

	end := make([]byte, 8)
	binary.BigEndian.PutUint64(end, uint64(compactMainRev+1))

	last := make([]byte, 8+1+8)
	for {
		var rev revision

		start := time.Now()

		tx := s.b.BatchTx()
		tx.Lock()
		keys, _ := tx.UnsafeRange(keyBucketName, last, end, int64(s.cfg.CompactionBatchLimit))
		for _, key := range keys {
			rev = bytesToRev(key)
			if _, ok := keep[rev]; !ok {
				tx.UnsafeDelete(keyBucketName, key)
				keyCompactions++
			}
		}

		if len(keys) < s.cfg.CompactionBatchLimit {
			rbytes := make([]byte, 8+1+8)
			revToBytes(revision{main: compactMainRev}, rbytes)
			tx.UnsafePut(metaBucketName, finishedCompactKeyName, rbytes)
			tx.Unlock()
			if s.lg != nil {
				s.lg.Info(
					"finished scheduled compaction",
					zap.Int64("compact-revision", compactMainRev),
					zap.Duration("took", time.Since(totalStart)),
				)
			} else {
				plog.Infof("finished scheduled compaction at %d (took %v)", compactMainRev, time.Since(totalStart))
			}
			return true
		}

		// update last
		revToBytes(revision{main: rev.main, sub: rev.sub + 1}, last)
		tx.Unlock()
		// Immediately commit the compaction deletes instead of letting them accumulate in the write buffer
		s.b.ForceCommit()
		dbCompactionPauseMs.Observe(float64(time.Since(start) / time.Millisecond))

		select {
		case <-time.After(10 * time.Millisecond):
		case <-s.stopc:
			return false
		}
	}
}
