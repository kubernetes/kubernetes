package bolt_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/fnv"
	"sync"
	"testing"

	"github.com/boltdb/bolt"
)

func validateBatchBench(b *testing.B, db *TestDB) {
	var rollback = errors.New("sentinel error to cause rollback")
	validate := func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("bench"))
		h := fnv.New32a()
		buf := make([]byte, 4)
		for id := uint32(0); id < 1000; id++ {
			binary.LittleEndian.PutUint32(buf, id)
			h.Reset()
			h.Write(buf[:])
			k := h.Sum(nil)
			v := bucket.Get(k)
			if v == nil {
				b.Errorf("not found id=%d key=%x", id, k)
				continue
			}
			if g, e := v, []byte("filler"); !bytes.Equal(g, e) {
				b.Errorf("bad value for id=%d key=%x: %s != %q", id, k, g, e)
			}
			if err := bucket.Delete(k); err != nil {
				return err
			}
		}
		// should be empty now
		c := bucket.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			b.Errorf("unexpected key: %x = %q", k, v)
		}
		return rollback
	}
	if err := db.Update(validate); err != nil && err != rollback {
		b.Error(err)
	}
}

func BenchmarkDBBatchAutomatic(b *testing.B) {
	db := NewTestDB()
	defer db.Close()
	db.MustCreateBucket([]byte("bench"))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for round := 0; round < 1000; round++ {
			wg.Add(1)

			go func(id uint32) {
				defer wg.Done()
				<-start

				h := fnv.New32a()
				buf := make([]byte, 4)
				binary.LittleEndian.PutUint32(buf, id)
				h.Write(buf[:])
				k := h.Sum(nil)
				insert := func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("bench"))
					return b.Put(k, []byte("filler"))
				}
				if err := db.Batch(insert); err != nil {
					b.Error(err)
					return
				}
			}(uint32(round))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}

func BenchmarkDBBatchSingle(b *testing.B) {
	db := NewTestDB()
	defer db.Close()
	db.MustCreateBucket([]byte("bench"))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for round := 0; round < 1000; round++ {
			wg.Add(1)
			go func(id uint32) {
				defer wg.Done()
				<-start

				h := fnv.New32a()
				buf := make([]byte, 4)
				binary.LittleEndian.PutUint32(buf, id)
				h.Write(buf[:])
				k := h.Sum(nil)
				insert := func(tx *bolt.Tx) error {
					b := tx.Bucket([]byte("bench"))
					return b.Put(k, []byte("filler"))
				}
				if err := db.Update(insert); err != nil {
					b.Error(err)
					return
				}
			}(uint32(round))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}

func BenchmarkDBBatchManual10x100(b *testing.B) {
	db := NewTestDB()
	defer db.Close()
	db.MustCreateBucket([]byte("bench"))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := make(chan struct{})
		var wg sync.WaitGroup

		for major := 0; major < 10; major++ {
			wg.Add(1)
			go func(id uint32) {
				defer wg.Done()
				<-start

				insert100 := func(tx *bolt.Tx) error {
					h := fnv.New32a()
					buf := make([]byte, 4)
					for minor := uint32(0); minor < 100; minor++ {
						binary.LittleEndian.PutUint32(buf, uint32(id*100+minor))
						h.Reset()
						h.Write(buf[:])
						k := h.Sum(nil)
						b := tx.Bucket([]byte("bench"))
						if err := b.Put(k, []byte("filler")); err != nil {
							return err
						}
					}
					return nil
				}
				if err := db.Update(insert100); err != nil {
					b.Fatal(err)
				}
			}(uint32(major))
		}
		close(start)
		wg.Wait()
	}

	b.StopTimer()
	validateBatchBench(b, db)
}
