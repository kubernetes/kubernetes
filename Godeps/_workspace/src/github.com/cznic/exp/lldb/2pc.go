// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Two Phase Commit & Structural ACID

package lldb

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/cznic/fileutil"
	"github.com/cznic/mathutil"
)

var _ Filer = &ACIDFiler0{} // Ensure ACIDFiler0 is a Filer

type acidWrite struct {
	b   []byte
	off int64
}

type acidWriter0 ACIDFiler0

func (a *acidWriter0) WriteAt(b []byte, off int64) (n int, err error) {
	f := (*ACIDFiler0)(a)
	if f.bwal == nil { // new epoch
		f.data = f.data[:0]
		f.bwal = bufio.NewWriter(f.wal)
		if err = a.writePacket([]interface{}{wpt00Header, walTypeACIDFiler0, ""}); err != nil {
			return
		}
	}

	if err = a.writePacket([]interface{}{wpt00WriteData, b, off}); err != nil {
		return
	}

	f.data = append(f.data, acidWrite{b, off})
	return len(b), nil
}

func (a *acidWriter0) writePacket(items []interface{}) (err error) {
	f := (*ACIDFiler0)(a)
	b, err := EncodeScalars(items...)
	if err != nil {
		return
	}

	var b4 [4]byte
	binary.BigEndian.PutUint32(b4[:], uint32(len(b)))
	if _, err = f.bwal.Write(b4[:]); err != nil {
		return
	}

	if _, err = f.bwal.Write(b); err != nil {
		return
	}

	if m := (4 + len(b)) % 16; m != 0 {
		var pad [15]byte
		_, err = f.bwal.Write(pad[:16-m])
	}
	return
}

// WAL Packet Tags
const (
	wpt00Header = iota
	wpt00WriteData
	wpt00Checkpoint
)

const (
	walTypeACIDFiler0 = iota
)

// ACIDFiler0 is a very simple, synchronous implementation of 2PC. It uses a
// single write ahead log file to provide the structural atomicity
// (BeginUpdate/EndUpdate/Rollback) and durability (DB can be recovered from
// WAL if a crash occurred).
//
// ACIDFiler0 is a Filer.
//
// NOTE: Durable synchronous 2PC involves three fsyncs in this implementation
// (WAL, DB, zero truncated WAL).  Where possible, it's recommended to collect
// transactions for, say one second before performing the two phase commit as
// the typical performance for rotational hard disks is about few tens of
// fsyncs per second atmost. For an example of such collective transaction
// approach please see the colecting FSM STT in Dbm's documentation[1].
//
//  [1]: http://godoc.org/github.com/cznic/exp/dbm
type ACIDFiler0 struct {
	*RollbackFiler
	wal               *os.File
	bwal              *bufio.Writer
	data              []acidWrite
	testHook          bool  // keeps WAL untruncated (once)
	peakWal           int64 // tracks WAL maximum used size
	peakBitFilerPages int   // track maximum transaction memory
}

// NewACIDFiler0 returns a  newly created ACIDFiler0 with WAL in wal.
//
// If the WAL is zero sized then a previous clean shutdown of db is taken for
// granted and no recovery procedure is taken.
//
// If the WAL is of non zero size then it is checked for having a
// commited/fully finished transaction not yet been reflected in db. If such
// transaction exists it's committed to db. If the recovery process finishes
// successfully, the WAL is truncated to zero size and fsync'ed prior to return
// from NewACIDFiler0.
func NewACIDFiler(db Filer, wal *os.File) (r *ACIDFiler0, err error) {
	fi, err := wal.Stat()
	if err != nil {
		return
	}

	r = &ACIDFiler0{wal: wal}

	if fi.Size() != 0 {
		if err = r.recoverDb(db); err != nil {
			return
		}
	}

	acidWriter := (*acidWriter0)(r)

	if r.RollbackFiler, err = NewRollbackFiler(
		db,
		func(sz int64) (err error) {
			// Checkpoint
			if err = acidWriter.writePacket([]interface{}{wpt00Checkpoint, sz}); err != nil {
				return
			}

			if err = r.bwal.Flush(); err != nil {
				return
			}

			r.bwal = nil

			if err = r.wal.Sync(); err != nil {
				return
			}

			wfi, err := r.wal.Stat()
			switch err != nil {
			case true:
				// unexpected, but ignored
			case false:
				r.peakWal = mathutil.MaxInt64(wfi.Size(), r.peakWal)
			}

			// Phase 1 commit complete

			for _, v := range r.data {
				if _, err := db.WriteAt(v.b, v.off); err != nil {
					return err
				}
			}

			if err = db.Truncate(sz); err != nil {
				return
			}

			if err = db.Sync(); err != nil {
				return
			}

			// Phase 2 commit complete

			if !r.testHook {
				if err = r.wal.Truncate(0); err != nil {
					return
				}

				if _, err = r.wal.Seek(0, 0); err != nil {
					return
				}
			}

			r.testHook = false
			return r.wal.Sync()

		},
		acidWriter,
	); err != nil {
		return
	}

	return r, nil
}

// PeakWALSize reports the maximum size WAL has ever used.
func (a ACIDFiler0) PeakWALSize() int64 {
	return a.peakWal
}

func (a *ACIDFiler0) readPacket(f *bufio.Reader) (items []interface{}, err error) {
	var b4 [4]byte
	n, err := io.ReadAtLeast(f, b4[:], 4)
	if n != 4 {
		return
	}

	ln := int(binary.BigEndian.Uint32(b4[:]))
	m := (4 + ln) % 16
	padd := (16 - m) % 16
	b := make([]byte, ln+padd)
	if n, err = io.ReadAtLeast(f, b, len(b)); n != len(b) {
		return
	}

	return DecodeScalars(b[:ln])
}

func (a *ACIDFiler0) recoverDb(db Filer) (err error) {
	fi, err := a.wal.Stat()
	if err != nil {
		return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: err}
	}

	if sz := fi.Size(); sz%16 != 0 {
		return &ErrILSEQ{Type: ErrFileSize, Name: a.wal.Name(), Arg: sz}
	}

	f := bufio.NewReader(a.wal)
	items, err := a.readPacket(f)
	if err != nil {
		return
	}

	if len(items) != 3 || items[0] != int64(wpt00Header) || items[1] != int64(walTypeACIDFiler0) {
		return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("invalid packet items %#v", items)}
	}

	tr := NewBTree(nil)

	for {
		items, err = a.readPacket(f)
		if err != nil {
			return
		}

		if len(items) < 2 {
			return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("too few packet items %#v", items)}
		}

		switch items[0] {
		case int64(wpt00WriteData):
			if len(items) != 3 {
				return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("invalid data packet items %#v", items)}
			}

			b, off := items[1].([]byte), items[2].(int64)
			var key [8]byte
			binary.BigEndian.PutUint64(key[:], uint64(off))
			if err = tr.Set(key[:], b); err != nil {
				return
			}
		case int64(wpt00Checkpoint):
			var b1 [1]byte
			if n, err := f.Read(b1[:]); n != 0 || err == nil {
				return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("checkpoint n %d, err %v", n, err)}
			}

			if len(items) != 2 {
				return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("checkpoint packet invalid items %#v", items)}
			}

			sz := items[1].(int64)
			enum, err := tr.seekFirst()
			if err != nil {
				return err
			}

			for {
				k, v, err := enum.current()
				if err != nil {
					if fileutil.IsEOF(err) {
						break
					}

					return err
				}

				if _, err = db.WriteAt(v, int64(binary.BigEndian.Uint64(k))); err != nil {
					return err
				}

				if err = enum.next(); err != nil {
					if fileutil.IsEOF(err) {
						break
					}

					return err
				}
			}

			if err = db.Truncate(sz); err != nil {
				return err
			}

			if err = db.Sync(); err != nil {
				return err
			}

			// Recovery complete

			if err = a.wal.Truncate(0); err != nil {
				return err
			}

			return a.wal.Sync()
		default:
			return &ErrILSEQ{Type: ErrInvalidWAL, Name: a.wal.Name(), More: fmt.Sprintf("packet tag %v", items[0])}
		}
	}
}
