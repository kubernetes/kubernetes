// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A memory-only implementation of Filer.

/*

pgBits: 8
BenchmarkMemFilerWrSeq	  100000	     19430 ns/op	1646.93 MB/s
BenchmarkMemFilerRdSeq	  100000	     17390 ns/op	1840.13 MB/s
BenchmarkMemFilerWrRand	 1000000	      1903 ns/op	 133.94 MB/s
BenchmarkMemFilerRdRand	 1000000	      1153 ns/op	 221.16 MB/s

pgBits: 9
BenchmarkMemFilerWrSeq	  100000	     16195 ns/op	1975.80 MB/s
BenchmarkMemFilerRdSeq	  200000	     13011 ns/op	2459.39 MB/s
BenchmarkMemFilerWrRand	 1000000	      2248 ns/op	 227.28 MB/s
BenchmarkMemFilerRdRand	 1000000	      1177 ns/op	 433.94 MB/s

pgBits: 10
BenchmarkMemFilerWrSeq	  100000	     16169 ns/op	1979.04 MB/s
BenchmarkMemFilerRdSeq	  200000	     12673 ns/op	2524.91 MB/s
BenchmarkMemFilerWrRand	 1000000	      5550 ns/op	 184.30 MB/s
BenchmarkMemFilerRdRand	 1000000	      1699 ns/op	 601.79 MB/s

pgBits: 11
BenchmarkMemFilerWrSeq	  100000	     13449 ns/op	2379.31 MB/s
BenchmarkMemFilerRdSeq	  200000	     12058 ns/op	2653.80 MB/s
BenchmarkMemFilerWrRand	  500000	      4335 ns/op	 471.47 MB/s
BenchmarkMemFilerRdRand	 1000000	      2843 ns/op	 719.47 MB/s

pgBits: 12
BenchmarkMemFilerWrSeq	  200000	     11976 ns/op	2672.00 MB/s
BenchmarkMemFilerRdSeq	  200000	     12255 ns/op	2611.06 MB/s
BenchmarkMemFilerWrRand	  200000	      8058 ns/op	 507.14 MB/s
BenchmarkMemFilerRdRand	  500000	      4365 ns/op	 936.15 MB/s

pgBits: 13
BenchmarkMemFilerWrSeq	  200000	     10852 ns/op	2948.69 MB/s
BenchmarkMemFilerRdSeq	  200000	     11561 ns/op	2767.77 MB/s
BenchmarkMemFilerWrRand	  200000	      9748 ns/op	 840.15 MB/s
BenchmarkMemFilerRdRand	  500000	      7236 ns/op	1131.59 MB/s

pgBits: 14
BenchmarkMemFilerWrSeq	  200000	     10328 ns/op	3098.12 MB/s
BenchmarkMemFilerRdSeq	  200000	     11292 ns/op	2833.66 MB/s
BenchmarkMemFilerWrRand	  100000	     16768 ns/op	 978.75 MB/s
BenchmarkMemFilerRdRand	  200000	     13033 ns/op	1258.43 MB/s

pgBits: 15
BenchmarkMemFilerWrSeq	  200000	     10309 ns/op	3103.93 MB/s
BenchmarkMemFilerRdSeq	  200000	     11126 ns/op	2876.12 MB/s
BenchmarkMemFilerWrRand	   50000	     31985 ns/op	1021.74 MB/s
BenchmarkMemFilerRdRand	  100000	     25217 ns/op	1297.65 MB/s

pgBits: 16
BenchmarkMemFilerWrSeq	  200000	     10324 ns/op	3099.45 MB/s
BenchmarkMemFilerRdSeq	  200000	     11201 ns/op	2856.80 MB/s
BenchmarkMemFilerWrRand	   20000	     55226 ns/op	1184.76 MB/s
BenchmarkMemFilerRdRand	   50000	     48316 ns/op	1355.16 MB/s

pgBits: 17
BenchmarkMemFilerWrSeq	  200000	     10377 ns/op	3083.53 MB/s
BenchmarkMemFilerRdSeq	  200000	     11018 ns/op	2904.18 MB/s
BenchmarkMemFilerWrRand	   10000	    143425 ns/op	 913.12 MB/s
BenchmarkMemFilerRdRand	   20000	     95267 ns/op	1376.99 MB/s

pgBits: 18
BenchmarkMemFilerWrSeq	  200000	     10312 ns/op	3102.96 MB/s
BenchmarkMemFilerRdSeq	  200000	     11069 ns/op	2890.84 MB/s
BenchmarkMemFilerWrRand	    5000	    280910 ns/op	 934.14 MB/s
BenchmarkMemFilerRdRand	   10000	    188500 ns/op	1388.17 MB/s

*/

package lldb

import (
	"bytes"
	"fmt"
	"io"

	"github.com/cznic/fileutil"
	"github.com/cznic/mathutil"
)

const (
	pgBits = 16
	pgSize = 1 << pgBits
	pgMask = pgSize - 1
)

var _ Filer = &MemFiler{} // Ensure MemFiler is a Filer.

type memFilerMap map[int64]*[pgSize]byte

// MemFiler is a memory backed Filer. It implements BeginUpdate, EndUpdate and
// Rollback as no-ops. MemFiler is not automatically persistent, but it has
// ReadFrom and WriteTo methods.
type MemFiler struct {
	m    memFilerMap
	nest int
	size int64
}

// NewMemFiler returns a new MemFiler.
func NewMemFiler() *MemFiler {
	return &MemFiler{m: memFilerMap{}}
}

// BeginUpdate implements Filer.
func (f *MemFiler) BeginUpdate() error {
	f.nest++
	return nil
}

// Close implements Filer.
func (f *MemFiler) Close() (err error) {
	if f.nest != 0 {
		return &ErrPERM{(f.Name() + ":Close")}
	}

	return
}

// EndUpdate implements Filer.
func (f *MemFiler) EndUpdate() (err error) {
	if f.nest == 0 {
		return &ErrPERM{(f.Name() + ": EndUpdate")}
	}

	f.nest--
	return
}

// Name implements Filer.
func (f *MemFiler) Name() string {
	return fmt.Sprintf("%p.memfiler", f)
}

// PunchHole implements Filer.
func (f *MemFiler) PunchHole(off, size int64) (err error) {
	if off < 0 {
		return &ErrINVAL{f.Name() + ": PunchHole off", off}
	}

	if size < 0 || off+size > f.size {
		return &ErrINVAL{f.Name() + ": PunchHole size", size}
	}

	first := off >> pgBits
	if off&pgMask != 0 {
		first++
	}
	off += size - 1
	last := off >> pgBits
	if off&pgMask != 0 {
		last--
	}
	if limit := f.size >> pgBits; last > limit {
		last = limit
	}
	for pg := first; pg <= last; pg++ {
		delete(f.m, pg)
	}
	return
}

var zeroPage [pgSize]byte

// ReadAt implements Filer.
func (f *MemFiler) ReadAt(b []byte, off int64) (n int, err error) {
	avail := f.size - off
	pgI := off >> pgBits
	pgO := int(off & pgMask)
	rem := len(b)
	if int64(rem) >= avail {
		rem = int(avail)
		err = io.EOF
	}
	for rem != 0 && avail > 0 {
		pg := f.m[pgI]
		if pg == nil {
			pg = &zeroPage
		}
		nc := copy(b[:mathutil.Min(rem, pgSize)], pg[pgO:])
		pgI++
		pgO = 0
		rem -= nc
		n += nc
		b = b[nc:]
	}
	return
}

// ReadFrom is a helper to populate MemFiler's content from r.  'n' reports the
// number of bytes read from 'r'.
func (f *MemFiler) ReadFrom(r io.Reader) (n int64, err error) {
	if err = f.Truncate(0); err != nil {
		return
	}

	var (
		b   [pgSize]byte
		rn  int
		off int64
	)

	var rerr error
	for rerr == nil {
		if rn, rerr = r.Read(b[:]); rn != 0 {
			f.WriteAt(b[:rn], off)
			off += int64(rn)
			n += int64(rn)
		}
	}
	if !fileutil.IsEOF(rerr) {
		err = rerr
	}
	return
}

// Rollback implements Filer.
func (f *MemFiler) Rollback() (err error) { return }

// Size implements Filer.
func (f *MemFiler) Size() (int64, error) {
	return f.size, nil
}

// Sync implements Filer.
func (f *MemFiler) Sync() error {
	return nil
}

// Truncate implements Filer.
func (f *MemFiler) Truncate(size int64) (err error) {
	switch {
	case size < 0:
		return &ErrINVAL{"Truncate size", size}
	case size == 0:
		f.m = memFilerMap{}
		f.size = 0
		return
	}

	first := size >> pgBits
	if size&pgMask != 0 {
		first++
	}
	last := f.size >> pgBits
	if f.size&pgMask != 0 {
		last++
	}
	for ; first < last; first++ {
		delete(f.m, first)
	}

	f.size = size
	return
}

// WriteAt implements Filer.
func (f *MemFiler) WriteAt(b []byte, off int64) (n int, err error) {
	pgI := off >> pgBits
	pgO := int(off & pgMask)
	n = len(b)
	rem := n
	var nc int
	for rem != 0 {
		if pgO == 0 && rem >= pgSize && bytes.Equal(b[:pgSize], zeroPage[:]) {
			delete(f.m, pgI)
			nc = pgSize
		} else {
			pg := f.m[pgI]
			if pg == nil {
				pg = new([pgSize]byte)
				f.m[pgI] = pg
			}
			nc = copy((*pg)[pgO:], b)
		}
		pgI++
		pgO = 0
		rem -= nc
		b = b[nc:]
	}
	f.size = mathutil.MaxInt64(f.size, off+int64(n))
	return
}

// WriteTo is a helper to copy/persist MemFiler's content to w.  If w is also
// an io.WriterAt then WriteTo may attempt to _not_ write any big, for some
// value of big, runs of zeros, i.e. it will attempt to punch holes, where
// possible, in `w` if that happens to be a freshly created or to zero length
// truncated OS file.  'n' reports the number of bytes written to 'w'.
func (f *MemFiler) WriteTo(w io.Writer) (n int64, err error) {
	var (
		b      [pgSize]byte
		wn, rn int
		off    int64
		rerr   error
	)

	if wa, ok := w.(io.WriterAt); ok {
		lastPgI := f.size >> pgBits
		for pgI := int64(0); pgI <= lastPgI; pgI++ {
			sz := pgSize
			if pgI == lastPgI {
				sz = int(f.size & pgMask)
			}
			pg := f.m[pgI]
			if pg != nil {
				wn, err = wa.WriteAt(pg[:sz], off)
				if err != nil {
					return
				}

				n += int64(wn)
				off += int64(sz)
				if wn != sz {
					return n, io.ErrShortWrite
				}
			}
		}
		return
	}

	var werr error
	for rerr == nil {
		if rn, rerr = f.ReadAt(b[:], off); rn != 0 {
			off += int64(rn)
			if wn, werr = w.Write(b[:rn]); werr != nil {
				return n, werr
			}

			n += int64(wn)
		}
	}
	if !fileutil.IsEOF(rerr) {
		err = rerr
	}
	return
}
