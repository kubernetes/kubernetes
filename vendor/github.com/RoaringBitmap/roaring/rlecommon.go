package roaring

import (
	"fmt"
)

// common to rle32.go and rle16.go

// rleVerbose controls whether p() prints show up.
// The testing package sets this based on
// testing.Verbose().
var rleVerbose bool

// p is a shorthand for fmt.Printf with beginning and
// trailing newlines. p() makes it easy
// to add diagnostic print statements.
func p(format string, args ...interface{}) {
	if rleVerbose {
		fmt.Printf("\n"+format+"\n", args...)
	}
}

// MaxUint32 is the largest uint32 value.
const MaxUint32 = 4294967295

// MaxUint16 is the largest 16 bit unsigned int.
// This is the largest value an interval16 can store.
const MaxUint16 = 65535

// searchOptions allows us to accelerate runContainer32.search with
// prior knowledge of (mostly lower) bounds. This is used by Union
// and Intersect.
type searchOptions struct {

	// start here instead of at 0
	startIndex int64

	// upper bound instead of len(rc.iv);
	// endxIndex == 0 means ignore the bound and use
	// endxIndex == n ==len(rc.iv) which is also
	// naturally the default for search()
	// when opt = nil.
	endxIndex int64
}

// And finds the intersection of rc and b.
func (rc *runContainer32) And(b *Bitmap) *Bitmap {
	out := NewBitmap()
	for _, p := range rc.iv {
		for i := p.start; i <= p.last; i++ {
			if b.Contains(i) {
				out.Add(i)
			}
		}
	}
	return out
}

// Xor returns the exclusive-or of rc and b.
func (rc *runContainer32) Xor(b *Bitmap) *Bitmap {
	out := b.Clone()
	for _, p := range rc.iv {
		for v := p.start; v <= p.last; v++ {
			if out.Contains(v) {
				out.RemoveRange(uint64(v), uint64(v+1))
			} else {
				out.Add(v)
			}
		}
	}
	return out
}

// Or returns the union of rc and b.
func (rc *runContainer32) Or(b *Bitmap) *Bitmap {
	out := b.Clone()
	for _, p := range rc.iv {
		for v := p.start; v <= p.last; v++ {
			out.Add(v)
		}
	}
	return out
}

// trial is used in the randomized testing of runContainers
type trial struct {
	n           int
	percentFill float64
	ntrial      int

	// only in the union test
	// only subtract test
	percentDelete float64

	// only in 067 randomized operations
	// we do this + 1 passes
	numRandomOpsPass int

	// allow sampling range control
	// only recent tests respect this.
	srang *interval16
}

// And finds the intersection of rc and b.
func (rc *runContainer16) And(b *Bitmap) *Bitmap {
	out := NewBitmap()
	for _, p := range rc.iv {
		for i := p.start; i <= p.last; i++ {
			if b.Contains(uint32(i)) {
				out.Add(uint32(i))
			}
		}
	}
	return out
}

// Xor returns the exclusive-or of rc and b.
func (rc *runContainer16) Xor(b *Bitmap) *Bitmap {
	out := b.Clone()
	for _, p := range rc.iv {
		for v := p.start; v <= p.last; v++ {
			w := uint32(v)
			if out.Contains(w) {
				out.RemoveRange(uint64(w), uint64(w+1))
			} else {
				out.Add(w)
			}
		}
	}
	return out
}

// Or returns the union of rc and b.
func (rc *runContainer16) Or(b *Bitmap) *Bitmap {
	out := b.Clone()
	for _, p := range rc.iv {
		for v := p.start; v <= p.last; v++ {
			out.Add(uint32(v))
		}
	}
	return out
}

//func (rc *runContainer32) and(container) container {
//	panic("TODO. not yet implemented")
//}

// serializedSizeInBytes returns the number of bytes of memory
// required by this runContainer16. This is for the
// Roaring format, as specified https://github.com/RoaringBitmap/RoaringFormatSpec/
func (rc *runContainer16) serializedSizeInBytes() int {
	// number of runs in one uint16, then each run
	// needs two more uint16
	return 2 + len(rc.iv)*4
}

// serializedSizeInBytes returns the number of bytes of memory
// required by this runContainer32.
func (rc *runContainer32) serializedSizeInBytes() int {
	return 4 + len(rc.iv)*8
}
