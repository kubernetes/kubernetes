package idxfile

import "gopkg.in/src-d/go-git.v4/plumbing"

const (
	// VersionSupported is the only idx version supported.
	VersionSupported = 2

	offsetLimit = 0x7fffffff
)

var (
	idxHeader = []byte{255, 't', 'O', 'c'}
)

// Idxfile is the in memory representation of an idx file.
type Idxfile struct {
	Version          uint32
	Fanout           [255]uint32
	ObjectCount      uint32
	Entries          EntryList
	PackfileChecksum [20]byte
	IdxChecksum      [20]byte
}

func NewIdxfile() *Idxfile {
	return &Idxfile{}
}

// Entry is the in memory representation of an object entry in the idx file.
type Entry struct {
	Hash   plumbing.Hash
	CRC32  uint32
	Offset uint64
}

// Add adds a new Entry with the given values to the Idxfile.
func (idx *Idxfile) Add(h plumbing.Hash, offset uint64, crc32 uint32) {
	idx.Entries = append(idx.Entries, &Entry{
		Hash:   h,
		Offset: offset,
		CRC32:  crc32,
	})
}

func (idx *Idxfile) isValid() bool {
	fanout := idx.calculateFanout()
	for k, c := range idx.Fanout {
		if fanout[k] != c {
			return false
		}
	}

	return true
}

func (idx *Idxfile) calculateFanout() [256]uint32 {
	fanout := [256]uint32{}
	for _, e := range idx.Entries {
		fanout[e.Hash[0]]++
	}

	for i := 1; i < 256; i++ {
		fanout[i] += fanout[i-1]
	}

	return fanout
}
