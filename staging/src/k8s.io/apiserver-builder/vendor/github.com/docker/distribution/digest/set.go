package digest

import (
	"errors"
	"sort"
	"strings"
	"sync"
)

var (
	// ErrDigestNotFound is used when a matching digest
	// could not be found in a set.
	ErrDigestNotFound = errors.New("digest not found")

	// ErrDigestAmbiguous is used when multiple digests
	// are found in a set. None of the matching digests
	// should be considered valid matches.
	ErrDigestAmbiguous = errors.New("ambiguous digest string")
)

// Set is used to hold a unique set of digests which
// may be easily referenced by easily  referenced by a string
// representation of the digest as well as short representation.
// The uniqueness of the short representation is based on other
// digests in the set. If digests are omitted from this set,
// collisions in a larger set may not be detected, therefore it
// is important to always do short representation lookups on
// the complete set of digests. To mitigate collisions, an
// appropriately long short code should be used.
type Set struct {
	mutex   sync.RWMutex
	entries digestEntries
}

// NewSet creates an empty set of digests
// which may have digests added.
func NewSet() *Set {
	return &Set{
		entries: digestEntries{},
	}
}

// checkShortMatch checks whether two digests match as either whole
// values or short values. This function does not test equality,
// rather whether the second value could match against the first
// value.
func checkShortMatch(alg Algorithm, hex, shortAlg, shortHex string) bool {
	if len(hex) == len(shortHex) {
		if hex != shortHex {
			return false
		}
		if len(shortAlg) > 0 && string(alg) != shortAlg {
			return false
		}
	} else if !strings.HasPrefix(hex, shortHex) {
		return false
	} else if len(shortAlg) > 0 && string(alg) != shortAlg {
		return false
	}
	return true
}

// Lookup looks for a digest matching the given string representation.
// If no digests could be found ErrDigestNotFound will be returned
// with an empty digest value. If multiple matches are found
// ErrDigestAmbiguous will be returned with an empty digest value.
func (dst *Set) Lookup(d string) (Digest, error) {
	dst.mutex.RLock()
	defer dst.mutex.RUnlock()
	if len(dst.entries) == 0 {
		return "", ErrDigestNotFound
	}
	var (
		searchFunc func(int) bool
		alg        Algorithm
		hex        string
	)
	dgst, err := ParseDigest(d)
	if err == ErrDigestInvalidFormat {
		hex = d
		searchFunc = func(i int) bool {
			return dst.entries[i].val >= d
		}
	} else {
		hex = dgst.Hex()
		alg = dgst.Algorithm()
		searchFunc = func(i int) bool {
			if dst.entries[i].val == hex {
				return dst.entries[i].alg >= alg
			}
			return dst.entries[i].val >= hex
		}
	}
	idx := sort.Search(len(dst.entries), searchFunc)
	if idx == len(dst.entries) || !checkShortMatch(dst.entries[idx].alg, dst.entries[idx].val, string(alg), hex) {
		return "", ErrDigestNotFound
	}
	if dst.entries[idx].alg == alg && dst.entries[idx].val == hex {
		return dst.entries[idx].digest, nil
	}
	if idx+1 < len(dst.entries) && checkShortMatch(dst.entries[idx+1].alg, dst.entries[idx+1].val, string(alg), hex) {
		return "", ErrDigestAmbiguous
	}

	return dst.entries[idx].digest, nil
}

// Add adds the given digest to the set. An error will be returned
// if the given digest is invalid. If the digest already exists in the
// set, this operation will be a no-op.
func (dst *Set) Add(d Digest) error {
	if err := d.Validate(); err != nil {
		return err
	}
	dst.mutex.Lock()
	defer dst.mutex.Unlock()
	entry := &digestEntry{alg: d.Algorithm(), val: d.Hex(), digest: d}
	searchFunc := func(i int) bool {
		if dst.entries[i].val == entry.val {
			return dst.entries[i].alg >= entry.alg
		}
		return dst.entries[i].val >= entry.val
	}
	idx := sort.Search(len(dst.entries), searchFunc)
	if idx == len(dst.entries) {
		dst.entries = append(dst.entries, entry)
		return nil
	} else if dst.entries[idx].digest == d {
		return nil
	}

	entries := append(dst.entries, nil)
	copy(entries[idx+1:], entries[idx:len(entries)-1])
	entries[idx] = entry
	dst.entries = entries
	return nil
}

// Remove removes the given digest from the set. An err will be
// returned if the given digest is invalid. If the digest does
// not exist in the set, this operation will be a no-op.
func (dst *Set) Remove(d Digest) error {
	if err := d.Validate(); err != nil {
		return err
	}
	dst.mutex.Lock()
	defer dst.mutex.Unlock()
	entry := &digestEntry{alg: d.Algorithm(), val: d.Hex(), digest: d}
	searchFunc := func(i int) bool {
		if dst.entries[i].val == entry.val {
			return dst.entries[i].alg >= entry.alg
		}
		return dst.entries[i].val >= entry.val
	}
	idx := sort.Search(len(dst.entries), searchFunc)
	// Not found if idx is after or value at idx is not digest
	if idx == len(dst.entries) || dst.entries[idx].digest != d {
		return nil
	}

	entries := dst.entries
	copy(entries[idx:], entries[idx+1:])
	entries = entries[:len(entries)-1]
	dst.entries = entries

	return nil
}

// All returns all the digests in the set
func (dst *Set) All() []Digest {
	dst.mutex.RLock()
	defer dst.mutex.RUnlock()
	retValues := make([]Digest, len(dst.entries))
	for i := range dst.entries {
		retValues[i] = dst.entries[i].digest
	}

	return retValues
}

// ShortCodeTable returns a map of Digest to unique short codes. The
// length represents the minimum value, the maximum length may be the
// entire value of digest if uniqueness cannot be achieved without the
// full value. This function will attempt to make short codes as short
// as possible to be unique.
func ShortCodeTable(dst *Set, length int) map[Digest]string {
	dst.mutex.RLock()
	defer dst.mutex.RUnlock()
	m := make(map[Digest]string, len(dst.entries))
	l := length
	resetIdx := 0
	for i := 0; i < len(dst.entries); i++ {
		var short string
		extended := true
		for extended {
			extended = false
			if len(dst.entries[i].val) <= l {
				short = dst.entries[i].digest.String()
			} else {
				short = dst.entries[i].val[:l]
				for j := i + 1; j < len(dst.entries); j++ {
					if checkShortMatch(dst.entries[j].alg, dst.entries[j].val, "", short) {
						if j > resetIdx {
							resetIdx = j
						}
						extended = true
					} else {
						break
					}
				}
				if extended {
					l++
				}
			}
		}
		m[dst.entries[i].digest] = short
		if i >= resetIdx {
			l = length
		}
	}
	return m
}

type digestEntry struct {
	alg    Algorithm
	val    string
	digest Digest
}

type digestEntries []*digestEntry

func (d digestEntries) Len() int {
	return len(d)
}

func (d digestEntries) Less(i, j int) bool {
	if d[i].val != d[j].val {
		return d[i].val < d[j].val
	}
	return d[i].alg < d[j].alg
}

func (d digestEntries) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}
