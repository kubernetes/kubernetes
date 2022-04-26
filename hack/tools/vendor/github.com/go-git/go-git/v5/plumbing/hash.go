package plumbing

import (
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"hash"
	"sort"
	"strconv"
)

// Hash SHA1 hashed content
type Hash [20]byte

// ZeroHash is Hash with value zero
var ZeroHash Hash

// ComputeHash compute the hash for a given ObjectType and content
func ComputeHash(t ObjectType, content []byte) Hash {
	h := NewHasher(t, int64(len(content)))
	h.Write(content)
	return h.Sum()
}

// NewHash return a new Hash from a hexadecimal hash representation
func NewHash(s string) Hash {
	b, _ := hex.DecodeString(s)

	var h Hash
	copy(h[:], b)

	return h
}

func (h Hash) IsZero() bool {
	var empty Hash
	return h == empty
}

func (h Hash) String() string {
	return hex.EncodeToString(h[:])
}

type Hasher struct {
	hash.Hash
}

func NewHasher(t ObjectType, size int64) Hasher {
	h := Hasher{sha1.New()}
	h.Write(t.Bytes())
	h.Write([]byte(" "))
	h.Write([]byte(strconv.FormatInt(size, 10)))
	h.Write([]byte{0})
	return h
}

func (h Hasher) Sum() (hash Hash) {
	copy(hash[:], h.Hash.Sum(nil))
	return
}

// HashesSort sorts a slice of Hashes in increasing order.
func HashesSort(a []Hash) {
	sort.Sort(HashSlice(a))
}

// HashSlice attaches the methods of sort.Interface to []Hash, sorting in
// increasing order.
type HashSlice []Hash

func (p HashSlice) Len() int           { return len(p) }
func (p HashSlice) Less(i, j int) bool { return bytes.Compare(p[i][:], p[j][:]) < 0 }
func (p HashSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// IsHash returns true if the given string is a valid hash.
func IsHash(s string) bool {
	if len(s) != 40 {
		return false
	}

	_, err := hex.DecodeString(s)
	return err == nil
}
