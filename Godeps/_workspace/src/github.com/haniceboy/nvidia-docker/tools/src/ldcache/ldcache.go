// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package ldcache

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

const ldcachePath = "/etc/ld.so.cache"

const (
	magicString1 = "ld.so-1.7.0"
	magicString2 = "glibc-ld.so.cache"
	magicVersion = "1.1"
)

const (
	flagTypeMask = 0x00ff
	flagTypeELF  = 0x0001

	flagArchMask  = 0xff00
	flagArchI386  = 0x0000
	flagArchX8664 = 0x0300
	flagArchX32   = 0x0800
)

var ErrInvalidCache = errors.New("invalid ld.so.cache file")

type Header1 struct {
	Magic [len(magicString1) + 1]byte // include null delimiter
	NLibs uint32
}

type Entry1 struct {
	Flags      int32
	Key, Value uint32
}

type Header2 struct {
	Magic     [len(magicString2)]byte
	Version   [len(magicVersion)]byte
	NLibs     uint32
	TableSize uint32
	_         [3]uint32 // unused
	_         uint64    // force 8 byte alignment
}

type Entry2 struct {
	Flags      int32
	Key, Value uint32
	OSVersion  uint32
	HWCap      uint64
}

type LDCache struct {
	*bytes.Reader

	data, libs []byte
	header     Header2
	entries    []Entry2
}

func Open() (*LDCache, error) {
	f, err := os.Open(ldcachePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	d, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()),
		syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}

	cache := &LDCache{data: d, Reader: bytes.NewReader(d)}
	return cache, cache.parse()
}

func (c *LDCache) Close() error {
	return syscall.Munmap(c.data)
}

func (c *LDCache) Magic() string {
	return string(c.header.Magic[:])
}

func (c *LDCache) Version() string {
	return string(c.header.Version[:])
}

func strn(b []byte, n int) string {
	return string(b[:n])
}

func (c *LDCache) parse() error {
	var header Header1

	// Check for the old format (< glibc-2.2)
	if c.Len() <= int(unsafe.Sizeof(header)) {
		return ErrInvalidCache
	}
	if strn(c.data, len(magicString1)) == magicString1 {
		if err := binary.Read(c, binary.LittleEndian, &header); err != nil {
			return err
		}
		n := int64(header.NLibs) * int64(unsafe.Sizeof(Entry1{}))
		offset, err := c.Seek(n, 1) // skip old entries
		if err != nil {
			return err
		}
		n = (-offset) & int64(unsafe.Alignof(c.header)-1)
		_, err = c.Seek(n, 1) // skip padding
		if err != nil {
			return err
		}
	}

	c.libs = c.data[c.Size()-int64(c.Len()):] // kv offsets start here
	if err := binary.Read(c, binary.LittleEndian, &c.header); err != nil {
		return err
	}
	if c.Magic() != magicString2 || c.Version() != magicVersion {
		return ErrInvalidCache
	}
	c.entries = make([]Entry2, c.header.NLibs)
	if err := binary.Read(c, binary.LittleEndian, &c.entries); err != nil {
		return err
	}
	return nil
}

func (c *LDCache) Lookup(libs ...string) (paths32, paths64 []string) {
	type void struct{}
	var paths *[]string

	set := make(map[string]void)
	prefix := make([][]byte, len(libs))

	for i := range libs {
		prefix[i] = []byte(libs[i])
	}
	for _, e := range c.entries {
		if ((e.Flags & flagTypeMask) & flagTypeELF) == 0 {
			continue
		}
		switch e.Flags & flagArchMask {
		case flagArchX8664:
			paths = &paths64
		case flagArchX32:
			fallthrough
		case flagArchI386:
			paths = &paths32
		default:
			continue
		}
		if e.Key > uint32(len(c.libs)) || e.Value > uint32(len(c.libs)) {
			continue
		}
		lib := c.libs[e.Key:]
		value := c.libs[e.Value:]

		for _, p := range prefix {
			if bytes.HasPrefix(lib, p) {
				n := bytes.IndexByte(value, 0)
				if n < 0 {
					break
				}
				path, err := filepath.EvalSymlinks(strn(value, n))
				if err != nil {
					break
				}
				if _, ok := set[path]; ok {
					break
				}
				set[path] = void{}
				*paths = append(*paths, path)
				break
			}
		}
	}
	return
}
