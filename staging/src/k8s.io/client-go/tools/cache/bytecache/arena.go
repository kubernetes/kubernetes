//go:build linux && amd64 && !race

/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package bytecache is a prototype of an informer-style object cache that
// stores objects as serialized bytes in an mmap-ed file instead of as live
// Go objects on the heap.
//
// The file is mapped twice with MAP_SHARED:
//   - rw: PROT_READ|PROT_WRITE — only the store's write path touches this.
//   - ro: PROT_READ — everything handed out to readers aliases this mapping,
//     so a stray write is a hardware fault, not silent corruption.
//
// Because both mappings are MAP_SHARED views of the same file, they are backed
// by the same physical pages in the page cache: a write through rw is
// immediately visible through ro, and the kernel may write pages back to the
// file and evict them under memory pressure (no swap required — the file
// itself is the backing store).
package bytecache

import (
	"fmt"
	"os"
	"syscall"
)

// Arena is an append-only allocator over a twice-mapped file.
type Arena struct {
	f   *os.File
	rw  []byte // PROT_READ|PROT_WRITE MAP_SHARED
	ro  []byte // PROT_READ MAP_SHARED — same pages as rw
	off int
}

func NewArena(path string, size int) (*Arena, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	// Sparse file: pages are allocated lazily as we write.
	if err := f.Truncate(int64(size)); err != nil {
		_ = f.Close()
		return nil, err
	}
	rw, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("rw mmap: %w", err)
	}
	ro, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = syscall.Munmap(rw)
		_ = f.Close()
		return nil, fmt.Errorf("ro mmap: %w", err)
	}
	return &Arena{f: f, rw: rw, ro: ro}, nil
}

// Bump reserves size bytes at the given alignment and returns the offset.
func (a *Arena) Bump(size, align int) (int, error) {
	off := (a.off + align - 1) &^ (align - 1)
	if off+size > len(a.rw) {
		return 0, fmt.Errorf("arena full (%d used of %d)", a.off, len(a.rw))
	}
	a.off = off + size
	return off, nil
}

// Reset discards all allocations (benchmark convenience; a real store needs
// compaction instead).
func (a *Arena) Reset() { a.off = 0 }

// Alloc copies b into the arena and returns its offset.
// Append-only: updates write a new copy; reclaiming dead space is a
// compaction problem left out of the prototype.
func (a *Arena) Alloc(b []byte) (int, error) {
	if a.off+len(b) > len(a.rw) {
		return 0, fmt.Errorf("arena full (%d used of %d)", a.off, len(a.rw))
	}
	off := a.off
	copy(a.rw[off:], b)
	a.off += len(b)
	return off, nil
}

// ReadOnly returns a slice aliasing the read-only mapping. Writing through the
// result is a segfault by construction.
func (a *Arena) ReadOnly(off, n int) []byte {
	return a.ro[off : off+n : off+n]
}

func (a *Arena) Used() int { return a.off }

func (a *Arena) Close() error {
	_ = syscall.Munmap(a.ro)
	_ = syscall.Munmap(a.rw)
	name := a.f.Name()
	_ = a.f.Close()
	_ = os.Remove(name) // may already be unlinked
	return nil
}
