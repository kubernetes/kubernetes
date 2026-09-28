// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"errors"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"syscall"
)

// OverlayFS is a read-only [fs.FS] that stacks overlays on top of a base file system.
//
// A name is resolved in the topmost layer that holds it, then down to the base.
// See [NewOverlayFS] for the order in which layers are stacked.
// When the name is absent from all layers, every method returns a [fs.PathError] that reports
// the name and matches [fs.ErrNotExist].
//
// [OverlayFS] implements [fs.FS], [fs.ReadFileFS], [fs.StatFS] and [fs.ReadDirFS], but not [fs.GlobFS].
// A directory returns an error when the resolved layer does not support reading directories.
//
// Directories are merged: a directory reports the union of the entries held by every layer,
// and the topmost layer wins whenever the same name is held by several of them.
// A layer may claim a directory for itself with [NewOpaqueFS], which stops the merge
// and hides everything the lower layers hold under that directory.
type OverlayFS struct {
	layers []fs.FS
}

// NewOverlayFS builds an overlay file system from a base file system and a list of overlays.
//
// Overlays are stacked in the order in which they are provided:
// the last one sits on top and is resolved first, then the preceding ones in reverse order.
// The base is always resolved last.
//
// An empty list of overlays yields a file system that resolves against the base alone.
func NewOverlayFS(base fs.FS, overlays ...fs.FS) *OverlayFS {
	layers := make([]fs.FS, 0, len(overlays)+1)
	for _, overlay := range slices.Backward(overlays) {
		layers = append(layers, overlay)
	}
	layers = append(layers, base)

	return &OverlayFS{
		layers: layers,
	}
}

// Open opens a file, resolving layers from the topmost overlay down to the base.
//
// Opening a directory yields a [fs.ReadDirFile] that reports the same entries as
// [OverlayFS.ReadDir], so that a merged directory reads alike either way.
func (f *OverlayFS) Open(name string) (fs.File, error) {
	file, err := f.openInLayers(name)
	if err != nil {
		return nil, err
	}

	info, err := file.Stat()
	if err != nil {
		_ = file.Close()

		return nil, err
	}

	if !info.IsDir() {
		return file, nil
	}

	// the entries of a directory come from every layer, not from the one that resolved it
	_ = file.Close()

	entries, err := f.readMergedDir(name)
	if err != nil {
		return nil, err
	}

	return &mergedDir{name: name, info: info, entries: entries}, nil
}

// ReadFile reads a file, resolving layers from the topmost overlay down to the base.
func (f *OverlayFS) ReadFile(name string) ([]byte, error) {
	layer, err := f.findInLayers("open", name)
	if err != nil {
		return nil, err
	}

	return fs.ReadFile(layer, name)
}

// Stat returns the [fs.FileInfo] of a file, resolving layers from the topmost overlay down to the base.
func (f *OverlayFS) Stat(name string) (fs.FileInfo, error) {
	layer, err := f.findInLayers("stat", name)
	if err != nil {
		return nil, err
	}

	return fs.Stat(layer, name)
}

// ReadDir lists a directory, resolving layers from the topmost overlay down to the base.
//
// The entries of every layer holding the directory are merged, sorted by file name,
// and a name held by several layers is reported by the topmost of them.
// The merge stops at the topmost layer that owns the directory, as declared by [NewOpaqueFS].
func (f *OverlayFS) ReadDir(name string) ([]fs.DirEntry, error) {
	return f.readMergedDir(name)
}

// readMergedDir collects the entries of every layer that holds a directory.
//
// A name that is not a directory in some layer shadows the layers below it,
// just like a regular file does.
func (f *OverlayFS) readMergedDir(name string) ([]fs.DirEntry, error) {
	var (
		merged []fs.DirEntry
		found  bool
	)
	seen := make(map[string]struct{})

	for _, layer := range f.layers {
		info, err := fs.Stat(layer, name)
		if err != nil {
			if !isNotFound(err) {
				return nil, err
			}

			if ownsSubtreeOf(layer, name) {
				break
			}

			continue
		}

		if !info.IsDir() {
			if found {
				// a directory held by an upper layer shadows this entry
				break
			}

			// let the layer report why the name cannot be listed
			return fs.ReadDir(layer, name)
		}

		entries, err := fs.ReadDir(layer, name)
		if err != nil {
			return nil, err
		}

		found = true
		for _, entry := range entries {
			if _, isShadowed := seen[entry.Name()]; isShadowed {
				continue
			}

			seen[entry.Name()] = struct{}{}
			merged = append(merged, entry)
		}

		if declaresOpaqueDir(layer, name) || ownsSubtreeOf(layer, name) {
			// this layer owns the directory, or one of its parents:
			// the layers below it contribute nothing
			break
		}
	}

	if !found {
		return nil, notFound("readdir", name)
	}

	slices.SortFunc(merged, func(a, b fs.DirEntry) int {
		return strings.Compare(a.Name(), b.Name())
	})

	return merged, nil
}

// openInLayers opens a name in the topmost layer that holds it.
func (f *OverlayFS) openInLayers(name string) (fs.File, error) {
	for _, layer := range f.layers {
		file, err := layer.Open(name)
		if err == nil {
			return file, nil
		}

		if !isNotFound(err) {
			return nil, err
		}

		if ownsSubtreeOf(layer, name) {
			break
		}
	}

	return nil, notFound("open", name)
}

// findInLayers returns the topmost layer that holds a name.
//
// op names the operation reported by the [fs.PathError] raised when no layer holds the name.
func (f *OverlayFS) findInLayers(op, name string) (fs.FS, error) {
	for _, layer := range f.layers {
		err := probeInLayer(layer, name)
		if err == nil {
			return layer, nil
		}

		if !isNotFound(err) {
			return nil, err
		}

		if ownsSubtreeOf(layer, name) {
			break
		}
	}

	return nil, notFound(op, name)
}

// probeInLayer reports the error raised by a layer when looking a name up.
//
// A layer implementing [fs.StatFS] is probed with Stat.
// Otherwise the name is opened, then closed again.
func probeInLayer(layer fs.FS, name string) error {
	if statFS, supportsStat := layer.(fs.StatFS); supportsStat {
		_, err := statFS.Stat(name)

		return err
	}

	file, err := layer.Open(name)
	if err == nil {
		_ = file.Close()
	}

	return err
}

// declaresOpaqueDir tells whether a layer marks a directory as entirely owned.
func declaresOpaqueDir(layer fs.FS, name string) bool {
	opaque, isOpaqueFS := layer.(OpaqueDirFS)

	return isOpaqueFS && opaque.IsOpaqueDir(name)
}

// ownsDir tells whether a layer marks a directory as entirely owned, and holds it.
func ownsDir(layer fs.FS, name string) bool {
	if !declaresOpaqueDir(layer, name) {
		return false
	}

	info, err := fs.Stat(layer, name)

	return err == nil && info.IsDir()
}

// ownsSubtreeOf tells whether a layer owns one of the parent directories of a name.
//
// The layers below such a layer hold nothing that is reachable under that name.
func ownsSubtreeOf(layer fs.FS, name string) bool {
	if _, isOpaqueFS := layer.(OpaqueDirFS); !isOpaqueFS {
		return false
	}

	for dir := path.Dir(name); ; {
		if ownsDir(layer, dir) {
			return true
		}

		// "." and "/" are their own parent, so this is where the walk up ends
		parent := path.Dir(dir)
		if parent == dir {
			return false
		}

		dir = parent
	}
}

// errIsDir is reported when a merged directory is read as a regular file.
var errIsDir = errors.New("is a directory")

// mergedDir is the [fs.ReadDirFile] returned when opening a directory with merged layers.
//
// Its [fs.FileInfo] is the one of the topmost layer holding the directory,
// while its entries come from all the layers holding it.
type mergedDir struct {
	name    string
	info    fs.FileInfo
	entries []fs.DirEntry
	offset  int
}

func (d *mergedDir) Stat() (fs.FileInfo, error) { return d.info, nil }

func (d *mergedDir) Close() error { return nil }

func (d *mergedDir) Read([]byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.name, Err: errIsDir}
}

// ReadDir returns the next n entries, or all the remaining ones when n is not positive.
func (d *mergedDir) ReadDir(n int) ([]fs.DirEntry, error) {
	remaining := len(d.entries) - d.offset
	if n <= 0 {
		entries := d.entries[d.offset:]
		d.offset = len(d.entries)

		return entries, nil
	}

	if remaining == 0 {
		return nil, io.EOF
	}

	n = min(n, remaining)
	entries := d.entries[d.offset : d.offset+n]
	d.offset += n

	return entries, nil
}

// notFound builds the error reported when a name is absent from all layers.
//
// It mirrors what the os package raises for a missing file,
// so that a caller may retrieve the name from the error alone.
func notFound(op, name string) error {
	return &fs.PathError{Op: op, Path: name, Err: fs.ErrNotExist}
}

// isNotFound tells whether an error raised by a layer means that the name is absent from that layer.
//
// The whole error chain is inspected, so a layer that wraps its errors resolves like any other,
// and so does a layer that is itself an [OverlayFS].
//
// [syscall.ENOTDIR] is matched explicitly, because a path traversing a regular file
// does not report [fs.ErrNotExist] on all platforms.
// Any other error stops the resolution and is reported to the caller.
func isNotFound(err error) bool {
	return errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR)
}
