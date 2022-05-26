package afero

import (
	"os"
	"syscall"
	"time"
)

// If the cache duration is 0, cache time will be unlimited, i.e. once
// a file is in the layer, the base will never be read again for this file.
//
// For cache times greater than 0, the modification time of a file is
// checked. Note that a lot of file system implementations only allow a
// resolution of a second for timestamps... or as the godoc for os.Chtimes()
// states: "The underlying filesystem may truncate or round the values to a
// less precise time unit."
//
// This caching union will forward all write calls also to the base file
// system first. To prevent writing to the base Fs, wrap it in a read-only
// filter - Note: this will also make the overlay read-only, for writing files
// in the overlay, use the overlay Fs directly, not via the union Fs.
type CacheOnReadFs struct {
	base      Fs
	layer     Fs
	cacheTime time.Duration
}

func NewCacheOnReadFs(base Fs, layer Fs, cacheTime time.Duration) Fs {
	return &CacheOnReadFs{base: base, layer: layer, cacheTime: cacheTime}
}

type cacheState int

const (
	// not present in the overlay, unknown if it exists in the base:
	cacheMiss cacheState = iota
	// present in the overlay and in base, base file is newer:
	cacheStale
	// present in the overlay - with cache time == 0 it may exist in the base,
	// with cacheTime > 0 it exists in the base and is same age or newer in the
	// overlay
	cacheHit
	// happens if someone writes directly to the overlay without
	// going through this union
	cacheLocal
)

func (u *CacheOnReadFs) cacheStatus(name string) (state cacheState, fi os.FileInfo, err error) {
	var lfi, bfi os.FileInfo
	lfi, err = u.layer.Stat(name)
	if err == nil {
		if u.cacheTime == 0 {
			return cacheHit, lfi, nil
		}
		if lfi.ModTime().Add(u.cacheTime).Before(time.Now()) {
			bfi, err = u.base.Stat(name)
			if err != nil {
				return cacheLocal, lfi, nil
			}
			if bfi.ModTime().After(lfi.ModTime()) {
				return cacheStale, bfi, nil
			}
		}
		return cacheHit, lfi, nil
	}

	if err == syscall.ENOENT || os.IsNotExist(err) {
		return cacheMiss, nil, nil
	}

	return cacheMiss, nil, err
}

func (u *CacheOnReadFs) copyToLayer(name string) error {
	return copyToLayer(u.base, u.layer, name)
}

func (u *CacheOnReadFs) Chtimes(name string, atime, mtime time.Time) error {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit:
		err = u.base.Chtimes(name, atime, mtime)
	case cacheStale, cacheMiss:
		if err := u.copyToLayer(name); err != nil {
			return err
		}
		err = u.base.Chtimes(name, atime, mtime)
	}
	if err != nil {
		return err
	}
	return u.layer.Chtimes(name, atime, mtime)
}

func (u *CacheOnReadFs) Chmod(name string, mode os.FileMode) error {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit:
		err = u.base.Chmod(name, mode)
	case cacheStale, cacheMiss:
		if err := u.copyToLayer(name); err != nil {
			return err
		}
		err = u.base.Chmod(name, mode)
	}
	if err != nil {
		return err
	}
	return u.layer.Chmod(name, mode)
}

func (u *CacheOnReadFs) Chown(name string, uid, gid int) error {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit:
		err = u.base.Chown(name, uid, gid)
	case cacheStale, cacheMiss:
		if err := u.copyToLayer(name); err != nil {
			return err
		}
		err = u.base.Chown(name, uid, gid)
	}
	if err != nil {
		return err
	}
	return u.layer.Chown(name, uid, gid)
}

func (u *CacheOnReadFs) Stat(name string) (os.FileInfo, error) {
	st, fi, err := u.cacheStatus(name)
	if err != nil {
		return nil, err
	}
	switch st {
	case cacheMiss:
		return u.base.Stat(name)
	default: // cacheStale has base, cacheHit and cacheLocal the layer os.FileInfo
		return fi, nil
	}
}

func (u *CacheOnReadFs) Rename(oldname, newname string) error {
	st, _, err := u.cacheStatus(oldname)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit:
		err = u.base.Rename(oldname, newname)
	case cacheStale, cacheMiss:
		if err := u.copyToLayer(oldname); err != nil {
			return err
		}
		err = u.base.Rename(oldname, newname)
	}
	if err != nil {
		return err
	}
	return u.layer.Rename(oldname, newname)
}

func (u *CacheOnReadFs) Remove(name string) error {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit, cacheStale, cacheMiss:
		err = u.base.Remove(name)
	}
	if err != nil {
		return err
	}
	return u.layer.Remove(name)
}

func (u *CacheOnReadFs) RemoveAll(name string) error {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return err
	}
	switch st {
	case cacheLocal:
	case cacheHit, cacheStale, cacheMiss:
		err = u.base.RemoveAll(name)
	}
	if err != nil {
		return err
	}
	return u.layer.RemoveAll(name)
}

func (u *CacheOnReadFs) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	st, _, err := u.cacheStatus(name)
	if err != nil {
		return nil, err
	}
	switch st {
	case cacheLocal, cacheHit:
	default:
		if err := u.copyToLayer(name); err != nil {
			return nil, err
		}
	}
	if flag&(os.O_WRONLY|syscall.O_RDWR|os.O_APPEND|os.O_CREATE|os.O_TRUNC) != 0 {
		bfi, err := u.base.OpenFile(name, flag, perm)
		if err != nil {
			return nil, err
		}
		lfi, err := u.layer.OpenFile(name, flag, perm)
		if err != nil {
			bfi.Close() // oops, what if O_TRUNC was set and file opening in the layer failed...?
			return nil, err
		}
		return &UnionFile{Base: bfi, Layer: lfi}, nil
	}
	return u.layer.OpenFile(name, flag, perm)
}

func (u *CacheOnReadFs) Open(name string) (File, error) {
	st, fi, err := u.cacheStatus(name)
	if err != nil {
		return nil, err
	}

	switch st {
	case cacheLocal:
		return u.layer.Open(name)

	case cacheMiss:
		bfi, err := u.base.Stat(name)
		if err != nil {
			return nil, err
		}
		if bfi.IsDir() {
			return u.base.Open(name)
		}
		if err := u.copyToLayer(name); err != nil {
			return nil, err
		}
		return u.layer.Open(name)

	case cacheStale:
		if !fi.IsDir() {
			if err := u.copyToLayer(name); err != nil {
				return nil, err
			}
			return u.layer.Open(name)
		}
	case cacheHit:
		if !fi.IsDir() {
			return u.layer.Open(name)
		}
	}
	// the dirs from cacheHit, cacheStale fall down here:
	bfile, _ := u.base.Open(name)
	lfile, err := u.layer.Open(name)
	if err != nil && bfile == nil {
		return nil, err
	}
	return &UnionFile{Base: bfile, Layer: lfile}, nil
}

func (u *CacheOnReadFs) Mkdir(name string, perm os.FileMode) error {
	err := u.base.Mkdir(name, perm)
	if err != nil {
		return err
	}
	return u.layer.MkdirAll(name, perm) // yes, MkdirAll... we cannot assume it exists in the cache
}

func (u *CacheOnReadFs) Name() string {
	return "CacheOnReadFs"
}

func (u *CacheOnReadFs) MkdirAll(name string, perm os.FileMode) error {
	err := u.base.MkdirAll(name, perm)
	if err != nil {
		return err
	}
	return u.layer.MkdirAll(name, perm)
}

func (u *CacheOnReadFs) Create(name string) (File, error) {
	bfh, err := u.base.Create(name)
	if err != nil {
		return nil, err
	}
	lfh, err := u.layer.Create(name)
	if err != nil {
		// oops, see comment about OS_TRUNC above, should we remove? then we have to
		// remember if the file did not exist before
		bfh.Close()
		return nil, err
	}
	return &UnionFile{Base: bfh, Layer: lfh}, nil
}
