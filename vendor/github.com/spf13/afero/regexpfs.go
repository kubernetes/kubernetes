package afero

import (
	"os"
	"regexp"
	"syscall"
	"time"
)

// The RegexpFs filters files (not directories) by regular expression. Only
// files matching the given regexp will be allowed, all others get a ENOENT error (
// "No such file or directory").
//
type RegexpFs struct {
	re     *regexp.Regexp
	source Fs
}

func NewRegexpFs(source Fs, re *regexp.Regexp) Fs {
	return &RegexpFs{source: source, re: re}
}

type RegexpFile struct {
	f  File
	re *regexp.Regexp
}

func (r *RegexpFs) matchesName(name string) error {
	if r.re == nil {
		return nil
	}
	if r.re.MatchString(name) {
		return nil
	}
	return syscall.ENOENT
}

func (r *RegexpFs) dirOrMatches(name string) error {
	dir, err := IsDir(r.source, name)
	if err != nil {
		return err
	}
	if dir {
		return nil
	}
	return r.matchesName(name)
}

func (r *RegexpFs) Chtimes(name string, a, m time.Time) error {
	if err := r.dirOrMatches(name); err != nil {
		return err
	}
	return r.source.Chtimes(name, a, m)
}

func (r *RegexpFs) Chmod(name string, mode os.FileMode) error {
	if err := r.dirOrMatches(name); err != nil {
		return err
	}
	return r.source.Chmod(name, mode)
}

func (r *RegexpFs) Name() string {
	return "RegexpFs"
}

func (r *RegexpFs) Stat(name string) (os.FileInfo, error) {
	if err := r.dirOrMatches(name); err != nil {
		return nil, err
	}
	return r.source.Stat(name)
}

func (r *RegexpFs) Rename(oldname, newname string) error {
	dir, err := IsDir(r.source, oldname)
	if err != nil {
		return err
	}
	if dir {
		return nil
	}
	if err := r.matchesName(oldname); err != nil {
		return err
	}
	if err := r.matchesName(newname); err != nil {
		return err
	}
	return r.source.Rename(oldname, newname)
}

func (r *RegexpFs) RemoveAll(p string) error {
	dir, err := IsDir(r.source, p)
	if err != nil {
		return err
	}
	if !dir {
		if err := r.matchesName(p); err != nil {
			return err
		}
	}
	return r.source.RemoveAll(p)
}

func (r *RegexpFs) Remove(name string) error {
	if err := r.dirOrMatches(name); err != nil {
		return err
	}
	return r.source.Remove(name)
}

func (r *RegexpFs) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	if err := r.dirOrMatches(name); err != nil {
		return nil, err
	}
	return r.source.OpenFile(name, flag, perm)
}

func (r *RegexpFs) Open(name string) (File, error) {
	dir, err := IsDir(r.source, name)
	if err != nil {
		return nil, err
	}
	if !dir {
		if err := r.matchesName(name); err != nil {
			return nil, err
		}
	}
	f, err := r.source.Open(name)
	return &RegexpFile{f: f, re: r.re}, nil
}

func (r *RegexpFs) Mkdir(n string, p os.FileMode) error {
	return r.source.Mkdir(n, p)
}

func (r *RegexpFs) MkdirAll(n string, p os.FileMode) error {
	return r.source.MkdirAll(n, p)
}

func (r *RegexpFs) Create(name string) (File, error) {
	if err := r.matchesName(name); err != nil {
		return nil, err
	}
	return r.source.Create(name)
}

func (f *RegexpFile) Close() error {
	return f.f.Close()
}

func (f *RegexpFile) Read(s []byte) (int, error) {
	return f.f.Read(s)
}

func (f *RegexpFile) ReadAt(s []byte, o int64) (int, error) {
	return f.f.ReadAt(s, o)
}

func (f *RegexpFile) Seek(o int64, w int) (int64, error) {
	return f.f.Seek(o, w)
}

func (f *RegexpFile) Write(s []byte) (int, error) {
	return f.f.Write(s)
}

func (f *RegexpFile) WriteAt(s []byte, o int64) (int, error) {
	return f.f.WriteAt(s, o)
}

func (f *RegexpFile) Name() string {
	return f.f.Name()
}

func (f *RegexpFile) Readdir(c int) (fi []os.FileInfo, err error) {
	var rfi []os.FileInfo
	rfi, err = f.f.Readdir(c)
	if err != nil {
		return nil, err
	}
	for _, i := range rfi {
		if i.IsDir() || f.re.MatchString(i.Name()) {
			fi = append(fi, i)
		}
	}
	return fi, nil
}

func (f *RegexpFile) Readdirnames(c int) (n []string, err error) {
	fi, err := f.Readdir(c)
	if err != nil {
		return nil, err
	}
	for _, s := range fi {
		n = append(n, s.Name())
	}
	return n, nil
}

func (f *RegexpFile) Stat() (os.FileInfo, error) {
	return f.f.Stat()
}

func (f *RegexpFile) Sync() error {
	return f.f.Sync()
}

func (f *RegexpFile) Truncate(s int64) error {
	return f.f.Truncate(s)
}

func (f *RegexpFile) WriteString(s string) (int, error) {
	return f.f.WriteString(s)
}
