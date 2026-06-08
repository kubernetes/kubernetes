// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"archive/zip"
	"bytes"
	"fmt"
)

type FileSystem interface {
	LocaleList() []string
	LoadMessagesFile(domain, lang, ext string) ([]byte, error)
	LoadResourceFile(domain, lang, name string) ([]byte, error)
	String() string
}

func NewFS(name string, x interface{}) FileSystem {
	if x == nil {
		if name != "" {
			return OS(name)
		}
		return NilFS(name)
	}

	switch x := x.(type) {
	case []byte:
		if len(x) == 0 {
			return OS(name)
		}
		if r, err := zip.NewReader(bytes.NewReader(x), int64(len(x))); err == nil {
			return ZipFS(r, name)
		}
		if fs, err := newJson(x, name); err == nil {
			return fs
		}
	case string:
		if len(x) == 0 {
			return OS(name)
		}
		if r, err := zip.NewReader(bytes.NewReader([]byte(x)), int64(len(x))); err == nil {
			return ZipFS(r, name)
		}
		if fs, err := newJson([]byte(x), name); err == nil {
			return fs
		}
	case FileSystem:
		return x
	}

	return NilFS(name)
}

func OS(root string) FileSystem {
	return newOsFS(root)
}

func ZipFS(r *zip.Reader, name string) FileSystem {
	return newZipFS(r, name)
}

func NilFS(name string) FileSystem {
	return &nilFS{name}
}

type nilFS struct {
	name string
}

func (p *nilFS) LocaleList() []string {
	return nil
}

func (p *nilFS) LoadMessagesFile(domain, lang, ext string) ([]byte, error) {
	return nil, fmt.Errorf("not found")
}
func (p *nilFS) LoadResourceFile(domain, lang, name string) ([]byte, error) {
	return nil, fmt.Errorf("not found")
}
func (p *nilFS) String() string {
	return "gettext.nilfs(" + p.name + ")"
}
