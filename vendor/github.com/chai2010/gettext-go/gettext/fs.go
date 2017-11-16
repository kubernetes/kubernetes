// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gettext

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
)

type fileSystem struct {
	FsName    string
	FsRoot    string
	FsZipData []byte
	LocaleMap map[string]bool
}

func newFileSystem(path string, data []byte) *fileSystem {
	fs := &fileSystem{
		FsName:    path,
		FsZipData: data,
	}
	if err := fs.init(); err != nil {
		log.Printf("gettext-go: invalid domain, err = %v", err)
	}
	return fs
}

func (p *fileSystem) init() error {
	zipName := func(name string) string {
		if x := strings.LastIndexAny(name, `\/`); x != -1 {
			name = name[x+1:]
		}
		name = strings.TrimSuffix(name, ".zip")
		return name
	}

	// zip data
	if len(p.FsZipData) != 0 {
		p.FsRoot = zipName(p.FsName)
		p.LocaleMap = p.lsZip(p.FsZipData)
		return nil
	}

	// local dir or zip file
	fi, err := os.Stat(p.FsName)
	if err != nil {
		return err
	}

	// local dir
	if fi.IsDir() {
		p.FsRoot = p.FsName
		p.LocaleMap = p.lsDir(p.FsName)
		return nil
	}

	// local zip file
	p.FsZipData, err = ioutil.ReadFile(p.FsName)
	if err != nil {
		return err
	}
	p.FsRoot = zipName(p.FsName)
	p.LocaleMap = p.lsZip(p.FsZipData)
	return nil
}

func (p *fileSystem) LoadMessagesFile(domain, local, ext string) ([]byte, error) {
	if len(p.FsZipData) == 0 {
		trName := p.makeMessagesFileName(domain, local, ext)
		rcData, err := ioutil.ReadFile(trName)
		if err != nil {
			return nil, err
		}
		return rcData, nil
	} else {
		r, err := zip.NewReader(bytes.NewReader(p.FsZipData), int64(len(p.FsZipData)))
		if err != nil {
			return nil, err
		}

		trName := p.makeMessagesFileName(domain, local, ext)
		for _, f := range r.File {
			if f.Name != trName {
				continue
			}
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			rcData, err := ioutil.ReadAll(rc)
			rc.Close()
			return rcData, err
		}
		return nil, fmt.Errorf("not found")
	}
}

func (p *fileSystem) LoadResourceFile(domain, local, name string) ([]byte, error) {
	if len(p.FsZipData) == 0 {
		rcName := p.makeResourceFileName(domain, local, name)
		rcData, err := ioutil.ReadFile(rcName)
		if err != nil {
			return nil, err
		}
		return rcData, nil
	} else {
		r, err := zip.NewReader(bytes.NewReader(p.FsZipData), int64(len(p.FsZipData)))
		if err != nil {
			return nil, err
		}

		rcName := p.makeResourceFileName(domain, local, name)
		for _, f := range r.File {
			if f.Name != rcName {
				continue
			}
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			rcData, err := ioutil.ReadAll(rc)
			rc.Close()
			return rcData, err
		}
		return nil, fmt.Errorf("not found")
	}
}

func (p *fileSystem) makeMessagesFileName(domain, local, ext string) string {
	return fmt.Sprintf("%s/%s/LC_MESSAGES/%s%s", p.FsRoot, local, domain, ext)
}

func (p *fileSystem) makeResourceFileName(domain, local, name string) string {
	return fmt.Sprintf("%s/%s/LC_RESOURCE/%s/%s", p.FsRoot, local, domain, name)
}

func (p *fileSystem) lsZip(data []byte) map[string]bool {
	r, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil
	}
	ssMap := make(map[string]bool)
	for _, f := range r.File {
		if x := strings.Index(f.Name, "LC_MESSAGES"); x != -1 {
			s := strings.TrimRight(f.Name[:x], `\/`)
			if x = strings.LastIndexAny(s, `\/`); x != -1 {
				s = s[x+1:]
			}
			if s != "" {
				ssMap[s] = true
			}
			continue
		}
		if x := strings.Index(f.Name, "LC_RESOURCE"); x != -1 {
			s := strings.TrimRight(f.Name[:x], `\/`)
			if x = strings.LastIndexAny(s, `\/`); x != -1 {
				s = s[x+1:]
			}
			if s != "" {
				ssMap[s] = true
			}
			continue
		}
	}
	return ssMap
}

func (p *fileSystem) lsDir(path string) map[string]bool {
	list, err := ioutil.ReadDir(path)
	if err != nil {
		return nil
	}
	ssMap := make(map[string]bool)
	for _, dir := range list {
		if dir.IsDir() {
			ssMap[dir.Name()] = true
		}
	}
	return ssMap
}
