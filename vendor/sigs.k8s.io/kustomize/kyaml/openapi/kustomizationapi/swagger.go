// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Code generated for package kustomizationapi by go-bindata DO NOT EDIT. (@generated)
// sources:
// kustomizationapi/swagger.json
package kustomizationapi

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func bindataRead(data []byte, name string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}

	var buf bytes.Buffer
	_, err = io.Copy(&buf, gz)
	clErr := gz.Close()

	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}
	if clErr != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

type asset struct {
	bytes []byte
	info  os.FileInfo
}

type bindataFileInfo struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
}

// Name return file name
func (fi bindataFileInfo) Name() string {
	return fi.name
}

// Size return file size
func (fi bindataFileInfo) Size() int64 {
	return fi.size
}

// Mode return file mode
func (fi bindataFileInfo) Mode() os.FileMode {
	return fi.mode
}

// Mode return file modify time
func (fi bindataFileInfo) ModTime() time.Time {
	return fi.modTime
}

// IsDir return file whether a directory
func (fi bindataFileInfo) IsDir() bool {
	return fi.mode&os.ModeDir != 0
}

// Sys return file is sys mode
func (fi bindataFileInfo) Sys() interface{} {
	return nil
}

var _kustomizationapiSwaggerJson = []byte("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xe4\x56\xc1\x6e\xdb\x30\x0c\xbd\xe7\x2b\x04\x6d\xc7\xd8\x45\x6e\x43\x6e\xc3\x0e\x3b\x14\x05\x0a\x74\xb7\xa1\x07\xc6\xa1\x5d\xce\x8e\xa4\x51\xb4\xb1\x6c\xc8\xbf\x0f\xd6\x62\xd7\x4a\xec\x75\x0b\x1a\xac\x4b\x0f\x06\x0c\x99\x7c\x4f\xe4\x7b\x24\xfc\x63\xa6\x94\x5e\x63\x4e\x86\x84\xac\xf1\x7a\xa9\xda\x23\xa5\x34\xd9\xb4\x7c\xe7\x53\x70\x94\x82\x73\x3e\x6d\x16\xe9\x07\x6b\x72\x2a\x6e\xc0\xbd\xe7\xe2\x31\x52\x29\xed\xd8\x3a\x64\x21\x1c\x9e\x2a\xa5\x3f\xa2\x41\x06\xb1\x7c\x90\x10\x3e\xbe\x65\xcc\xf5\x52\xe9\x37\x57\x03\xfe\xab\x11\xda\x18\xa5\x87\xd8\xed\xdf\x76\xf3\xee\x1a\xb0\x5e\x07\x14\xa8\x6e\x87\x17\xca\xa1\xf2\xd8\x07\xc9\xd6\x61\x4b\x6b\x57\x5f\x30\x13\xdd\x9f\x7f\x4b\xca\x7a\x85\x6c\x50\xd0\x27\x05\xdb\xda\x25\x0d\xb2\x27\x6b\x92\x92\xcc\x5a\x2f\xd5\xe7\x9e\x3a\xaa\x23\xc4\xb6\x88\x65\xed\xc5\x6e\xe8\x3b\xa6\x59\x68\x54\x28\x84\x6c\x4f\x11\xa2\xf7\x58\x3a\xee\x65\x14\xb2\xa7\x6d\xa3\x9a\xc5\x0a\x05\x16\xc7\x45\xdf\xcf\x06\xa5\x8f\x69\x75\x87\x19\xa3\xbc\x0c\xa1\x1e\xab\xeb\xba\x1f\xe1\x77\x8a\x78\x61\x32\xc5\xa5\x08\x3c\x10\xe0\xf9\xd5\x9d\xd2\x6b\x52\x60\x03\x1b\xf4\x0e\xb2\x3f\x6f\xfe\x3c\x4e\x3e\x25\x6f\x85\x0f\xd0\x90\xe5\x53\x72\xaf\x9b\x5b\x20\xbe\xb3\x35\x67\x78\xba\x23\x63\x94\x0b\x71\x56\x2c\xfe\xf3\x9b\xeb\x7a\x7f\x19\x90\x5f\x50\xbd\xb9\x18\xbf\xd6\xc4\x18\x17\xa4\x3f\x6d\x1d\xde\xa0\x40\xc7\x74\x3f\x7f\xca\x8c\x59\xb7\xfb\xfa\x4a\x0e\x05\x26\xc1\xcd\xa1\xea\x7f\xa3\x7b\xbc\x5d\x07\x20\xbb\xf9\x98\x11\x81\x19\xb6\x71\x27\x23\x4d\x1d\x48\xf6\x90\x6c\x90\x0b\x4c\x4a\xdc\xb6\x29\x61\x26\x9e\xca\xf0\xc2\x20\x58\x84\x84\x90\x3d\xee\x75\x1f\x56\xc5\xd9\x9a\x31\xd8\x44\x2f\xb2\x13\xff\xf5\x30\xc6\xc3\x72\x86\x61\x9c\xd8\x83\x93\xc3\x55\x91\x20\x43\x75\xb4\x33\x27\x5c\x34\xb5\x8b\x7f\x6f\x90\x51\x1b\xe7\x54\x1d\xaf\xea\xf3\xd3\xa2\x69\xfe\x0d\xeb\xeb\xf8\x8f\x89\x0d\x78\xaa\xc1\x67\xed\xb3\xfb\x19\x00\x00\xff\xff\x2f\x39\x79\xd0\x6e\x0c\x00\x00")

func kustomizationapiSwaggerJsonBytes() ([]byte, error) {
	return bindataRead(
		_kustomizationapiSwaggerJson,
		"kustomizationapi/swagger.json",
	)
}

func kustomizationapiSwaggerJson() (*asset, error) {
	bytes, err := kustomizationapiSwaggerJsonBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "kustomizationapi/swagger.json", size: 3182, mode: os.FileMode(420), modTime: time.Unix(1615228558, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

// Asset loads and returns the asset for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func Asset(name string) ([]byte, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("Asset %s can't read by error: %v", name, err)
		}
		return a.bytes, nil
	}
	return nil, fmt.Errorf("Asset %s not found", name)
}

// MustAsset is like Asset but panics when Asset would return an error.
// It simplifies safe initialization of global variables.
func MustAsset(name string) []byte {
	a, err := Asset(name)
	if err != nil {
		panic("asset: Asset(" + name + "): " + err.Error())
	}

	return a
}

// AssetInfo loads and returns the asset info for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func AssetInfo(name string) (os.FileInfo, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("AssetInfo %s can't read by error: %v", name, err)
		}
		return a.info, nil
	}
	return nil, fmt.Errorf("AssetInfo %s not found", name)
}

// AssetNames returns the names of the assets.
func AssetNames() []string {
	names := make([]string, 0, len(_bindata))
	for name := range _bindata {
		names = append(names, name)
	}
	return names
}

// _bindata is a table, holding each asset generator, mapped to its name.
var _bindata = map[string]func() (*asset, error){
	"kustomizationapi/swagger.json": kustomizationapiSwaggerJson,
}

// AssetDir returns the file names below a certain
// directory embedded in the file by go-bindata.
// For example if you run go-bindata on data/... and data contains the
// following hierarchy:
//     data/
//       foo.txt
//       img/
//         a.png
//         b.png
// then AssetDir("data") would return []string{"foo.txt", "img"}
// AssetDir("data/img") would return []string{"a.png", "b.png"}
// AssetDir("foo.txt") and AssetDir("notexist") would return an error
// AssetDir("") will return []string{"data"}.
func AssetDir(name string) ([]string, error) {
	node := _bintree
	if len(name) != 0 {
		cannonicalName := strings.Replace(name, "\\", "/", -1)
		pathList := strings.Split(cannonicalName, "/")
		for _, p := range pathList {
			node = node.Children[p]
			if node == nil {
				return nil, fmt.Errorf("Asset %s not found", name)
			}
		}
	}
	if node.Func != nil {
		return nil, fmt.Errorf("Asset %s not found", name)
	}
	rv := make([]string, 0, len(node.Children))
	for childName := range node.Children {
		rv = append(rv, childName)
	}
	return rv, nil
}

type bintree struct {
	Func     func() (*asset, error)
	Children map[string]*bintree
}

var _bintree = &bintree{nil, map[string]*bintree{
	"kustomizationapi": &bintree{nil, map[string]*bintree{
		"swagger.json": &bintree{kustomizationapiSwaggerJson, map[string]*bintree{}},
	}},
}}

// RestoreAsset restores an asset under the given directory
func RestoreAsset(dir, name string) error {
	data, err := Asset(name)
	if err != nil {
		return err
	}
	info, err := AssetInfo(name)
	if err != nil {
		return err
	}
	err = os.MkdirAll(_filePath(dir, filepath.Dir(name)), os.FileMode(0755))
	if err != nil {
		return err
	}
	err = os.WriteFile(_filePath(dir, name), data, info.Mode())
	if err != nil {
		return err
	}
	err = os.Chtimes(_filePath(dir, name), info.ModTime(), info.ModTime())
	if err != nil {
		return err
	}
	return nil
}

// RestoreAssets restores an asset under the given directory recursively
func RestoreAssets(dir, name string) error {
	children, err := AssetDir(name)
	// File
	if err != nil {
		return RestoreAsset(dir, name)
	}
	// Dir
	for _, child := range children {
		err = RestoreAssets(dir, filepath.Join(name, child))
		if err != nil {
			return err
		}
	}
	return nil
}

func _filePath(dir, name string) string {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	return filepath.Join(append([]string{dir}, strings.Split(cannonicalName, "/")...)...)
}
