/*
Copyright 2022 The Kubernetes Authors.

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

package disk

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/peterbourgon/diskv"
	"k8s.io/klog/v2"
)

// SizeLimitedDiskCache is an implementation of httpcache.Cache that supplements
// the in-memory map with persistent storage up to a pre-defined maxSize. After
// the disk cache size breaches the size limit, the oldest files (by file
// modification times) are evicted until the size limit is satisfied. This cache
// is meant to be a substitute for "github.com/gregjones/httpcache/diskcache".
type SizeLimitedDiskCache struct {
	maxSize     uint64
	currentSize uint64
	fileInfos   []fileInfo
	diskv       *diskv.Diskv
	mutex       sync.Mutex
}

// Get returns the response corresponding to key if present.
func (c *SizeLimitedDiskCache) Get(key string) (resp []byte, ok bool) {
	key = keyToFilename(key)
	resp, err := c.diskv.Read(key)
	if err != nil {
		return []byte{}, false
	}
	return resp, true
}

// Set saves a response to the cache as key, and updates the stored
// fileInfos and currentSize. This function can cause cache evictions.
func (c *SizeLimitedDiskCache) Set(key string, resp []byte) {
	key = keyToFilename(key)
	size := uint64(len(resp))
	c.trimDiskCache(size)
	err := c.diskv.WriteStream(key, bytes.NewReader(resp), true)
	if err == nil {
		c.appendFileInfo(fileInfo{name: key, size: size, modTime: time.Now()})
	} else {
		klog.Errorf("Error disk cache entry (%s):  [%v]", key, err)
	}
}

// Delete removes the response with key from the cache, updating the
// caches fileInfos and the caches currentSize.
func (c *SizeLimitedDiskCache) Delete(key string) {
	key = keyToFilename(key)
	err := c.diskv.Erase(key)
	if err == nil {
		c.removeFileInfo(key)
	} else {
		klog.Errorf("Error deleting entry (%s) from disk cache:  [%v]", key, err)
	}
}

// Passing the size of a file that is going to be added, evicts entries
// from the cache until the current cache size plus the passed size is below
// the maximum cache disk size. Can update the cache fileInfos and currentSize
// if evictions occur.
func (c *SizeLimitedDiskCache) trimDiskCache(size uint64) {
	for ((c.currentSize + size) > c.maxSize) && (len(c.fileInfos) > 0) {
		key := c.fileInfos[0].name // Evict oldest file by modification time
		err := c.diskv.Erase(key)
		if err == nil {
			c.removeFileInfo(key)
		} else {
			// If disk cache eviction fails, just exit.
			klog.Errorf("Error trimming disk cache: [%v]\n", err)
			break
		}
	}
}

// IMPORTANT: This function MUST be the same as
// "github.com/gregjones/httpcache/diskcache.keyToFilename()" in
// order to maintain compatibility with existing http disk caches.
func keyToFilename(key string) string {
	h := md5.New()
	io.WriteString(h, key)
	return hex.EncodeToString(h.Sum(nil))
}

// NewSizeLimitedDiskCache returns a new Cache up to "maxSize" number
// of total file bytes in the cache. Uses the provided Diskv as
// underlying storage. Walks the entire tree of files in the cache.
// Can cause evictions from the cache if the initial size of the cache
// is beyond "maxSize".
func NewSizeLimitedDiskCache(maxSize uint64, cacheDir string) *SizeLimitedDiskCache {
	begin := time.Now()
	klog.V(7).Infof("Creating http disk cache, max size: %d\n", maxSize)
	// Walk the cache files to store initial file metadata and initial cache size.
	fileInfos, cacheSize, err := initExistingCache(cacheDir)
	if err != nil {
		klog.Errorf("Error during disk cache walk: [%v]\n", err)
	}
	cache := &SizeLimitedDiskCache{
		maxSize:     maxSize,
		currentSize: cacheSize,
		fileInfos:   fileInfos,
		diskv: diskv.New(diskv.Options{
			PathPerm: os.FileMode(0750),
			FilePerm: os.FileMode(0660),
			BasePath: cacheDir,
			TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
		}),
	}
	numInitialFiles := len(fileInfos)
	klog.V(7).Infof("Before evictions, disk cache initially contains %d files, total size: %d",
		numInitialFiles, cacheSize)
	beforeCacheTrim := time.Now()
	cache.trimDiskCache(uint64(0)) // Ensure the initial disk cache size is below maxSize.
	numFilesEvicted := numInitialFiles - len(cache.fileInfos)
	finish := time.Now()
	klog.V(7).Infof("%d files evicted from cache, eviction time: %s",
		numFilesEvicted, finish.Sub(beforeCacheTrim))
	klog.V(7).Infof("total cache initialization time: %s", finish.Sub(begin))
	return cache
}

// initExistCache returns the slice of fileInfo for each of the files
// in the existing cache directory, as well as the total size of
// all these files. Sorts the fileInfo slice by ascending
// file modification time (oldest files first). Returns an error
// if one occurred during filepath.Walk.
func initExistingCache(cacheDir string) ([]fileInfo, uint64, error) {
	fileInfos := []fileInfo{}
	currentSize := uint64(0)
	err := filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		fileSize := uint64(info.Size())
		currentSize += fileSize
		fi := fileInfo{
			name:    info.Name(),
			size:    fileSize,
			modTime: info.ModTime(),
		}
		fileInfos = append(fileInfos, fi)
		return nil
	})
	sort.Sort(byFileModTime(fileInfos)) // Sort by ascending modTime
	return fileInfos, currentSize, err
}

// fileInfo stores relevant information for files in the disk cache.
// This structure stores the size of each file to determine the
// size of the entire disk cache, and it stores the modTime to
// determine the oldest file to evict from the cache when necessary.
// The name is the filename, which should be an MD5 hash (see
// keyToFilename()).
type fileInfo struct {
	name    string
	size    uint64
	modTime time.Time
}

// Sort cache fileInfos by ascending last modification time.
type byFileModTime []fileInfo

func (fi byFileModTime) Len() int {
	return len(fi)
}

func (fi byFileModTime) Less(i, j int) bool {
	return fi[i].modTime.Before(fi[j].modTime)
}

func (fi byFileModTime) Swap(i, j int) {
	fi[i], fi[j] = fi[j], fi[i]
}

// fileInfoIndex returns the index in the slice where the fileInfo
// exists, and true if found. Returns unused integer and false if not found.
func (c *SizeLimitedDiskCache) fileInfoIndex(key string) (int, bool) {
	for i := 0; i < len(c.fileInfos); i++ {
		if key == c.fileInfos[i].name {
			return i, true
		}
	}
	return 0, false
}

// removeFileInfo removes the fileInfo struct from the c.fileInfos
// slice for the item with the name equals key. Returns true if
// the items was found and deleted; false otherwise. Updates the
// cache current size.
func (c *SizeLimitedDiskCache) removeFileInfo(key string) bool {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	i, exists := c.fileInfoIndex(key)
	if exists {
		c.currentSize -= c.fileInfos[i].size
		c.fileInfos = append(c.fileInfos[:i], c.fileInfos[i+1:]...)
	}
	return exists
}

// appendFileInfo adds the passed fileInfo struct to the end of
// the c.fileInfos slice, assuming this file is more recent than
// all others. Before appending, any duplicates (by "name") are
// first removed. Ensures no duplicate fileInfos exist in c.fileInfos.
// Updates the cache currentSize.
func (c *SizeLimitedDiskCache) appendFileInfo(fi fileInfo) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	// Before appending the fileInfo, remove possible duplicate.
	if i, duplicate := c.fileInfoIndex(fi.name); duplicate {
		c.currentSize -= c.fileInfos[i].size
		c.fileInfos = append(c.fileInfos[:i], c.fileInfos[i+1:]...)
	}
	c.currentSize += fi.size
	c.fileInfos = append(c.fileInfos, fi)
}
