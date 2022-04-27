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
	"os"
	"path/filepath"
	"sort"
	"time"

	"golang.org/x/sync/singleflight"

	"github.com/gregjones/httpcache"
	"github.com/peterbourgon/diskv"
	"k8s.io/klog/v2"
)

const (
	oneDay                 = 24 * time.Hour
	lastCompactionFilename = "lastcompactiontime"
)

// CompactableDiskCache implements the httpcache.Cache interface. It delegates to
// sumDiskCache for Cache functions (Get, Set, Delete), and occasionally removes
// old cached files. The specified "cacheSize" is NOT a hard limit; the disk
// cache can breach this threshold. When cache compaction occurs, it will resize
// the disk cache below this "cacheSize".
//
// Trigger:
//
//	disk cache compactions are triggered when a file is accessed and the
//	last compaction occurred over a day ago. Every cache file access checks
//	when the last compaction occurred by checking the modification time
//	on an empty file in the cache -- "lastcompactiontime". File access
//	time for a cached file is not used directly because it is not well
//	supported by all OSes.
//
// Concurrency:
//
//	Within the same process: the goroutines of a single process are
//	coordinated using "sync/singleflight" functionality.
//
//	Among different processes: if (for example) a separate kubectl command
//	compacts the disk cache at the same time, separate processes can compete
//	to delete the same files. This code must be tolerant to attempting to
//	delete files already deleted.
type CompactableDiskCache struct {
	cacheDir  string
	cacheSize int64
	delegate  httpcache.Cache
}

// Get returns the []byte representation of a cached response and a bool
// set to true if the value isn't empty. This function can trigger a
// disk cache compaction which runs in a separate goroutine. Touches
// cached file to update file modification to match access time.
func (cdc *CompactableDiskCache) Get(key string) ([]byte, bool) {
	// If the file exists, update file modification time = access time.
	_ = touchFile(sanitize(key))
	go possiblyCompactDiskCache(cdc.cacheDir, cdc.cacheSize)
	return cdc.delegate.Get(key)
}

// Set stores the []byte representation of a response against a key.
func (cdc *CompactableDiskCache) Set(key string, responseBytes []byte) {
	cdc.delegate.Set(key, responseBytes)
}

// Delete removes the value associated with the key.
func (cdc *CompactableDiskCache) Delete(key string) {
	cdc.delegate.Delete(key)
}

// NewCompactableDiskCache returns a newly created CompactableDiskCache,
// which includes a newly created sumDiskCache delegate.
func NewCompactableDiskCache(cacheDir string, cacheSize int64) *CompactableDiskCache {
	return &CompactableDiskCache{
		cacheDir:  cacheDir,
		cacheSize: cacheSize,
		delegate: &sumDiskCache{
			disk: diskv.New(diskv.Options{
				PathPerm: os.FileMode(0750),
				FilePerm: os.FileMode(0660),
				BasePath: cacheDir,
				TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
			}),
		},
	}
}

// possiblyCompactDiskCache checks if the disk cache compaction
// trigger condition has occurred, and if it has, it runs (only
// one) compaction.
func possiblyCompactDiskCache(cacheDir string, cacheSize int64) {
	compactionFile := filepath.Join(cacheDir, lastCompactionFilename)
	if shouldCompactCache(compactionFile) {
		klog.V(7).Infoln("Disk cache compaction triggered")
		// "singleflight" ensures that only one disk compaction occurs at a time within
		// the current process.
		var requestGroup singleflight.Group
		begin := time.Now()
		numFilesEvicted, err, _ := requestGroup.Do("disk-cache-compaction", func() (interface{}, error) {
			return compactDiskCache(cacheDir, cacheSize)
		})
		finish := time.Now()
		if err != nil {
			klog.Errorf("Error during disk cache compaction: %s", err)
		}
		klog.V(7).Infof("%d files evicted from cache, compaction time: %s",
			numFilesEvicted.(int), finish.Sub(begin))
		// Set new, current modification time for "lastcompactiontime" file.
		if err := touchFile(compactionFile); err != nil {
			klog.Errorf("error touching last compaction file: %s", err)
		}
	}
}

// shouldCompactCache returns true if the disk cache compaction
// trigger condition is met; false otherwise. The trigger condition
// is if the modification time of the passed empty "lastcompactiontime"
// file is greater than one day. If the file is created for the first
// time, we will not trigger a compaction.
func shouldCompactCache(compactionFile string) bool {
	fileInfo, err := os.Stat(compactionFile)
	if err != nil {
		if os.IsNotExist(err) {
			// First time creating empty "lastcompactiontime" file.
			_, _ = os.Create(compactionFile)
		}
		return false
	}
	durationSinceLastChange := time.Now().Sub(fileInfo.ModTime())
	if durationSinceLastChange > oneDay {
		return true
	}
	return false
}

// touchFile updates the passed file's access time and modification
// time to the current time. Returns an error if one occurred while
// updating the file.
func touchFile(fullFilePath string) error {
	currentTime := time.Now()
	return os.Chtimes(fullFilePath, currentTime, currentTime)
}

// compactDiskCache removes old cache files (determined by modification time)
// if the current cache size is larger than "cacheSize". Walks the entire
// disk cache, and sorts files by ascending modification time. Removes files
// until the disk cache is compacted below the "cacheSize" threshold. Returns
// the number of files evicted from the cache and an error if one occurred.
func compactDiskCache(cacheDir string, cacheSize int64) (int, error) {
	currentCacheSize := int64(0)
	files := []*fileInfo{}
	// Walk the disk cache files to determine the cache size and file
	// modification times.
	err := filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		fileSize := info.Size()
		files = append(files, &fileInfo{
			filePath: path,
			modTime:  info.ModTime(),
			fileSize: fileSize,
		})
		currentCacheSize += fileSize
		return nil
	})
	if err != nil {
		return 0, err
	}
	// Only compact disk cache if it is beyond the "cacheSize" threshold. This
	// threshold is NOT a hard limit. The cache will breach this threshold, but
	// the compaction will return the disk cache size below the "cacheSize".
	klog.V(7).Infof("Disk cache current size: %d", currentCacheSize)
	klog.V(7).Infof("Disk cache threshold size: %d", cacheSize)
	numFilesEvicted := 0
	if currentCacheSize > cacheSize {
		klog.V(7).Infoln("Removing old cache files...")
		// Starting with the oldest files (by modification time), delete each
		// file until the disk cache size is below the "cacheSize" threshold.
		// Other processes may be deleting the same cache files at the same time.
		files = sortFiles(files)
		for i := 0; currentCacheSize > cacheSize && i < len(files); i++ {
			fileInfo := files[i]
			// Another process may have already deleted the file, so we only
			// log error, and we always decrement the currentCacheSize.
			err := os.Remove(fileInfo.filePath)
			currentCacheSize -= fileInfo.fileSize
			if err != nil {
				klog.Errorf("error deleting cached file: %s", err)
			} else {
				numFilesEvicted++
			}
		}
	}
	return numFilesEvicted, nil
}

// sortFiles returns a slice of *fileInfo sorted by increasing
// file modification time; oldest files (by modtime) are first.
func sortFiles(fileInfos []*fileInfo) []*fileInfo {
	// Sort the fileInfo's by ascending file modification time, so the
	// oldest files will be at the beginning of the slice.
	sort.Slice(fileInfos, func(i, j int) bool {
		return fileInfos[i].modTime.Before(fileInfos[j].modTime)
	})
	return fileInfos
}

// fileInfo aggregates pertinent information about a cached file.
type fileInfo struct {
	filePath string
	fileSize int64
	modTime  time.Time
}
