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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBasicCacheOperations(t *testing.T) {
	assert := assert.New(t)
	var file1 = []byte("abcd")
	var file2 = []byte("1234")
	var file3 = []byte("fghijklmno")
	testCacheSize := int64(1000)
	testCacheDir, err := ioutil.TempDir("", "cache")
	defer os.RemoveAll(testCacheDir)
	assert.NoError(err)
	cache := NewCompactableDiskCache(testCacheDir, testCacheSize)
	// Check basic structure variables are set.
	assert.Equal(testCacheDir, cache.cacheDir)
	assert.Equal(testCacheSize, cache.cacheSize)
	// Check "Get()" for non-existent key
	_, found := cache.Get("non-existent")
	assert.False(found, "Non-existent file should not be found in cache")
	// Set file1 into the disk cache, and validate we can "Get()" it.
	cache.Set("file1", file1)
	actualFile1, found := cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	assert.Equal(file1, actualFile1)
	// Set file2 into the disk cache, and validate we can "Get()" it.
	cache.Set("file2", file2)
	actualFile2, found := cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	assert.Equal(file2, actualFile2)
	// Set file3 into the disk cache, and validate we can "Get()" it
	// and the other files.
	cache.Set("file3", file3)
	_, found = cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	_, found = cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	actualFile3, found := cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")
	assert.Equal(file3, actualFile3)
	// Delete file2, and validate it is no longer in the cache, while
	// file1 and file3 are still in the cache.
	cache.Delete("file2")
	_, found = cache.Get("file2")
	assert.False(found, "Expected 'file2' to NOT be found in cache")
	_, found = cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	_, found = cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")
}

func TestDiskCacheCompaction(t *testing.T) {
	assert := assert.New(t)
	testCacheDir, err := ioutil.TempDir("", "compaction-cache")
	defer os.RemoveAll(testCacheDir)
	assert.NoError(err)
	var file1 = []byte("abcdefghijklmno") // 15 chars
	var file2 = []byte("1234567890")      // 10 chars
	var file3 = []byte("fghijklm")        // 8 chars
	var file4 = []byte("0987654321")      // 10 chars
	fullFilePath1 := filepath.Join(testCacheDir, sanitize("file1"))
	err = ioutil.WriteFile(fullFilePath1, file1, 0644)
	assert.NoError(err)
	fullFilePath2 := filepath.Join(testCacheDir, sanitize("file2"))
	err = ioutil.WriteFile(fullFilePath2, file2, 0644)
	assert.NoError(err)
	fullFilePath3 := filepath.Join(testCacheDir, sanitize("file3"))
	err = ioutil.WriteFile(fullFilePath3, file3, 0644)
	assert.NoError(err)
	fullFilePath4 := filepath.Join(testCacheDir, sanitize("file4"))
	err = ioutil.WriteFile(fullFilePath4, file4, 0644)
	assert.NoError(err)
	// Files by oldest modTime: file2, file4, file1, file3
	now := time.Now()
	setFileModTime(t, fullFilePath1, now.Add(5*time.Hour))
	setFileModTime(t, fullFilePath2, now.Add(-10*time.Hour))
	setFileModTime(t, fullFilePath3, now.Add(10*time.Hour))
	setFileModTime(t, fullFilePath4, now.Add(-5*time.Hour))
	cacheSize := len(file1) + len(file2) + len(file3) + len(file4)
	// Set the cache size threshold above the sizes of all files.
	actualNumEvictedFiles, err := compactDiskCache(testCacheDir, int64(cacheSize+10))
	assert.NoError(err)
	assert.Equal(0, actualNumEvictedFiles)
	assert.FileExists(fullFilePath1, "Expected file1 NOT to be evicted")
	assert.FileExists(fullFilePath2, "Expected file2 NOT to be evicted")
	assert.FileExists(fullFilePath3, "Expected file3 NOT to be evicted")
	assert.FileExists(fullFilePath3, "Expected file4 NOT to be evicted")
	// Set the cache size threshold small enough to evict two files.
	// Files 2 and 4 should be evicted since they are the oldest.
	actualNumEvictedFiles, err = compactDiskCache(testCacheDir, int64(cacheSize-15))
	assert.NoError(err)
	assert.Equal(2, actualNumEvictedFiles)
	assert.FileExists(fullFilePath1, "Expected file1 NOT to be evicted")
	assert.NoFileExists(fullFilePath2, "Expected file2 to be evicted")
	assert.FileExists(fullFilePath3, "Expected file3 NOT to be evicted")
	assert.NoFileExists(fullFilePath4, "Expected file4 to be evicted")
}

func TestShouldCompactCache(t *testing.T) {
	assert := assert.New(t)
	testCacheDir, err := ioutil.TempDir("", "cache")
	defer os.RemoveAll(testCacheDir)
	assert.NoError(err)
	var file1 = []byte("abcd")
	fullFilePath := filepath.Join(testCacheDir, sanitize("file1"))
	err = ioutil.WriteFile(fullFilePath, file1, 0644)
	assert.NoError(err)
	// First, check shouldCompactCache first-time non-existent
	// compaction file.
	compactionFile := filepath.Join(testCacheDir, lastCompactionFilename)
	assert.NoFileExists(compactionFile, "Zero-length compaction file should not exist yet.")
	actual := shouldCompactCache(compactionFile)
	assert.False(actual, "First create of compaction file should NOT trigger compaction")
	assert.FileExists(compactionFile, "Zero-length compaction file should be created.")
	// Next, check shoulCompactCache with file mod time set to current time.
	now := time.Now()
	setFileModTime(t, compactionFile, now)
	actual = shouldCompactCache(compactionFile)
	assert.False(actual, "File with recent modTime should NOT trigger compaction")
	// Finally, check shoulCompactCache with file mod time set over a day ago.
	setFileModTime(t, compactionFile, now.Add(-25*time.Hour))
	actual = shouldCompactCache(compactionFile)
	assert.True(actual, "File modTime older than one day should trigger compaction")
}

func TestTouchFile(t *testing.T) {
	assert := assert.New(t)
	testCacheDir, err := ioutil.TempDir("", "cache")
	defer os.RemoveAll(testCacheDir)
	assert.NoError(err)
	var file1 = []byte("abcd")
	fullFilePath := filepath.Join(testCacheDir, sanitize("file1"))
	err = ioutil.WriteFile(fullFilePath, file1, 0644)
	assert.NoError(err)
	// First, check the set mod time from two days ago.
	twoDaysAgo := time.Now().Add(-48 * time.Hour)
	setFileModTime(t, fullFilePath, twoDaysAgo)
	fileInfo, err := os.Stat(fullFilePath)
	assert.NoError(err)
	assert.Equal((0 * time.Second), twoDaysAgo.Sub(fileInfo.ModTime()))
	// Next, check after touching the same file, the mod time
	// is very close to now.
	err = touchFile(fullFilePath)
	assert.NoError(err)
	fileInfo, err = os.Stat(fullFilePath)
	assert.NoError(err)
	assert.Greater((1 * time.Minute), time.Now().Sub(fileInfo.ModTime()))
}

func TestFileModTimeSorting(t *testing.T) {
	now := time.Now()
	info1 := createFileInfo("fake1", int64(10), now)
	info2 := createFileInfo("fake2", int64(20), now.Add(-2*time.Hour))
	info3 := createFileInfo("fake3", int64(30), now.Add(24*time.Hour))
	info4 := createFileInfo("fake4", int64(40), now.Add(-48*time.Hour))
	info5 := createFileInfo("fake5", int64(50), now.Add(16*time.Hour))
	fileInfos := []*fileInfo{info1, info2, info3, info4, info5}
	// Sort files by mod time.
	sortedFileInfos := sortFiles(fileInfos)
	assert := assert.New(t)
	// Validate fileInfo ordering: info4, info2, info1, info5, info3
	assert.Equal(info4, sortedFileInfos[0])
	assert.Equal(info2, sortedFileInfos[1])
	assert.Equal(info1, sortedFileInfos[2])
	assert.Equal(info5, sortedFileInfos[3])
	assert.Equal(info3, sortedFileInfos[4])
}

func createFileInfo(filePath string, fileSize int64, modTime time.Time) *fileInfo {
	return &fileInfo{
		filePath: filePath,
		fileSize: fileSize,
		modTime:  modTime,
	}
}

func setFileModTime(t *testing.T, filePath string, modTime time.Time) {
	err := os.Chtimes(filePath, modTime, modTime)
	assert := assert.New(t)
	assert.NoError(err)
}
