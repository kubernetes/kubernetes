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
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestEmptyNewCache(t *testing.T) {
	// Initialize the disk cache on top of an empty file system, and
	// validate the cache is empty.
	testCacheMaxSize := uint64(1001)
	testCacheDir1, err := ioutil.TempDir("", "cache1")
	defer os.RemoveAll(testCacheDir1)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewSizeLimitedDiskCache(testCacheMaxSize, testCacheDir1)
	assert := assert.New(t)
	assert.Equal(testCacheMaxSize, cache.maxSize)
	assert.Equal(uint64(0), cache.currentSize)
	assert.Equal(0, len(cache.fileInfos))

	// Create a second cache with cached files already present. Create the
	// disk cache, initializing with the existing files. Validate the
	// initialized cache values.
	var file1 = []byte("abcd")
	var file2 = []byte("1234")
	var file3 = []byte("fghijklmno")
	testCacheDir2, err := ioutil.TempDir("", "cache2")
	defer os.RemoveAll(testCacheDir2)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(testCacheDir2, keyToFilename("file1")), file1, 0644)
	assert.NoError(err)
	err = ioutil.WriteFile(filepath.Join(testCacheDir2, keyToFilename("file2")), file2, 0644)
	assert.NoError(err)
	err = ioutil.WriteFile(filepath.Join(testCacheDir2, keyToFilename("file3")), file3, 0644)
	assert.NoError(err)
	cache = NewSizeLimitedDiskCache(testCacheMaxSize, testCacheDir2)
	assert.Equal(3, len(cache.fileInfos))
	expectedCacheSize := uint64(len(file1) + len(file2) + len(file3))
	assert.Equal(expectedCacheSize, cache.currentSize)
	_, found := cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	_, found = cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	_, found = cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")

	// Create a third cache with cached files already present and over the
	// cache size limit. Create the cache and validate the cache size limit
	// is correctly enforced with cache evictions.
	var file4 = []byte("0123456789") // size: 10
	var file5 = []byte("987654321")  // size: 9
	var file6 = []byte("6543210")    // size: 7
	testCacheDir3, err := ioutil.TempDir("", "cache3")
	defer os.RemoveAll(testCacheDir3)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(testCacheDir3, keyToFilename("file4")), file4, 0644)
	assert.NoError(err)
	err = ioutil.WriteFile(filepath.Join(testCacheDir3, keyToFilename("file5")), file5, 0644)
	assert.NoError(err)
	err = ioutil.WriteFile(filepath.Join(testCacheDir3, keyToFilename("file6")), file6, 0644)
	assert.NoError(err)
	cacheMaxSize := uint64(len(file4) + len(file5) + len(file6) - 1)
	cache = NewSizeLimitedDiskCache(cacheMaxSize, testCacheDir3)
	// file4 should be evicted upon initialization; file5 and file6 remain.
	assert.Equal(2, len(cache.fileInfos))
	expectedCacheSize = uint64(len(file5) + len(file6))
	assert.Equal(expectedCacheSize, cache.currentSize)
	_, found = cache.Get("file4")
	assert.False(found, "Expected 'file4' to be evicted in cache")
	_, found = cache.Get("file5")
	assert.True(found, "Expected 'file5' to be found in cache")
	_, found = cache.Get("file6")
	assert.True(found, "Expected 'file6' to be found in cache")
}

func TestFileModTimeSorting(t *testing.T) {
	now := time.Now()
	info1 := createFileInfo("fake1", uint64(10), now)
	info2 := createFileInfo("fake2", uint64(20), now.Add(-2*time.Hour))
	info3 := createFileInfo("fake3", uint64(30), now.Add(24*time.Hour))
	info4 := createFileInfo("fake4", uint64(40), now.Add(-48*time.Hour))
	info5 := createFileInfo("fake5", uint64(50), now.Add(16*time.Hour))
	fileInfos := []fileInfo{info1, info2, info3, info4, info5}
	sort.Sort(byFileModTime(fileInfos))
	assert := assert.New(t)
	// File modification time ordering: info4, info2, info1, info5, info3
	assert.Equal(info4, fileInfos[0])
	assert.Equal(info2, fileInfos[1])
	assert.Equal(info1, fileInfos[2])
	assert.Equal(info5, fileInfos[3])
	assert.Equal(info3, fileInfos[4])
}

func createFileInfo(name string, size uint64, modTime time.Time) fileInfo {
	return fileInfo{
		name:    name,
		size:    size,
		modTime: modTime,
	}
}

func TestBasicCacheOperations(t *testing.T) {
	var file1 = []byte("abcd")
	var file2 = []byte("1234")
	var file3 = []byte("fghijklmno")

	testCacheMaxSize := uint64(1000)
	testCacheDir, err := ioutil.TempDir("", "cache")
	defer os.RemoveAll(testCacheDir)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewSizeLimitedDiskCache(testCacheMaxSize, testCacheDir)
	assert := assert.New(t)
	_, found := cache.Get("non-existent")
	assert.False(found, "Non-existent file should not be found in cache")

	cache.Set("file1", file1)
	actualFile1, found := cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	assert.Equal(file1, actualFile1)
	assert.Equal(1, len(cache.fileInfos))
	assert.Equal(uint64(len(file1)), cache.currentSize)

	cache.Set("file2", file2)
	actualFile2, found := cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	assert.Equal(file2, actualFile2)
	assert.Equal(2, len(cache.fileInfos))
	expectedCacheSize := uint64(len(file1) + len(file2))
	assert.Equal(expectedCacheSize, cache.currentSize)

	cache.Set("file3", file3)
	_, found = cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	_, found = cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	actualFile3, found := cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")
	assert.Equal(file3, actualFile3)
	assert.Equal(3, len(cache.fileInfos))
	expectedCacheSize = uint64(len(file1) + len(file2) + len(file3))
	assert.Equal(expectedCacheSize, cache.currentSize)

	cache.Delete("file2")
	_, found = cache.Get("file2")
	assert.False(found, "Expected 'file2' to NOT be found in cache")
	_, found = cache.Get("file1")
	assert.True(found, "Expected 'file1' to be found in cache")
	_, found = cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")
	assert.Equal(2, len(cache.fileInfos))
	expectedCacheSize = uint64(len(file1) + len(file3))
	assert.Equal(expectedCacheSize, cache.currentSize)
}

func TestCacheEvictions(t *testing.T) {
	var file1 = []byte("0123456789") // size: 10
	var file2 = []byte("98765432")   // size: 8
	var file3 = []byte("43210")      // size: 5

	testCacheMaxSize := uint64(22)
	testCacheDir, err := ioutil.TempDir("", "cache")
	defer os.RemoveAll(testCacheDir)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewSizeLimitedDiskCache(testCacheMaxSize, testCacheDir)
	assert := assert.New(t)

	// Add file1 and file2. Set the file modification time on file1
	// to be earlier than file2.
	cache.Set("file1", file1)
	cache.Set("file2", file2)
	assert.Equal(2, len(cache.fileInfos))
	expectedCacheSize := uint64(len(file1) + len(file2))
	assert.Equal(expectedCacheSize, cache.currentSize)
	currentTime := time.Now()
	cache.fileInfos[0].modTime = currentTime.Add(-1 * time.Hour)
	cache.fileInfos[1].modTime = currentTime
	// Adding file3 causes eviction, since the cache size (23) exceeds
	// the max cache size (22).
	cache.Set("file3", file3)
	assert.Equal(2, len(cache.fileInfos))
	_, found := cache.Get("file1")
	assert.False(found, "Expected 'file1' to be evicted cache")
	_, found = cache.Get("file2")
	assert.True(found, "Expected 'file2' to be found in cache")
	_, found = cache.Get("file3")
	assert.True(found, "Expected 'file3' to be found in cache")
	expectedCacheSize = uint64(len(file2) + len(file3))
	assert.Equal(expectedCacheSize, cache.currentSize)
}

func TestFileInfo(t *testing.T) {
	cache := &SizeLimitedDiskCache{
		maxSize:     uint64(10000),
		currentSize: uint64(0),
		fileInfos:   []fileInfo{},
	}
	assert := assert.New(t)
	_, found := cache.fileInfoIndex("non-existent")
	assert.False(found, "Non-existent file info should not be found")
	now := time.Now()
	var fake1Size = uint64(10)
	info1 := createFileInfo("fake1", fake1Size, now)
	cache.appendFileInfo(info1)
	var fake2Size = uint64(20)
	info2 := createFileInfo("fake2", fake2Size, now.Add(1*time.Hour))
	cache.appendFileInfo(info2)
	assert.Equal(2, len(cache.fileInfos))
	assert.Equal(fake1Size+fake2Size, cache.currentSize)
	var fake3Size = uint64(30)
	info3 := createFileInfo("fake3", fake3Size, now.Add(2*time.Hour))
	cache.appendFileInfo(info3)
	assert.Equal(3, len(cache.fileInfos))
	assert.Equal(fake1Size+fake2Size+fake3Size, cache.currentSize)
	// Duplicate key "fake2", fileInfo should get replaced when appended.
	var duplicateSize = uint64(100)
	infoDuplicate := createFileInfo("fake2", duplicateSize, now.Add(3*time.Hour))
	cache.appendFileInfo(infoDuplicate) // Replace previous "fake3"
	assert.Equal(3, len(cache.fileInfos))
	assert.Equal(fake1Size+fake3Size+duplicateSize, cache.currentSize)
	removed := cache.removeFileInfo("fake1")
	assert.True(removed, "fake1 fileInfo should have been removed from fileInfos")
	assert.Equal(2, len(cache.fileInfos))
	assert.Equal(fake3Size+duplicateSize, cache.currentSize)
	assert.Equal(info3, cache.fileInfos[0])
	assert.Equal(infoDuplicate, cache.fileInfos[1])
}
