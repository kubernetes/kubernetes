/*
Copyright 2025 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"hash/crc32"
)

// The max length for ntfs, ext4, xfs and btrfs.
const maxFileNameLength = 255

// Shorten a file name to size allowed by the most common filesystems.
// If the filename is too long, cut it + add a short hash (crc32) that makes it unique.
// Note that the input should be a single file / directory name, not a path
// composed of several directories.
func ShortenFileName(filename string) string {
	if len(filename) <= maxFileNameLength {
		return filename
	}

	hash := crc32.ChecksumIEEE([]byte(filename))
	hashString := fmt.Sprintf("%x", hash)
	hashLen := len(hashString)

	return fmt.Sprintf("%s-%s", filename[:maxFileNameLength-1-hashLen], hashString)
}
