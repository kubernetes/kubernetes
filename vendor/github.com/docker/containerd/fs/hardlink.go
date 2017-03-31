package fs

import "os"

// GetLinkSource returns a path for the given name and
// file info to its link source in the provided inode
// map. If the given file name is not in the map and
// has other links, it is added to the inode map
// to be a source for other link locations.
func GetLinkSource(name string, fi os.FileInfo, inodes map[uint64]string) (string, error) {
	return getHardLink(name, fi, inodes)
}
