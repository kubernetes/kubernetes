package fs

// Usage of disk information
type Usage struct {
	Inodes int64
	Size   int64
}

// DiskUsage counts the number of inodes and disk usage for the resources under
// path.
func DiskUsage(roots ...string) (Usage, error) {
	return diskUsage(roots...)
}
