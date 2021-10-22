package awstesting

// DiscardAt is an io.WriteAt that discards
// the requested bytes to be written
type DiscardAt struct{}

// WriteAt discards the given []byte slice and returns len(p) bytes
// as having been written at the given offset. It will never return an error.
func (d DiscardAt) WriteAt(p []byte, off int64) (n int, err error) {
	return len(p), nil
}
