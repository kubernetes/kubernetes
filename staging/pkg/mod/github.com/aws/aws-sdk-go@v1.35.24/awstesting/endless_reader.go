package awstesting

// EndlessReader is an io.Reader that will always return
// that bytes have been read.
type EndlessReader struct{}

// Read will report that it has read len(p) bytes in p.
// The content in the []byte will be unmodified.
// This will never return an error.
func (e EndlessReader) Read(p []byte) (int, error) {
	return len(p), nil
}
