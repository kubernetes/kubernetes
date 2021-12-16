package out

import (
	"os"
	"path"
)

// LazyFile is an io.WriteCloser which defers creation of the file it is supposed to write in
// till the first call to its write function in order to prevent creation of file, if no write
// is supposed to happen.
type LazyFile struct {
	// FileName is path to the file to which genny will write.
	FileName string
	file     *os.File
}

// Close closes the file if it is created. Returns nil if no file is created.
func (lw *LazyFile) Close() error {
	if lw.file != nil {
		return lw.file.Close()
	}
	return nil
}

// Write writes to the specified file and creates the file first time it is called.
func (lw *LazyFile) Write(p []byte) (int, error) {
	if lw.file == nil {
		err := os.MkdirAll(path.Dir(lw.FileName), 0755)
		if err != nil {
			return 0, err
		}
		lw.file, err = os.Create(lw.FileName)
		if err != nil {
			return 0, err
		}
	}
	return lw.file.Write(p)
}
