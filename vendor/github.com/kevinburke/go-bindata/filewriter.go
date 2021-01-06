package bindata

import (
	"bytes"
	"crypto/sha256"
	"io"
	"os"
)

func diffAndWrite(filename string, data []byte, mode os.FileMode) error {
	// If the file has the same contents as data, try to avoid a write.
	f, err := os.Open(filename)
	if err != nil {
		return safefileWriteFile(filename, data, mode)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return safefileWriteFile(filename, data, mode)
	}
	if err := f.Close(); err != nil {
		return safefileWriteFile(filename, data, mode)
	}
	h2 := sha256.New()
	if _, err := h2.Write(data); err != nil {
		return safefileWriteFile(filename, data, mode)
	}
	if bytes.Equal(h.Sum(nil), h2.Sum(nil)) {
		return nil
	}
	return safefileWriteFile(filename, data, mode)
}
