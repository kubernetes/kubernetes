package utils

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"path/filepath"
	"syscall"
)

const (
	exitSignalOffset = 128
)

// GenerateRandomName returns a new name joined with a prefix.  This size
// specified is used to truncate the randomly generated value
func GenerateRandomName(prefix string, size int) (string, error) {
	id := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, id); err != nil {
		return "", err
	}
	if size > 64 {
		size = 64
	}
	return prefix + hex.EncodeToString(id)[:size], nil
}

// ResolveRootfs ensures that the current working directory is
// not a symlink and returns the absolute path to the rootfs
func ResolveRootfs(uncleanRootfs string) (string, error) {
	rootfs, err := filepath.Abs(uncleanRootfs)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(rootfs)
}

// ExitStatus returns the correct exit status for a process based on if it
// was signaled or exited cleanly
func ExitStatus(status syscall.WaitStatus) int {
	if status.Signaled() {
		return exitSignalOffset + int(status.Signal())
	}
	return status.ExitStatus()
}

// WriteJSON writes the provided struct v to w using standard json marshaling
func WriteJSON(w io.Writer, v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}
