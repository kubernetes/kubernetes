// +build !windows

package godirwalk

import (
	"os"
	"syscall"
	"unsafe"
)

// MinimumScratchBufferSize specifies the minimum size of the scratch buffer
// that ReadDirents, ReadDirnames, Scanner, and Walk will use when reading file
// entries from the operating system. During program startup it is initialized
// to the result from calling `os.Getpagesize()` for non Windows environments,
// and 0 for Windows.
var MinimumScratchBufferSize = os.Getpagesize()

func newScratchBuffer() []byte { return make([]byte, MinimumScratchBufferSize) }

func readDirents(osDirname string, scratchBuffer []byte) ([]*Dirent, error) {
	var entries []*Dirent
	var workBuffer []byte

	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}
	fd := int(dh.Fd())

	if len(scratchBuffer) < MinimumScratchBufferSize {
		scratchBuffer = newScratchBuffer()
	}

	var sde syscall.Dirent
	for {
		if len(workBuffer) == 0 {
			n, err := syscall.ReadDirent(fd, scratchBuffer)
			// n, err := unix.ReadDirent(fd, scratchBuffer)
			if err != nil {
				if err == syscall.EINTR /* || err == unix.EINTR */ {
					continue
				}
				_ = dh.Close()
				return nil, err
			}
			if n <= 0 { // end of directory: normal exit
				if err = dh.Close(); err != nil {
					return nil, err
				}
				return entries, nil
			}
			workBuffer = scratchBuffer[:n] // trim work buffer to number of bytes read
		}

		copy((*[unsafe.Sizeof(syscall.Dirent{})]byte)(unsafe.Pointer(&sde))[:], workBuffer)
		workBuffer = workBuffer[reclen(&sde):] // advance buffer for next iteration through loop

		if inoFromDirent(&sde) == 0 {
			continue // inode set to 0 indicates an entry that was marked as deleted
		}

		nameSlice := nameFromDirent(&sde)
		nameLength := len(nameSlice)

		if nameLength == 0 || (nameSlice[0] == '.' && (nameLength == 1 || (nameLength == 2 && nameSlice[1] == '.'))) {
			continue
		}

		childName := string(nameSlice)
		mt, err := modeTypeFromDirent(&sde, osDirname, childName)
		if err != nil {
			_ = dh.Close()
			return nil, err
		}
		entries = append(entries, &Dirent{name: childName, path: osDirname, modeType: mt})
	}
}

func readDirnames(osDirname string, scratchBuffer []byte) ([]string, error) {
	var entries []string
	var workBuffer []byte
	var sde *syscall.Dirent

	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}
	fd := int(dh.Fd())

	if len(scratchBuffer) < MinimumScratchBufferSize {
		scratchBuffer = newScratchBuffer()
	}

	for {
		if len(workBuffer) == 0 {
			n, err := syscall.ReadDirent(fd, scratchBuffer)
			// n, err := unix.ReadDirent(fd, scratchBuffer)
			if err != nil {
				if err == syscall.EINTR /* || err == unix.EINTR */ {
					continue
				}
				_ = dh.Close()
				return nil, err
			}
			if n <= 0 { // end of directory: normal exit
				if err = dh.Close(); err != nil {
					return nil, err
				}
				return entries, nil
			}
			workBuffer = scratchBuffer[:n] // trim work buffer to number of bytes read
		}

		sde = (*syscall.Dirent)(unsafe.Pointer(&workBuffer[0])) // point entry to first syscall.Dirent in buffer
		// Handle first entry in the work buffer.
		workBuffer = workBuffer[reclen(sde):] // advance buffer for next iteration through loop

		if inoFromDirent(sde) == 0 {
			continue // inode set to 0 indicates an entry that was marked as deleted
		}

		nameSlice := nameFromDirent(sde)
		nameLength := len(nameSlice)

		if nameLength == 0 || (nameSlice[0] == '.' && (nameLength == 1 || (nameLength == 2 && nameSlice[1] == '.'))) {
			continue
		}

		entries = append(entries, string(nameSlice))
	}
}
