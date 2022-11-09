//go:build !windows
// +build !windows

package godirwalk

import (
	"os"
	"syscall"
	"unsafe"
)

// Scanner is an iterator to enumerate the contents of a directory.
type Scanner struct {
	scratchBuffer []byte // read directory bytes from file system into this buffer
	workBuffer    []byte // points into scratchBuffer, from which we chunk out directory entries
	osDirname     string
	childName     string
	err           error    // err is the error associated with scanning directory
	statErr       error    // statErr is any error return while attempting to stat an entry
	dh            *os.File // used to close directory after done reading
	de            *Dirent  // most recently decoded directory entry
	sde           syscall.Dirent
	fd            int // file descriptor used to read entries from directory
}

// NewScanner returns a new directory Scanner that lazily enumerates
// the contents of a single directory. To prevent resource leaks,
// caller must invoke either the Scanner's Close or Err method after
// it has completed scanning a directory.
//
//     scanner, err := godirwalk.NewScanner(dirname)
//     if err != nil {
//         fatal("cannot scan directory: %s", err)
//     }
//
//     for scanner.Scan() {
//         dirent, err := scanner.Dirent()
//         if err != nil {
//             warning("cannot get dirent: %s", err)
//             continue
//         }
//         name := dirent.Name()
//         if name == "break" {
//             break
//         }
//         if name == "continue" {
//             continue
//         }
//         fmt.Printf("%v %v\n", dirent.ModeType(), dirent.Name())
//     }
//     if err := scanner.Err(); err != nil {
//         fatal("cannot scan directory: %s", err)
//     }
func NewScanner(osDirname string) (*Scanner, error) {
	return NewScannerWithScratchBuffer(osDirname, nil)
}

// NewScannerWithScratchBuffer returns a new directory Scanner that
// lazily enumerates the contents of a single directory. On platforms
// other than Windows it uses the provided scratch buffer to read from
// the file system. On Windows the scratch buffer is ignored. To
// prevent resource leaks, caller must invoke either the Scanner's
// Close or Err method after it has completed scanning a directory.
func NewScannerWithScratchBuffer(osDirname string, scratchBuffer []byte) (*Scanner, error) {
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}
	if len(scratchBuffer) < MinimumScratchBufferSize {
		scratchBuffer = newScratchBuffer()
	}
	scanner := &Scanner{
		scratchBuffer: scratchBuffer,
		osDirname:     osDirname,
		dh:            dh,
		fd:            int(dh.Fd()),
	}
	return scanner, nil
}

// Close releases resources associated with scanning a directory. Call
// either this or the Err method when the directory no longer needs to
// be scanned.
func (s *Scanner) Close() error {
	return s.Err()
}

// Dirent returns the current directory entry while scanning a directory.
func (s *Scanner) Dirent() (*Dirent, error) {
	if s.de == nil {
		s.de = &Dirent{name: s.childName, path: s.osDirname}
		s.de.modeType, s.statErr = modeTypeFromDirent(&s.sde, s.osDirname, s.childName)
	}
	return s.de, s.statErr
}

// done is called when directory scanner unable to continue, with either the
// triggering error, or nil when there are simply no more entries to read from
// the directory.
func (s *Scanner) done(err error) {
	if s.dh == nil {
		return
	}

	s.err = err

	if err = s.dh.Close(); s.err == nil {
		s.err = err
	}

	s.osDirname, s.childName = "", ""
	s.scratchBuffer, s.workBuffer = nil, nil
	s.dh, s.de, s.statErr = nil, nil, nil
	s.sde = syscall.Dirent{}
	s.fd = 0
}

// Err returns any error associated with scanning a directory. It is
// normal to call Err after Scan returns false, even though they both
// ensure Scanner resources are released. Call either this or the
// Close method when the directory no longer needs to be scanned.
func (s *Scanner) Err() error {
	s.done(nil)
	return s.err
}

// Name returns the base name of the current directory entry while scanning a
// directory.
func (s *Scanner) Name() string { return s.childName }

// Scan potentially reads and then decodes the next directory entry from the
// file system.
//
// When it returns false, this releases resources used by the Scanner then
// returns any error associated with closing the file system directory resource.
func (s *Scanner) Scan() bool {
	if s.dh == nil {
		return false
	}

	s.de = nil

	for {
		// When the work buffer has nothing remaining to decode, we need to load
		// more data from disk.
		if len(s.workBuffer) == 0 {
			n, err := syscall.ReadDirent(s.fd, s.scratchBuffer)
			// n, err := unix.ReadDirent(s.fd, s.scratchBuffer)
			if err != nil {
				if err == syscall.EINTR /* || err == unix.EINTR */ {
					continue
				}
				s.done(err) // any other error forces a stop
				return false
			}
			if n <= 0 { // end of directory: normal exit
				s.done(nil)
				return false
			}
			s.workBuffer = s.scratchBuffer[:n] // trim work buffer to number of bytes read
		}

		// point entry to first syscall.Dirent in buffer
		copy((*[unsafe.Sizeof(syscall.Dirent{})]byte)(unsafe.Pointer(&s.sde))[:], s.workBuffer)
		s.workBuffer = s.workBuffer[reclen(&s.sde):] // advance buffer for next iteration through loop

		if inoFromDirent(&s.sde) == 0 {
			continue // inode set to 0 indicates an entry that was marked as deleted
		}

		nameSlice := nameFromDirent(&s.sde)
		nameLength := len(nameSlice)

		if nameLength == 0 || (nameSlice[0] == '.' && (nameLength == 1 || (nameLength == 2 && nameSlice[1] == '.'))) {
			continue
		}

		s.childName = string(nameSlice)
		return true
	}
}
