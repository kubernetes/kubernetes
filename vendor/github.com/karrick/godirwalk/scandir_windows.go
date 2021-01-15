// +build windows

package godirwalk

import (
	"fmt"
	"os"
)

// Scanner is an iterator to enumerate the contents of a directory.
type Scanner struct {
	osDirname string
	childName string
	dh        *os.File // dh is handle to open directory
	de        *Dirent
	err       error // err is the error associated with scanning directory
	childMode os.FileMode
}

// NewScanner returns a new directory Scanner that lazily enumerates the
// contents of a single directory.
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
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}
	scanner := &Scanner{
		osDirname: osDirname,
		dh:        dh,
	}
	return scanner, nil
}

// NewScannerWithScratchBuffer returns a new directory Scanner that lazily
// enumerates the contents of a single directory. On platforms other than
// Windows it uses the provided scratch buffer to read from the file system. On
// Windows the scratch buffer parameter is ignored.
func NewScannerWithScratchBuffer(osDirname string, scratchBuffer []byte) (*Scanner, error) {
	return NewScanner(osDirname)
}

// Dirent returns the current directory entry while scanning a directory.
func (s *Scanner) Dirent() (*Dirent, error) {
	if s.de == nil {
		s.de = &Dirent{
			name:     s.childName,
			path:     s.osDirname,
			modeType: s.childMode,
		}
	}
	return s.de, nil
}

// done is called when directory scanner unable to continue, with either the
// triggering error, or nil when there are simply no more entries to read from
// the directory.
func (s *Scanner) done(err error) {
	if s.dh == nil {
		return
	}

	if cerr := s.dh.Close(); err == nil {
		s.err = cerr
	}

	s.childName, s.osDirname = "", ""
	s.de, s.dh = nil, nil
}

// Err returns any error associated with scanning a directory. It is normal to
// call Err after Scan returns false, even though they both ensure Scanner
// resources are released. Do not call until done scanning a directory.
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

	fileinfos, err := s.dh.Readdir(1)
	if err != nil {
		s.done(err)
		return false
	}

	if l := len(fileinfos); l != 1 {
		s.done(fmt.Errorf("expected a single entry rather than %d", l))
		return false
	}

	fi := fileinfos[0]
	s.childMode = fi.Mode() & os.ModeType
	s.childName = fi.Name()
	return true
}
