// +build windows

package winio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"syscall"
	"unicode/utf16"
)

//sys backupRead(h syscall.Handle, b []byte, bytesRead *uint32, abort bool, processSecurity bool, context *uintptr) (err error) = BackupRead
//sys backupWrite(h syscall.Handle, b []byte, bytesWritten *uint32, abort bool, processSecurity bool, context *uintptr) (err error) = BackupWrite

const (
	BackupData = uint32(iota + 1)
	BackupEaData
	BackupSecurity
	BackupAlternateData
	BackupLink
	BackupPropertyData
	BackupObjectId
	BackupReparseData
	BackupSparseBlock
	BackupTxfsData
)

const (
	StreamSparseAttributes = uint32(8)
)

const (
	WRITE_DAC              = 0x40000
	WRITE_OWNER            = 0x80000
	ACCESS_SYSTEM_SECURITY = 0x1000000
)

// BackupHeader represents a backup stream of a file.
type BackupHeader struct {
	Id         uint32 // The backup stream ID
	Attributes uint32 // Stream attributes
	Size       int64  // The size of the stream in bytes
	Name       string // The name of the stream (for BackupAlternateData only).
	Offset     int64  // The offset of the stream in the file (for BackupSparseBlock only).
}

type win32StreamId struct {
	StreamId   uint32
	Attributes uint32
	Size       uint64
	NameSize   uint32
}

// BackupStreamReader reads from a stream produced by the BackupRead Win32 API and produces a series
// of BackupHeader values.
type BackupStreamReader struct {
	r         io.Reader
	bytesLeft int64
}

// NewBackupStreamReader produces a BackupStreamReader from any io.Reader.
func NewBackupStreamReader(r io.Reader) *BackupStreamReader {
	return &BackupStreamReader{r, 0}
}

// Next returns the next backup stream and prepares for calls to Read(). It skips the remainder of the current stream if
// it was not completely read.
func (r *BackupStreamReader) Next() (*BackupHeader, error) {
	if r.bytesLeft > 0 {
		if s, ok := r.r.(io.Seeker); ok {
			// Make sure Seek on io.SeekCurrent sometimes succeeds
			// before trying the actual seek.
			if _, err := s.Seek(0, io.SeekCurrent); err == nil {
				if _, err = s.Seek(r.bytesLeft, io.SeekCurrent); err != nil {
					return nil, err
				}
				r.bytesLeft = 0
			}
		}
		if _, err := io.Copy(ioutil.Discard, r); err != nil {
			return nil, err
		}
	}
	var wsi win32StreamId
	if err := binary.Read(r.r, binary.LittleEndian, &wsi); err != nil {
		return nil, err
	}
	hdr := &BackupHeader{
		Id:         wsi.StreamId,
		Attributes: wsi.Attributes,
		Size:       int64(wsi.Size),
	}
	if wsi.NameSize != 0 {
		name := make([]uint16, int(wsi.NameSize/2))
		if err := binary.Read(r.r, binary.LittleEndian, name); err != nil {
			return nil, err
		}
		hdr.Name = syscall.UTF16ToString(name)
	}
	if wsi.StreamId == BackupSparseBlock {
		if err := binary.Read(r.r, binary.LittleEndian, &hdr.Offset); err != nil {
			return nil, err
		}
		hdr.Size -= 8
	}
	r.bytesLeft = hdr.Size
	return hdr, nil
}

// Read reads from the current backup stream.
func (r *BackupStreamReader) Read(b []byte) (int, error) {
	if r.bytesLeft == 0 {
		return 0, io.EOF
	}
	if int64(len(b)) > r.bytesLeft {
		b = b[:r.bytesLeft]
	}
	n, err := r.r.Read(b)
	r.bytesLeft -= int64(n)
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	} else if r.bytesLeft == 0 && err == nil {
		err = io.EOF
	}
	return n, err
}

// BackupStreamWriter writes a stream compatible with the BackupWrite Win32 API.
type BackupStreamWriter struct {
	w         io.Writer
	bytesLeft int64
}

// NewBackupStreamWriter produces a BackupStreamWriter on top of an io.Writer.
func NewBackupStreamWriter(w io.Writer) *BackupStreamWriter {
	return &BackupStreamWriter{w, 0}
}

// WriteHeader writes the next backup stream header and prepares for calls to Write().
func (w *BackupStreamWriter) WriteHeader(hdr *BackupHeader) error {
	if w.bytesLeft != 0 {
		return fmt.Errorf("missing %d bytes", w.bytesLeft)
	}
	name := utf16.Encode([]rune(hdr.Name))
	wsi := win32StreamId{
		StreamId:   hdr.Id,
		Attributes: hdr.Attributes,
		Size:       uint64(hdr.Size),
		NameSize:   uint32(len(name) * 2),
	}
	if hdr.Id == BackupSparseBlock {
		// Include space for the int64 block offset
		wsi.Size += 8
	}
	if err := binary.Write(w.w, binary.LittleEndian, &wsi); err != nil {
		return err
	}
	if len(name) != 0 {
		if err := binary.Write(w.w, binary.LittleEndian, name); err != nil {
			return err
		}
	}
	if hdr.Id == BackupSparseBlock {
		if err := binary.Write(w.w, binary.LittleEndian, hdr.Offset); err != nil {
			return err
		}
	}
	w.bytesLeft = hdr.Size
	return nil
}

// Write writes to the current backup stream.
func (w *BackupStreamWriter) Write(b []byte) (int, error) {
	if w.bytesLeft < int64(len(b)) {
		return 0, fmt.Errorf("too many bytes by %d", int64(len(b))-w.bytesLeft)
	}
	n, err := w.w.Write(b)
	w.bytesLeft -= int64(n)
	return n, err
}

// BackupFileReader provides an io.ReadCloser interface on top of the BackupRead Win32 API.
type BackupFileReader struct {
	f               *os.File
	includeSecurity bool
	ctx             uintptr
}

// NewBackupFileReader returns a new BackupFileReader from a file handle. If includeSecurity is true,
// Read will attempt to read the security descriptor of the file.
func NewBackupFileReader(f *os.File, includeSecurity bool) *BackupFileReader {
	r := &BackupFileReader{f, includeSecurity, 0}
	return r
}

// Read reads a backup stream from the file by calling the Win32 API BackupRead().
func (r *BackupFileReader) Read(b []byte) (int, error) {
	var bytesRead uint32
	err := backupRead(syscall.Handle(r.f.Fd()), b, &bytesRead, false, r.includeSecurity, &r.ctx)
	if err != nil {
		return 0, &os.PathError{"BackupRead", r.f.Name(), err}
	}
	runtime.KeepAlive(r.f)
	if bytesRead == 0 {
		return 0, io.EOF
	}
	return int(bytesRead), nil
}

// Close frees Win32 resources associated with the BackupFileReader. It does not close
// the underlying file.
func (r *BackupFileReader) Close() error {
	if r.ctx != 0 {
		backupRead(syscall.Handle(r.f.Fd()), nil, nil, true, false, &r.ctx)
		runtime.KeepAlive(r.f)
		r.ctx = 0
	}
	return nil
}

// BackupFileWriter provides an io.WriteCloser interface on top of the BackupWrite Win32 API.
type BackupFileWriter struct {
	f               *os.File
	includeSecurity bool
	ctx             uintptr
}

// NewBackupFileWriter returns a new BackupFileWriter from a file handle. If includeSecurity is true,
// Write() will attempt to restore the security descriptor from the stream.
func NewBackupFileWriter(f *os.File, includeSecurity bool) *BackupFileWriter {
	w := &BackupFileWriter{f, includeSecurity, 0}
	return w
}

// Write restores a portion of the file using the provided backup stream.
func (w *BackupFileWriter) Write(b []byte) (int, error) {
	var bytesWritten uint32
	err := backupWrite(syscall.Handle(w.f.Fd()), b, &bytesWritten, false, w.includeSecurity, &w.ctx)
	if err != nil {
		return 0, &os.PathError{"BackupWrite", w.f.Name(), err}
	}
	runtime.KeepAlive(w.f)
	if int(bytesWritten) != len(b) {
		return int(bytesWritten), errors.New("not all bytes could be written")
	}
	return len(b), nil
}

// Close frees Win32 resources associated with the BackupFileWriter. It does not
// close the underlying file.
func (w *BackupFileWriter) Close() error {
	if w.ctx != 0 {
		backupWrite(syscall.Handle(w.f.Fd()), nil, nil, true, false, &w.ctx)
		runtime.KeepAlive(w.f)
		w.ctx = 0
	}
	return nil
}

// OpenForBackup opens a file or directory, potentially skipping access checks if the backup
// or restore privileges have been acquired.
//
// If the file opened was a directory, it cannot be used with Readdir().
func OpenForBackup(path string, access uint32, share uint32, createmode uint32) (*os.File, error) {
	winPath, err := syscall.UTF16FromString(path)
	if err != nil {
		return nil, err
	}
	h, err := syscall.CreateFile(&winPath[0], access, share, nil, createmode, syscall.FILE_FLAG_BACKUP_SEMANTICS|syscall.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		err = &os.PathError{Op: "open", Path: path, Err: err}
		return nil, err
	}
	return os.NewFile(uintptr(h), path), nil
}
