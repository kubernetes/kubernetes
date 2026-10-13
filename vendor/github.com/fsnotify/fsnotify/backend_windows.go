//go:build windows

// Windows backend based on ReadDirectoryChangesW()
//
// https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-readdirectorychangesw

package fsnotify

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/fsnotify/fsnotify/internal"
	"golang.org/x/sys/windows"
)

type readDirChangesW struct {
	Events chan Event
	Errors chan error

	port  windows.Handle // Handle to completion port
	input chan *input    // Inputs to the reader are sent on this channel
	done  chan chan<- error

	mu      sync.Mutex // Protects access to watches, closed
	watches watchMap   // Map of watches (key: i-number)
	closed  bool       // Set to true when Close() is first called
}

var defaultBufferSize = 50

func newBackend(ev chan Event, errs chan error) (backend, error) {
	port, err := windows.CreateIoCompletionPort(windows.InvalidHandle, 0, 0, 0)
	if err != nil {
		return nil, os.NewSyscallError("CreateIoCompletionPort", err)
	}
	w := &readDirChangesW{
		Events:  ev,
		Errors:  errs,
		port:    port,
		watches: make(watchMap),
		input:   make(chan *input, 1),
		done:    make(chan chan<- error, 1),
	}
	go w.readEvents()
	return w, nil
}

func (w *readDirChangesW) isClosed() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.closed
}

func (w *readDirChangesW) sendEvent(name, renamedFrom string, mask uint64) bool {
	if mask == 0 {
		return false
	}

	event := w.newEvent(name, uint32(mask))
	event.renamedFrom = renamedFrom
	select {
	case ch := <-w.done:
		w.done <- ch
	case w.Events <- event:
	}
	return true
}

// Returns true if the error was sent, or false if watcher is closed.
func (w *readDirChangesW) sendError(err error) bool {
	if err == nil {
		return true
	}
	select {
	case <-w.done:
		return false
	case w.Errors <- err:
		return true
	}
}

func (w *readDirChangesW) Close() error {
	if w.isClosed() {
		return nil
	}

	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()

	// Send "done" message to the reader goroutine
	ch := make(chan error)
	w.done <- ch
	if err := w.wakeupReader(); err != nil {
		return err
	}
	return <-ch
}

func (w *readDirChangesW) Add(name string) error { return w.AddWith(name) }

func (w *readDirChangesW) AddWith(name string, opts ...addOpt) error {
	if w.isClosed() {
		return ErrClosed
	}
	if debug {
		fmt.Fprintf(os.Stderr, "FSNOTIFY_DEBUG: %s  AddWith(%q)\n",
			time.Now().Format("15:04:05.000000000"), filepath.ToSlash(name))
	}

	with := getOptions(opts...)
	if !w.xSupports(with.op) {
		return fmt.Errorf("%w: %s", xErrUnsupported, with.op)
	}
	if with.bufsize < 4096 {
		return fmt.Errorf("fsnotify.WithBufferSize: buffer size cannot be smaller than 4096 bytes")
	}

	in := &input{
		op:      opAddWatch,
		path:    filepath.Clean(name),
		flags:   sysFSALLEVENTS,
		reply:   make(chan error),
		bufsize: with.bufsize,
	}
	w.input <- in
	if err := w.wakeupReader(); err != nil {
		return err
	}
	return <-in.reply
}

func (w *readDirChangesW) Remove(name string) error {
	if w.isClosed() {
		return nil
	}
	if debug {
		fmt.Fprintf(os.Stderr, "FSNOTIFY_DEBUG: %s  Remove(%q)\n",
			time.Now().Format("15:04:05.000000000"), filepath.ToSlash(name))
	}

	in := &input{
		op:    opRemoveWatch,
		path:  filepath.Clean(name),
		reply: make(chan error),
	}
	w.input <- in
	if err := w.wakeupReader(); err != nil {
		return err
	}
	return <-in.reply
}

func (w *readDirChangesW) WatchList() []string {
	if w.isClosed() {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	entries := make([]string, 0, len(w.watches))
	for _, entry := range w.watches {
		for _, watchEntry := range entry {
			for name := range watchEntry.names {
				entries = append(entries, filepath.Join(watchEntry.path, name))
			}
			// the directory itself is being watched
			if watchEntry.mask != 0 {
				entries = append(entries, watchEntry.path)
			}
		}
	}

	return entries
}

// These options are from the old golang.org/x/exp/winfsnotify, where you could
// add various options to the watch. This has long since been removed.
//
// The "sys" in the name is misleading as they're not part of any "system".
//
// This should all be removed at some point, and just use windows.FILE_NOTIFY_*
const (
	sysFSALLEVENTS  = 0xfff
	sysFSCREATE     = 0x100
	sysFSDELETE     = 0x200
	sysFSDELETESELF = 0x400
	sysFSMODIFY     = 0x2
	sysFSMOVE       = 0xc0
	sysFSMOVEDFROM  = 0x40
	sysFSMOVEDTO    = 0x80
	sysFSMOVESELF   = 0x800
	sysFSIGNORED    = 0x8000
)

func (w *readDirChangesW) newEvent(name string, mask uint32) Event {
	e := Event{Name: name}
	if mask&sysFSCREATE == sysFSCREATE || mask&sysFSMOVEDTO == sysFSMOVEDTO {
		e.Op |= Create
	}
	if mask&sysFSDELETE == sysFSDELETE || mask&sysFSDELETESELF == sysFSDELETESELF {
		e.Op |= Remove
	}
	if mask&sysFSMODIFY == sysFSMODIFY {
		e.Op |= Write
	}
	if mask&sysFSMOVE == sysFSMOVE || mask&sysFSMOVESELF == sysFSMOVESELF || mask&sysFSMOVEDFROM == sysFSMOVEDFROM {
		e.Op |= Rename
	}
	return e
}

const (
	opAddWatch = iota
	opRemoveWatch
)

const (
	provisional uint64 = 1 << (32 + iota)
)

type input struct {
	op      int
	path    string
	flags   uint32
	bufsize int
	reply   chan error
}

type inode struct {
	handle windows.Handle
	volume uint32
	index  uint64
}

type watch struct {
	ov      windows.Overlapped
	ino     *inode            // i-number
	recurse bool              // Recursive watch?
	path    string            // Directory path
	mask    uint64            // Directory itself is being watched with these notify flags
	names   map[string]uint64 // Map of names being watched and their notify flags
	rename  string            // Remembers the old name while renaming a file
	buf     []byte            // buffer, allocated later
}

type (
	indexMap map[uint64]*watch
	watchMap map[uint32]indexMap
)

func (w *readDirChangesW) wakeupReader() error {
	err := windows.PostQueuedCompletionStatus(w.port, 0, 0, nil)
	if err != nil {
		return os.NewSyscallError("PostQueuedCompletionStatus", err)
	}
	return nil
}

func (w *readDirChangesW) getDir(pathname string) (dir string, err error) {
	attr, err := windows.GetFileAttributes(windows.StringToUTF16Ptr(pathname))
	if err != nil {
		return "", os.NewSyscallError("GetFileAttributes", err)
	}
	if attr&windows.FILE_ATTRIBUTE_DIRECTORY != 0 {
		dir = pathname
	} else {
		dir, _ = filepath.Split(pathname)
		dir = filepath.Clean(dir)
	}
	return
}

func (w *readDirChangesW) getIno(path string) (ino *inode, err error) {
	h, err := windows.CreateFile(windows.StringToUTF16Ptr(path),
		windows.FILE_LIST_DIRECTORY,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING,
		windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OVERLAPPED, 0)
	if err != nil {
		return nil, os.NewSyscallError("CreateFile", err)
	}

	var fi windows.ByHandleFileInformation
	err = windows.GetFileInformationByHandle(h, &fi)
	if err != nil {
		windows.CloseHandle(h)
		return nil, os.NewSyscallError("GetFileInformationByHandle", err)
	}
	ino = &inode{
		handle: h,
		volume: fi.VolumeSerialNumber,
		index:  uint64(fi.FileIndexHigh)<<32 | uint64(fi.FileIndexLow),
	}
	return ino, nil
}

// Must run within the I/O thread.
func (m watchMap) get(ino *inode) *watch {
	if i := m[ino.volume]; i != nil {
		return i[ino.index]
	}
	return nil
}

// Must run within the I/O thread.
func (m watchMap) set(ino *inode, watch *watch) {
	i := m[ino.volume]
	if i == nil {
		i = make(indexMap)
		m[ino.volume] = i
	}
	i[ino.index] = watch
}

// Must run within the I/O thread.
func (w *readDirChangesW) addWatch(pathname string, flags uint64, bufsize int) error {
	pathname, recurse := recursivePath(pathname)

	dir, err := w.getDir(pathname)
	if err != nil {
		return err
	}

	ino, err := w.getIno(dir)
	if err != nil {
		return err
	}
	w.mu.Lock()
	watchEntry := w.watches.get(ino)
	w.mu.Unlock()
	if watchEntry == nil {
		_, err := windows.CreateIoCompletionPort(ino.handle, w.port, 0, 0)
		if err != nil {
			windows.CloseHandle(ino.handle)
			return os.NewSyscallError("CreateIoCompletionPort", err)
		}
		watchEntry = &watch{
			ino:     ino,
			path:    dir,
			names:   make(map[string]uint64),
			recurse: recurse,
			buf:     make([]byte, bufsize),
		}
		w.mu.Lock()
		w.watches.set(ino, watchEntry)
		w.mu.Unlock()
		flags |= provisional
	} else {
		windows.CloseHandle(ino.handle)
	}
	if pathname == dir {
		watchEntry.mask |= flags
	} else {
		watchEntry.names[filepath.Base(pathname)] |= flags
	}

	err = w.startRead(watchEntry)
	if err != nil {
		return err
	}

	if pathname == dir {
		watchEntry.mask &= ^provisional
	} else {
		watchEntry.names[filepath.Base(pathname)] &= ^provisional
	}
	return nil
}

// Must run within the I/O thread.
func (w *readDirChangesW) remWatch(pathname string) error {
	pathname, recurse := recursivePath(pathname)

	dir, err := w.getDir(pathname)
	if err != nil {
		return err
	}
	ino, err := w.getIno(dir)
	if err != nil {
		return err
	}

	w.mu.Lock()
	watch := w.watches.get(ino)
	w.mu.Unlock()

	if recurse && !watch.recurse {
		return fmt.Errorf("can't use \\... with non-recursive watch %q", pathname)
	}

	err = windows.CloseHandle(ino.handle)
	if err != nil {
		w.sendError(os.NewSyscallError("CloseHandle", err))
	}
	if watch == nil {
		return fmt.Errorf("%w: %s", ErrNonExistentWatch, pathname)
	}
	if pathname == dir {
		w.sendEvent(watch.path, "", watch.mask&sysFSIGNORED)
		watch.mask = 0
	} else {
		name := filepath.Base(pathname)
		w.sendEvent(filepath.Join(watch.path, name), "", watch.names[name]&sysFSIGNORED)
		delete(watch.names, name)
	}

	return w.startRead(watch)
}

// Must run within the I/O thread.
func (w *readDirChangesW) deleteWatch(watch *watch) {
	for name, mask := range watch.names {
		if mask&provisional == 0 {
			w.sendEvent(filepath.Join(watch.path, name), "", mask&sysFSIGNORED)
		}
		delete(watch.names, name)
	}
	if watch.mask != 0 {
		if watch.mask&provisional == 0 {
			w.sendEvent(watch.path, "", watch.mask&sysFSIGNORED)
		}
		watch.mask = 0
	}
}

// Must run within the I/O thread.
func (w *readDirChangesW) startRead(watch *watch) error {
	err := windows.CancelIo(watch.ino.handle)
	if err != nil {
		w.sendError(os.NewSyscallError("CancelIo", err))
		w.deleteWatch(watch)
	}
	mask := w.toWindowsFlags(watch.mask)
	for _, m := range watch.names {
		mask |= w.toWindowsFlags(m)
	}
	if mask == 0 {
		err := windows.CloseHandle(watch.ino.handle)
		if err != nil {
			w.sendError(os.NewSyscallError("CloseHandle", err))
		}
		w.mu.Lock()
		delete(w.watches[watch.ino.volume], watch.ino.index)
		w.mu.Unlock()
		return nil
	}

	// We need to pass the array, rather than the slice.
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&watch.buf))
	rdErr := windows.ReadDirectoryChanges(watch.ino.handle,
		(*byte)(unsafe.Pointer(hdr.Data)), uint32(hdr.Len),
		watch.recurse, mask, nil, &watch.ov, 0)
	if rdErr != nil {
		err := os.NewSyscallError("ReadDirectoryChanges", rdErr)
		if rdErr == windows.ERROR_ACCESS_DENIED && watch.mask&provisional == 0 {
			// Watched directory was probably removed
			w.sendEvent(watch.path, "", watch.mask&sysFSDELETESELF)
			err = nil
		}
		w.deleteWatch(watch)
		w.startRead(watch)
		return err
	}
	return nil
}

// readEvents reads from the I/O completion port, converts the
// received events into Event objects and sends them via the Events channel.
// Entry point to the I/O thread.
func (w *readDirChangesW) readEvents() {
	var (
		n   uint32
		key uintptr
		ov  *windows.Overlapped
	)
	runtime.LockOSThread()

	for {
		// This error is handled after the watch == nil check below.
		qErr := windows.GetQueuedCompletionStatus(w.port, &n, &key, &ov, windows.INFINITE)

		watch := (*watch)(unsafe.Pointer(ov))
		if watch == nil {
			select {
			case ch := <-w.done:
				w.mu.Lock()
				var indexes []indexMap
				for _, index := range w.watches {
					indexes = append(indexes, index)
				}
				w.mu.Unlock()
				for _, index := range indexes {
					for _, watch := range index {
						w.deleteWatch(watch)
						w.startRead(watch)
					}
				}

				err := windows.CloseHandle(w.port)
				if err != nil {
					err = os.NewSyscallError("CloseHandle", err)
				}
				close(w.Events)
				close(w.Errors)
				ch <- err
				return
			case in := <-w.input:
				switch in.op {
				case opAddWatch:
					in.reply <- w.addWatch(in.path, uint64(in.flags), in.bufsize)
				case opRemoveWatch:
					in.reply <- w.remWatch(in.path)
				}
			default:
			}
			continue
		}

		switch qErr {
		case nil:
			// No error
		case windows.ERROR_MORE_DATA:
			if watch == nil {
				w.sendError(errors.New("ERROR_MORE_DATA has unexpectedly null lpOverlapped buffer"))
			} else {
				// The i/o succeeded but the buffer is full.
				// In theory we should be building up a full packet.
				// In practice we can get away with just carrying on.
				n = uint32(unsafe.Sizeof(watch.buf))
			}
		case windows.ERROR_ACCESS_DENIED:
			// Watched directory was probably removed
			w.sendEvent(watch.path, "", watch.mask&sysFSDELETESELF)
			w.deleteWatch(watch)
			w.startRead(watch)
			continue
		case windows.ERROR_OPERATION_ABORTED:
			// CancelIo was called on this handle
			continue
		default:
			w.sendError(os.NewSyscallError("GetQueuedCompletionPort", qErr))
			continue
		}

		var offset uint32
		for {
			if n == 0 {
				w.sendError(ErrEventOverflow)
				break
			}

			// Point "raw" to the event in the buffer
			raw := (*windows.FileNotifyInformation)(unsafe.Pointer(&watch.buf[offset]))

			// Create a buf that is the size of the path name
			size := int(raw.FileNameLength / 2)
			var buf []uint16
			// TODO: Use unsafe.Slice in Go 1.17; https://stackoverflow.com/questions/51187973
			sh := (*reflect.SliceHeader)(unsafe.Pointer(&buf))
			sh.Data = uintptr(unsafe.Pointer(&raw.FileName))
			sh.Len = size
			sh.Cap = size
			name := windows.UTF16ToString(buf)
			fullname := filepath.Join(watch.path, name)

			if debug {
				internal.Debug(fullname, raw.Action)
			}

			var mask uint64
			switch raw.Action {
			case windows.FILE_ACTION_REMOVED:
				mask = sysFSDELETESELF
			case windows.FILE_ACTION_MODIFIED:
				mask = sysFSMODIFY
			case windows.FILE_ACTION_RENAMED_OLD_NAME:
				watch.rename = name
			case windows.FILE_ACTION_RENAMED_NEW_NAME:
				// Update saved path of all sub-watches.
				old := filepath.Join(watch.path, watch.rename)
				w.mu.Lock()
				for _, watchMap := range w.watches {
					for _, ww := range watchMap {
						if strings.HasPrefix(ww.path, old) {
							ww.path = filepath.Join(fullname, strings.TrimPrefix(ww.path, old))
						}
					}
				}
				w.mu.Unlock()

				if watch.names[watch.rename] != 0 {
					watch.names[name] |= watch.names[watch.rename]
					delete(watch.names, watch.rename)
					mask = sysFSMOVESELF
				}
			}

			if raw.Action != windows.FILE_ACTION_RENAMED_NEW_NAME {
				w.sendEvent(fullname, "", watch.names[name]&mask)
			}
			if raw.Action == windows.FILE_ACTION_REMOVED {
				w.sendEvent(fullname, "", watch.names[name]&sysFSIGNORED)
				delete(watch.names, name)
			}

			if watch.rename != "" && raw.Action == windows.FILE_ACTION_RENAMED_NEW_NAME {
				w.sendEvent(fullname, filepath.Join(watch.path, watch.rename), watch.mask&w.toFSnotifyFlags(raw.Action))
			} else {
				w.sendEvent(fullname, "", watch.mask&w.toFSnotifyFlags(raw.Action))
			}

			if raw.Action == windows.FILE_ACTION_RENAMED_NEW_NAME {
				w.sendEvent(filepath.Join(watch.path, watch.rename), "", watch.names[name]&mask)
			}

			// Move to the next event in the buffer
			if raw.NextEntryOffset == 0 {
				break
			}
			offset += raw.NextEntryOffset

			// Error!
			if offset >= n {
				//lint:ignore ST1005 Windows should be capitalized
				w.sendError(errors.New("Windows system assumed buffer larger than it is, events have likely been missed"))
				break
			}
		}

		if err := w.startRead(watch); err != nil {
			w.sendError(err)
		}
	}
}

func (w *readDirChangesW) toWindowsFlags(mask uint64) uint32 {
	var m uint32
	if mask&sysFSMODIFY != 0 {
		m |= windows.FILE_NOTIFY_CHANGE_LAST_WRITE
	}
	if mask&(sysFSMOVE|sysFSCREATE|sysFSDELETE) != 0 {
		m |= windows.FILE_NOTIFY_CHANGE_FILE_NAME | windows.FILE_NOTIFY_CHANGE_DIR_NAME
	}
	return m
}

func (w *readDirChangesW) toFSnotifyFlags(action uint32) uint64 {
	switch action {
	case windows.FILE_ACTION_ADDED:
		return sysFSCREATE
	case windows.FILE_ACTION_REMOVED:
		return sysFSDELETE
	case windows.FILE_ACTION_MODIFIED:
		return sysFSMODIFY
	case windows.FILE_ACTION_RENAMED_OLD_NAME:
		return sysFSMOVEDFROM
	case windows.FILE_ACTION_RENAMED_NEW_NAME:
		return sysFSMOVEDTO
	}
	return 0
}

func (w *readDirChangesW) xSupports(op Op) bool {
	if op.Has(xUnportableOpen) || op.Has(xUnportableRead) ||
		op.Has(xUnportableCloseWrite) || op.Has(xUnportableCloseRead) {
		return false
	}
	return true
}
