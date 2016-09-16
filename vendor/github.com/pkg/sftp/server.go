package sftp

// sftp server counterpart

import (
	"encoding"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/pkg/errors"
)

const (
	sftpServerWorkerCount = 8
)

// Server is an SSH File Transfer Protocol (sftp) server.
// This is intended to provide the sftp subsystem to an ssh server daemon.
// This implementation currently supports most of sftp server protocol version 3,
// as specified at http://tools.ietf.org/html/draft-ietf-secsh-filexfer-02
type Server struct {
	serverConn
	debugStream   io.Writer
	readOnly      bool
	pktChan       chan rxPacket
	openFiles     map[string]*os.File
	openFilesLock sync.RWMutex
	handleCount   int
	maxTxPacket   uint32
}

func (svr *Server) nextHandle(f *os.File) string {
	svr.openFilesLock.Lock()
	defer svr.openFilesLock.Unlock()
	svr.handleCount++
	handle := strconv.Itoa(svr.handleCount)
	svr.openFiles[handle] = f
	return handle
}

func (svr *Server) closeHandle(handle string) error {
	svr.openFilesLock.Lock()
	defer svr.openFilesLock.Unlock()
	if f, ok := svr.openFiles[handle]; ok {
		delete(svr.openFiles, handle)
		return f.Close()
	}

	return syscall.EBADF
}

func (svr *Server) getHandle(handle string) (*os.File, bool) {
	svr.openFilesLock.RLock()
	defer svr.openFilesLock.RUnlock()
	f, ok := svr.openFiles[handle]
	return f, ok
}

type serverRespondablePacket interface {
	encoding.BinaryUnmarshaler
	id() uint32
	respond(svr *Server) error
}

// NewServer creates a new Server instance around the provided streams, serving
// content from the root of the filesystem.  Optionally, ServerOption
// functions may be specified to further configure the Server.
//
// A subsequent call to Serve() is required to begin serving files over SFTP.
func NewServer(rwc io.ReadWriteCloser, options ...ServerOption) (*Server, error) {
	s := &Server{
		serverConn: serverConn{
			conn: conn{
				Reader:      rwc,
				WriteCloser: rwc,
			},
		},
		debugStream: ioutil.Discard,
		pktChan:     make(chan rxPacket, sftpServerWorkerCount),
		openFiles:   make(map[string]*os.File),
		maxTxPacket: 1 << 15,
	}

	for _, o := range options {
		if err := o(s); err != nil {
			return nil, err
		}
	}

	return s, nil
}

// A ServerOption is a function which applies configuration to a Server.
type ServerOption func(*Server) error

// WithDebug enables Server debugging output to the supplied io.Writer.
func WithDebug(w io.Writer) ServerOption {
	return func(s *Server) error {
		s.debugStream = w
		return nil
	}
}

// ReadOnly configures a Server to serve files in read-only mode.
func ReadOnly() ServerOption {
	return func(s *Server) error {
		s.readOnly = true
		return nil
	}
}

type rxPacket struct {
	pktType  fxp
	pktBytes []byte
}

// Up to N parallel servers
func (svr *Server) sftpServerWorker() error {
	for p := range svr.pktChan {
		var pkt interface {
			encoding.BinaryUnmarshaler
			id() uint32
		}
		var readonly = true
		switch p.pktType {
		case ssh_FXP_INIT:
			pkt = &sshFxInitPacket{}
		case ssh_FXP_LSTAT:
			pkt = &sshFxpLstatPacket{}
		case ssh_FXP_OPEN:
			pkt = &sshFxpOpenPacket{}
			// readonly handled specially below
		case ssh_FXP_CLOSE:
			pkt = &sshFxpClosePacket{}
		case ssh_FXP_READ:
			pkt = &sshFxpReadPacket{}
		case ssh_FXP_WRITE:
			pkt = &sshFxpWritePacket{}
			readonly = false
		case ssh_FXP_FSTAT:
			pkt = &sshFxpFstatPacket{}
		case ssh_FXP_SETSTAT:
			pkt = &sshFxpSetstatPacket{}
			readonly = false
		case ssh_FXP_FSETSTAT:
			pkt = &sshFxpFsetstatPacket{}
			readonly = false
		case ssh_FXP_OPENDIR:
			pkt = &sshFxpOpendirPacket{}
		case ssh_FXP_READDIR:
			pkt = &sshFxpReaddirPacket{}
		case ssh_FXP_REMOVE:
			pkt = &sshFxpRemovePacket{}
			readonly = false
		case ssh_FXP_MKDIR:
			pkt = &sshFxpMkdirPacket{}
			readonly = false
		case ssh_FXP_RMDIR:
			pkt = &sshFxpRmdirPacket{}
			readonly = false
		case ssh_FXP_REALPATH:
			pkt = &sshFxpRealpathPacket{}
		case ssh_FXP_STAT:
			pkt = &sshFxpStatPacket{}
		case ssh_FXP_RENAME:
			pkt = &sshFxpRenamePacket{}
			readonly = false
		case ssh_FXP_READLINK:
			pkt = &sshFxpReadlinkPacket{}
		case ssh_FXP_SYMLINK:
			pkt = &sshFxpSymlinkPacket{}
			readonly = false
		case ssh_FXP_EXTENDED:
			pkt = &sshFxpExtendedPacket{}
		default:
			return errors.Errorf("unhandled packet type: %s", p.pktType)
		}
		if err := pkt.UnmarshalBinary(p.pktBytes); err != nil {
			return err
		}

		// handle FXP_OPENDIR specially
		switch pkt := pkt.(type) {
		case *sshFxpOpenPacket:
			readonly = pkt.readonly()
		case *sshFxpExtendedPacket:
			readonly = pkt.SpecificPacket.readonly()
		}

		// If server is operating read-only and a write operation is requested,
		// return permission denied
		if !readonly && svr.readOnly {
			if err := svr.sendError(pkt, syscall.EPERM); err != nil {
				return errors.Wrap(err, "failed to send read only packet response")
			}
			continue
		}

		if err := handlePacket(svr, pkt); err != nil {
			return err
		}
	}
	return nil
}

func handlePacket(s *Server, p interface{}) error {
	switch p := p.(type) {
	case *sshFxInitPacket:
		return s.sendPacket(sshFxVersionPacket{sftpProtocolVersion, nil})
	case *sshFxpStatPacket:
		// stat the requested file
		info, err := os.Stat(p.Path)
		if err != nil {
			return s.sendError(p, err)
		}
		return s.sendPacket(sshFxpStatResponse{
			ID:   p.ID,
			info: info,
		})
	case *sshFxpLstatPacket:
		// stat the requested file
		info, err := os.Lstat(p.Path)
		if err != nil {
			return s.sendError(p, err)
		}
		return s.sendPacket(sshFxpStatResponse{
			ID:   p.ID,
			info: info,
		})
	case *sshFxpFstatPacket:
		f, ok := s.getHandle(p.Handle)
		if !ok {
			return s.sendError(p, syscall.EBADF)
		}

		info, err := f.Stat()
		if err != nil {
			return s.sendError(p, err)
		}

		return s.sendPacket(sshFxpStatResponse{
			ID:   p.ID,
			info: info,
		})
	case *sshFxpMkdirPacket:
		// TODO FIXME: ignore flags field
		err := os.Mkdir(p.Path, 0755)
		return s.sendError(p, err)
	case *sshFxpRmdirPacket:
		err := os.Remove(p.Path)
		return s.sendError(p, err)
	case *sshFxpRemovePacket:
		err := os.Remove(p.Filename)
		return s.sendError(p, err)
	case *sshFxpRenamePacket:
		err := os.Rename(p.Oldpath, p.Newpath)
		return s.sendError(p, err)
	case *sshFxpSymlinkPacket:
		err := os.Symlink(p.Targetpath, p.Linkpath)
		return s.sendError(p, err)
	case *sshFxpClosePacket:
		return s.sendError(p, s.closeHandle(p.Handle))
	case *sshFxpReadlinkPacket:
		f, err := os.Readlink(p.Path)
		if err != nil {
			return s.sendError(p, err)
		}

		return s.sendPacket(sshFxpNamePacket{
			ID: p.ID,
			NameAttrs: []sshFxpNameAttr{{
				Name:     f,
				LongName: f,
				Attrs:    emptyFileStat,
			}},
		})

	case *sshFxpRealpathPacket:
		f, err := filepath.Abs(p.Path)
		if err != nil {
			return s.sendError(p, err)
		}
		f = filepath.Clean(f)
		return s.sendPacket(sshFxpNamePacket{
			ID: p.ID,
			NameAttrs: []sshFxpNameAttr{{
				Name:     f,
				LongName: f,
				Attrs:    emptyFileStat,
			}},
		})
	case *sshFxpOpendirPacket:
		return sshFxpOpenPacket{
			ID:     p.ID,
			Path:   p.Path,
			Pflags: ssh_FXF_READ,
		}.respond(s)
	case *sshFxpReadPacket:
		f, ok := s.getHandle(p.Handle)
		if !ok {
			return s.sendError(p, syscall.EBADF)
		}

		data := make([]byte, clamp(p.Len, s.maxTxPacket))
		n, err := f.ReadAt(data, int64(p.Offset))
		if err != nil && (err != io.EOF || n == 0) {
			return s.sendError(p, err)
		}
		return s.sendPacket(sshFxpDataPacket{
			ID:     p.ID,
			Length: uint32(n),
			Data:   data[:n],
		})
	case *sshFxpWritePacket:
		f, ok := s.getHandle(p.Handle)
		if !ok {
			return s.sendError(p, syscall.EBADF)
		}

		_, err := f.WriteAt(p.Data, int64(p.Offset))
		return s.sendError(p, err)
	case serverRespondablePacket:
		err := p.respond(s)
		return errors.Wrap(err, "pkt.respond failed")
	default:
		return errors.Errorf("unexpected packet type %T", p)
	}
}

// Serve serves SFTP connections until the streams stop or the SFTP subsystem
// is stopped.
func (svr *Server) Serve() error {
	var wg sync.WaitGroup
	wg.Add(sftpServerWorkerCount)
	for i := 0; i < sftpServerWorkerCount; i++ {
		go func() {
			defer wg.Done()
			if err := svr.sftpServerWorker(); err != nil {
				svr.conn.Close() // shuts down recvPacket
			}
		}()
	}

	var err error
	var pktType uint8
	var pktBytes []byte
	for {
		pktType, pktBytes, err = svr.recvPacket()
		if err != nil {
			break
		}
		svr.pktChan <- rxPacket{fxp(pktType), pktBytes}
	}

	close(svr.pktChan) // shuts down sftpServerWorkers
	wg.Wait()          // wait for all workers to exit

	// close any still-open files
	for handle, file := range svr.openFiles {
		fmt.Fprintf(svr.debugStream, "sftp server file with handle %q left open: %v\n", handle, file.Name())
		file.Close()
	}
	return err // error from recvPacket
}

type id interface {
	id() uint32
}

// The init packet has no ID, so we just return a zero-value ID
func (p sshFxInitPacket) id() uint32 { return 0 }

type sshFxpStatResponse struct {
	ID   uint32
	info os.FileInfo
}

func (p sshFxpStatResponse) MarshalBinary() ([]byte, error) {
	b := []byte{ssh_FXP_ATTRS}
	b = marshalUint32(b, p.ID)
	b = marshalFileInfo(b, p.info)
	return b, nil
}

var emptyFileStat = []interface{}{uint32(0)}

func (p sshFxpOpenPacket) readonly() bool {
	return !p.hasPflags(ssh_FXF_WRITE)
}

func (p sshFxpOpenPacket) hasPflags(flags ...uint32) bool {
	for _, f := range flags {
		if p.Pflags&f == 0 {
			return false
		}
	}
	return true
}

func (p sshFxpOpenPacket) respond(svr *Server) error {
	var osFlags int
	if p.hasPflags(ssh_FXF_READ, ssh_FXF_WRITE) {
		osFlags |= os.O_RDWR
	} else if p.hasPflags(ssh_FXF_WRITE) {
		osFlags |= os.O_WRONLY
	} else if p.hasPflags(ssh_FXF_READ) {
		osFlags |= os.O_RDONLY
	} else {
		// how are they opening?
		return svr.sendError(p, syscall.EINVAL)
	}

	if p.hasPflags(ssh_FXF_APPEND) {
		osFlags |= os.O_APPEND
	}
	if p.hasPflags(ssh_FXF_CREAT) {
		osFlags |= os.O_CREATE
	}
	if p.hasPflags(ssh_FXF_TRUNC) {
		osFlags |= os.O_TRUNC
	}
	if p.hasPflags(ssh_FXF_EXCL) {
		osFlags |= os.O_EXCL
	}

	f, err := os.OpenFile(p.Path, osFlags, 0644)
	if err != nil {
		return svr.sendError(p, err)
	}

	handle := svr.nextHandle(f)
	return svr.sendPacket(sshFxpHandlePacket{p.ID, handle})
}

func (p sshFxpReaddirPacket) respond(svr *Server) error {
	f, ok := svr.getHandle(p.Handle)
	if !ok {
		return svr.sendError(p, syscall.EBADF)
	}

	dirname := f.Name()
	dirents, err := f.Readdir(128)
	if err != nil {
		return svr.sendError(p, err)
	}

	ret := sshFxpNamePacket{ID: p.ID}
	for _, dirent := range dirents {
		ret.NameAttrs = append(ret.NameAttrs, sshFxpNameAttr{
			Name:     dirent.Name(),
			LongName: runLs(dirname, dirent),
			Attrs:    []interface{}{dirent},
		})
	}
	return svr.sendPacket(ret)
}

func (p sshFxpSetstatPacket) respond(svr *Server) error {
	// additional unmarshalling is required for each possibility here
	b := p.Attrs.([]byte)
	var err error

	debug("setstat name \"%s\"", p.Path)
	if (p.Flags & ssh_FILEXFER_ATTR_SIZE) != 0 {
		var size uint64
		if size, b, err = unmarshalUint64Safe(b); err == nil {
			err = os.Truncate(p.Path, int64(size))
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_PERMISSIONS) != 0 {
		var mode uint32
		if mode, b, err = unmarshalUint32Safe(b); err == nil {
			err = os.Chmod(p.Path, os.FileMode(mode))
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_ACMODTIME) != 0 {
		var atime uint32
		var mtime uint32
		if atime, b, err = unmarshalUint32Safe(b); err != nil {
		} else if mtime, b, err = unmarshalUint32Safe(b); err != nil {
		} else {
			atimeT := time.Unix(int64(atime), 0)
			mtimeT := time.Unix(int64(mtime), 0)
			err = os.Chtimes(p.Path, atimeT, mtimeT)
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_UIDGID) != 0 {
		var uid uint32
		var gid uint32
		if uid, b, err = unmarshalUint32Safe(b); err != nil {
		} else if gid, b, err = unmarshalUint32Safe(b); err != nil {
		} else {
			err = os.Chown(p.Path, int(uid), int(gid))
		}
	}

	return svr.sendError(p, err)
}

func (p sshFxpFsetstatPacket) respond(svr *Server) error {
	f, ok := svr.getHandle(p.Handle)
	if !ok {
		return svr.sendError(p, syscall.EBADF)
	}

	// additional unmarshalling is required for each possibility here
	b := p.Attrs.([]byte)
	var err error

	debug("fsetstat name \"%s\"", f.Name())
	if (p.Flags & ssh_FILEXFER_ATTR_SIZE) != 0 {
		var size uint64
		if size, b, err = unmarshalUint64Safe(b); err == nil {
			err = f.Truncate(int64(size))
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_PERMISSIONS) != 0 {
		var mode uint32
		if mode, b, err = unmarshalUint32Safe(b); err == nil {
			err = f.Chmod(os.FileMode(mode))
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_ACMODTIME) != 0 {
		var atime uint32
		var mtime uint32
		if atime, b, err = unmarshalUint32Safe(b); err != nil {
		} else if mtime, b, err = unmarshalUint32Safe(b); err != nil {
		} else {
			atimeT := time.Unix(int64(atime), 0)
			mtimeT := time.Unix(int64(mtime), 0)
			err = os.Chtimes(f.Name(), atimeT, mtimeT)
		}
	}
	if (p.Flags & ssh_FILEXFER_ATTR_UIDGID) != 0 {
		var uid uint32
		var gid uint32
		if uid, b, err = unmarshalUint32Safe(b); err != nil {
		} else if gid, b, err = unmarshalUint32Safe(b); err != nil {
		} else {
			err = f.Chown(int(uid), int(gid))
		}
	}

	return svr.sendError(p, err)
}

// translateErrno translates a syscall error number to a SFTP error code.
func translateErrno(errno syscall.Errno) uint32 {
	switch errno {
	case 0:
		return ssh_FX_OK
	case syscall.ENOENT:
		return ssh_FX_NO_SUCH_FILE
	case syscall.EPERM:
		return ssh_FX_PERMISSION_DENIED
	}

	return ssh_FX_FAILURE
}

func statusFromError(p id, err error) sshFxpStatusPacket {
	ret := sshFxpStatusPacket{
		ID: p.id(),
		StatusError: StatusError{
			// ssh_FX_OK                = 0
			// ssh_FX_EOF               = 1
			// ssh_FX_NO_SUCH_FILE      = 2 ENOENT
			// ssh_FX_PERMISSION_DENIED = 3
			// ssh_FX_FAILURE           = 4
			// ssh_FX_BAD_MESSAGE       = 5
			// ssh_FX_NO_CONNECTION     = 6
			// ssh_FX_CONNECTION_LOST   = 7
			// ssh_FX_OP_UNSUPPORTED    = 8
			Code: ssh_FX_OK,
		},
	}
	if err != nil {
		debug("statusFromError: error is %T %#v", err, err)
		ret.StatusError.Code = ssh_FX_FAILURE
		ret.StatusError.msg = err.Error()
		if err == io.EOF {
			ret.StatusError.Code = ssh_FX_EOF
		} else if errno, ok := err.(syscall.Errno); ok {
			ret.StatusError.Code = translateErrno(errno)
		} else if pathError, ok := err.(*os.PathError); ok {
			debug("statusFromError: error is %T %#v", pathError.Err, pathError.Err)
			if errno, ok := pathError.Err.(syscall.Errno); ok {
				ret.StatusError.Code = translateErrno(errno)
			}
		}
	}
	return ret
}

func clamp(v, max uint32) uint32 {
	if v > max {
		return max
	}
	return v
}
