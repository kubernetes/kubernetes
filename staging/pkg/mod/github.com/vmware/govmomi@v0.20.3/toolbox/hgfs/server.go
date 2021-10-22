/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package hgfs

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// See: https://github.com/vmware/open-vm-tools/blob/master/open-vm-tools/lib/hgfsServer/hgfsServer.c

var (
	// Trace enables hgfs packet tracing
	Trace = false
)

// Server provides an HGFS protocol implementation to support guest tools VmxiHgfsSendPacketCommand
type Server struct {
	Capabilities []Capability

	handlers map[int32]func(*Packet) (interface{}, error)
	schemes  map[string]FileHandler
	sessions map[uint64]*session
	mu       sync.Mutex
	handle   uint32

	chmod func(string, os.FileMode) error
	chown func(string, int, int) error
}

// NewServer creates a new Server instance with the default handlers
func NewServer() *Server {
	if f := flag.Lookup("toolbox.trace"); f != nil {
		Trace, _ = strconv.ParseBool(f.Value.String())
	}

	s := &Server{
		sessions: make(map[uint64]*session),
		schemes:  make(map[string]FileHandler),
		chmod:    os.Chmod,
		chown:    os.Chown,
	}

	s.handlers = map[int32]func(*Packet) (interface{}, error){
		OpCreateSessionV4:  s.CreateSessionV4,
		OpDestroySessionV4: s.DestroySessionV4,
		OpGetattrV2:        s.GetattrV2,
		OpSetattrV2:        s.SetattrV2,
		OpOpen:             s.Open,
		OpClose:            s.Close,
		OpOpenV3:           s.OpenV3,
		OpReadV3:           s.ReadV3,
		OpWriteV3:          s.WriteV3,
	}

	for op := range s.handlers {
		s.Capabilities = append(s.Capabilities, Capability{Op: op, Flags: 0x1})
	}

	return s
}

// RegisterFileHandler enables dispatch to handler for the given scheme.
func (s *Server) RegisterFileHandler(scheme string, handler FileHandler) {
	if handler == nil {
		delete(s.schemes, scheme)
		return
	}
	s.schemes[scheme] = handler
}

// Dispatch unpacks the given request packet and dispatches to the appropriate handler
func (s *Server) Dispatch(packet []byte) ([]byte, error) {
	req := &Packet{}

	err := req.UnmarshalBinary(packet)
	if err != nil {
		return nil, err
	}

	if Trace {
		fmt.Fprintf(os.Stderr, "[hgfs] request  %#v\n", req.Header)
	}

	var res interface{}

	handler, ok := s.handlers[req.Op]
	if ok {
		res, err = handler(req)
	} else {
		err = &Status{
			Code: StatusOperationNotSupported,
			Err:  fmt.Errorf("unsupported Op(%d)", req.Op),
		}
	}

	return req.Reply(res, err)
}

// File interface abstracts standard i/o methods to support transfer
// of regular files and archives of directories.
type File interface {
	io.Reader
	io.Writer
	io.Closer

	Name() string
}

// FileHandler is the plugin interface for hgfs file transport.
type FileHandler interface {
	Stat(*url.URL) (os.FileInfo, error)
	Open(*url.URL, int32) (File, error)
}

// urlParse attempts to convert the given name to a URL with scheme for use as FileHandler dispatch.
func urlParse(name string) *url.URL {
	var info os.FileInfo

	u, err := url.Parse(name)
	if err == nil && u.Scheme == "" {
		info, err = os.Stat(u.Path)
		if err == nil && info.IsDir() {
			u.Scheme = ArchiveScheme // special case for IsDir()
			return u
		}
	}

	u, err = url.Parse(strings.TrimPrefix(name, "/")) // must appear to be an absolute path or hgfs errors
	if err != nil {
		u = &url.URL{Path: name}
	}

	if u.Scheme == "" {
		ix := strings.Index(u.Path, "/")
		if ix > 0 {
			u.Scheme = u.Path[:ix]
			u.Path = u.Path[ix:]
		}
	}

	return u
}

// OpenFile selects the File implementation based on file type and mode.
func (s *Server) OpenFile(name string, mode int32) (File, error) {
	u := urlParse(name)

	if h, ok := s.schemes[u.Scheme]; ok {
		f, serr := h.Open(u, mode)
		if serr != os.ErrNotExist {
			return f, serr
		}
	}

	switch mode {
	case OpenModeReadOnly:
		return os.Open(name)
	case OpenModeWriteOnly:
		flag := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
		return os.OpenFile(name, flag, 0600)
	default:
		return nil, &Status{
			Err:  fmt.Errorf("open mode(%d) not supported for file %q", mode, name),
			Code: StatusAccessDenied,
		}
	}
}

// Stat wraps os.Stat such that we can report directory types as regular files to support archive streaming.
// In the case of standard vmware-tools, attempts to transfer directories result
// with a VIX_E_NOT_A_FILE (see InitiateFileTransfer{To,From}Guest).
// Note that callers on the VMX side that reach this path are only concerned with:
// - does the file exist?
// - size:
//   + used for UI progress with desktop Drag-N-Drop operations, which toolbox does not support.
//   + sent to as Content-Length header in response to GET of FileTransferInformation.Url,
//     if the first ReadV3 size is > HGFS_LARGE_PACKET_MAX
func (s *Server) Stat(name string) (os.FileInfo, error) {
	u := urlParse(name)

	if h, ok := s.schemes[u.Scheme]; ok {
		sinfo, serr := h.Stat(u)
		if serr != os.ErrNotExist {
			return sinfo, serr
		}
	}

	return os.Stat(name)
}

type session struct {
	files map[uint32]File
	mu    sync.Mutex
}

// TODO: we currently depend on the VMX to close files and remove sessions,
// which it does provided it can communicate with the toolbox.  Let's look at
// adding session expiration when implementing OpenModeWriteOnly support.
func newSession() *session {
	return &session{
		files: make(map[uint32]File),
	}
}

func (s *Server) getSession(p *Packet) (*session, error) {
	s.mu.Lock()
	session, ok := s.sessions[p.SessionID]
	s.mu.Unlock()

	if !ok {
		return nil, &Status{
			Code: StatusStaleSession,
			Err:  errors.New("session not found"),
		}
	}

	return session, nil
}

func (s *Server) removeSession(id uint64) bool {
	s.mu.Lock()
	session, ok := s.sessions[id]
	delete(s.sessions, id)
	s.mu.Unlock()

	if !ok {
		return false
	}

	session.mu.Lock()
	defer session.mu.Unlock()

	for _, f := range session.files {
		log.Printf("[hgfs] session %X removed with open file: %s", id, f.Name())
		_ = f.Close()
	}

	return true
}

// open-vm-tools' session max is 1024, there shouldn't be more than a handful at a given time in our use cases
const maxSessions = 24

// CreateSessionV4 handls OpCreateSessionV4 requests
func (s *Server) CreateSessionV4(p *Packet) (interface{}, error) {
	const SessionMaxPacketSizeValid = 0x1

	req := new(RequestCreateSessionV4)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	res := &ReplyCreateSessionV4{
		SessionID:       uint64(rand.Int63()),
		NumCapabilities: uint32(len(s.Capabilities)),
		MaxPacketSize:   LargePacketMax,
		Flags:           SessionMaxPacketSizeValid,
		Capabilities:    s.Capabilities,
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.sessions) > maxSessions {
		return nil, &Status{Code: StatusTooManySessions}
	}

	s.sessions[res.SessionID] = newSession()

	return res, nil
}

// DestroySessionV4 handls OpDestroySessionV4 requests
func (s *Server) DestroySessionV4(p *Packet) (interface{}, error) {
	if s.removeSession(p.SessionID) {
		return &ReplyDestroySessionV4{}, nil
	}

	return nil, &Status{Code: StatusStaleSession}
}

// Stat maps os.FileInfo to AttrV2
func (a *AttrV2) Stat(info os.FileInfo) {
	switch {
	case info.IsDir():
		a.Type = FileTypeDirectory
	case info.Mode()&os.ModeSymlink == os.ModeSymlink:
		a.Type = FileTypeSymlink
	default:
		a.Type = FileTypeRegular
	}

	a.Size = uint64(info.Size())

	a.Mask = AttrValidType | AttrValidSize

	a.sysStat(info)
}

// GetattrV2 handles OpGetattrV2 requests
func (s *Server) GetattrV2(p *Packet) (interface{}, error) {
	res := &ReplyGetattrV2{}

	req := new(RequestGetattrV2)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	name := req.FileName.Path()
	info, err := s.Stat(name)
	if err != nil {
		return nil, err
	}

	res.Attr.Stat(info)

	return res, nil
}

// SetattrV2 handles OpSetattrV2 requests
func (s *Server) SetattrV2(p *Packet) (interface{}, error) {
	res := &ReplySetattrV2{}

	req := new(RequestSetattrV2)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	name := req.FileName.Path()

	_, err = os.Stat(name)
	if err != nil && os.IsNotExist(err) {
		// assuming this is a virtual file
		return res, nil
	}

	uid := -1
	if req.Attr.Mask&AttrValidUserID == AttrValidUserID {
		uid = int(req.Attr.UserID)
	}

	gid := -1
	if req.Attr.Mask&AttrValidGroupID == AttrValidGroupID {
		gid = int(req.Attr.GroupID)
	}

	err = s.chown(name, uid, gid)
	if err != nil {
		return nil, err
	}

	var perm os.FileMode

	if req.Attr.Mask&AttrValidOwnerPerms == AttrValidOwnerPerms {
		perm |= os.FileMode(req.Attr.OwnerPerms) << 6
	}

	if req.Attr.Mask&AttrValidGroupPerms == AttrValidGroupPerms {
		perm |= os.FileMode(req.Attr.GroupPerms) << 3
	}

	if req.Attr.Mask&AttrValidOtherPerms == AttrValidOtherPerms {
		perm |= os.FileMode(req.Attr.OtherPerms)
	}

	if perm != 0 {
		err = s.chmod(name, perm)
		if err != nil {
			return nil, err
		}
	}

	return res, nil
}

func (s *Server) newHandle() uint32 {
	return atomic.AddUint32(&s.handle, 1)
}

// Open handles OpOpen requests
func (s *Server) Open(p *Packet) (interface{}, error) {
	req := new(RequestOpen)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	session, err := s.getSession(p)
	if err != nil {
		return nil, err
	}

	name := req.FileName.Path()
	mode := req.OpenMode

	if mode != OpenModeReadOnly {
		return nil, &Status{
			Err:  fmt.Errorf("open mode(%d) not supported for file %q", mode, name),
			Code: StatusAccessDenied,
		}
	}

	file, err := s.OpenFile(name, mode)
	if err != nil {
		return nil, err
	}

	res := &ReplyOpen{
		Handle: s.newHandle(),
	}

	session.mu.Lock()
	session.files[res.Handle] = file
	session.mu.Unlock()

	return res, nil
}

// Close handles OpClose requests
func (s *Server) Close(p *Packet) (interface{}, error) {
	req := new(RequestClose)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	session, err := s.getSession(p)
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	file, ok := session.files[req.Handle]
	if ok {
		delete(session.files, req.Handle)
	}
	session.mu.Unlock()

	if ok {
		err = file.Close()
	} else {
		return nil, &Status{Code: StatusInvalidHandle}
	}

	return &ReplyClose{}, err
}

// OpenV3 handles OpOpenV3 requests
func (s *Server) OpenV3(p *Packet) (interface{}, error) {
	req := new(RequestOpenV3)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	session, err := s.getSession(p)
	if err != nil {
		return nil, err
	}

	name := req.FileName.Path()

	if req.DesiredLock != LockNone {
		return nil, &Status{
			Err:  fmt.Errorf("open lock type=%d not supported for file %q", req.DesiredLock, name),
			Code: StatusOperationNotSupported,
		}
	}

	file, err := s.OpenFile(name, req.OpenMode)
	if err != nil {
		return nil, err
	}

	res := &ReplyOpenV3{
		Handle: s.newHandle(),
	}

	session.mu.Lock()
	session.files[res.Handle] = file
	session.mu.Unlock()

	return res, nil
}

// ReadV3 handles OpReadV3 requests
func (s *Server) ReadV3(p *Packet) (interface{}, error) {
	req := new(RequestReadV3)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	session, err := s.getSession(p)
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	file, ok := session.files[req.Handle]
	session.mu.Unlock()

	if !ok {
		return nil, &Status{Code: StatusInvalidHandle}
	}

	buf := make([]byte, req.RequiredSize)

	// Use ReadFull as Read() of an archive io.Pipe may return much smaller chunks,
	// such as when we've read a tar header.
	n, err := io.ReadFull(file, buf)
	if err != nil && n == 0 {
		if err != io.EOF {
			return nil, err
		}
	}

	res := &ReplyReadV3{
		ActualSize: uint32(n),
		Payload:    buf[:n],
	}

	return res, nil
}

// WriteV3 handles OpWriteV3 requests
func (s *Server) WriteV3(p *Packet) (interface{}, error) {
	req := new(RequestWriteV3)
	err := UnmarshalBinary(p.Payload, req)
	if err != nil {
		return nil, err
	}

	session, err := s.getSession(p)
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	file, ok := session.files[req.Handle]
	session.mu.Unlock()

	if !ok {
		return nil, &Status{Code: StatusInvalidHandle}
	}

	n, err := file.Write(req.Payload)
	if err != nil {
		return nil, err
	}

	res := &ReplyWriteV3{
		ActualSize: uint32(n),
	}

	return res, nil
}
