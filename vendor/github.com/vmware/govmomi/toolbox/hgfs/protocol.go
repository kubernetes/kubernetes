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
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"strings"
)

// See: https://github.com/vmware/open-vm-tools/blob/master/open-vm-tools/lib/include/hgfsProto.h

// Opcodes for server operations as defined in hgfsProto.h
const (
	OpOpen               = iota /* Open file */
	OpRead                      /* Read from file */
	OpWrite                     /* Write to file */
	OpClose                     /* Close file */
	OpSearchOpen                /* Start new search */
	OpSearchRead                /* Get next search response */
	OpSearchClose               /* End a search */
	OpGetattr                   /* Get file attributes */
	OpSetattr                   /* Set file attributes */
	OpCreateDir                 /* Create new directory */
	OpDeleteFile                /* Delete a file */
	OpDeleteDir                 /* Delete a directory */
	OpRename                    /* Rename a file or directory */
	OpQueryVolumeInfo           /* Query volume information */
	OpOpenV2                    /* Open file */
	OpGetattrV2                 /* Get file attributes */
	OpSetattrV2                 /* Set file attributes */
	OpSearchReadV2              /* Get next search response */
	OpCreateSymlink             /* Create a symlink */
	OpServerLockChange          /* Change the oplock on a file */
	OpCreateDirV2               /* Create a directory */
	OpDeleteFileV2              /* Delete a file */
	OpDeleteDirV2               /* Delete a directory */
	OpRenameV2                  /* Rename a file or directory */
	OpOpenV3                    /* Open file */
	OpReadV3                    /* Read from file */
	OpWriteV3                   /* Write to file */
	OpCloseV3                   /* Close file */
	OpSearchOpenV3              /* Start new search */
	OpSearchReadV3              /* Read V3 directory entries */
	OpSearchCloseV3             /* End a search */
	OpGetattrV3                 /* Get file attributes */
	OpSetattrV3                 /* Set file attributes */
	OpCreateDirV3               /* Create new directory */
	OpDeleteFileV3              /* Delete a file */
	OpDeleteDirV3               /* Delete a directory */
	OpRenameV3                  /* Rename a file or directory */
	OpQueryVolumeInfoV3         /* Query volume information */
	OpCreateSymlinkV3           /* Create a symlink */
	OpServerLockChangeV3        /* Change the oplock on a file */
	OpWriteWin32StreamV3        /* Write WIN32_STREAM_ID format data to file */
	OpCreateSessionV4           /* Create a session and return host capabilities. */
	OpDestroySessionV4          /* Destroy/close session. */
	OpReadFastV4                /* Read */
	OpWriteFastV4               /* Write */
	OpSetWatchV4                /* Start monitoring directory changes. */
	OpRemoveWatchV4             /* Stop monitoring directory changes. */
	OpNotifyV4                  /* Notification for a directory change event. */
	OpSearchReadV4              /* Read V4 directory entries. */
	OpOpenV4                    /* Open file */
	OpEnumerateStreamsV4        /* Enumerate alternative named streams for a file. */
	OpGetattrV4                 /* Get file attributes */
	OpSetattrV4                 /* Set file attributes */
	OpDeleteV4                  /* Delete a file or a directory */
	OpLinkmoveV4                /* Rename/move/create hard link. */
	OpFsctlV4                   /* Sending FS control requests. */
	OpAccessCheckV4             /* Access check. */
	OpFsyncV4                   /* Flush all cached data to the disk. */
	OpQueryVolumeInfoV4         /* Query volume information. */
	OpOplockAcquireV4           /* Acquire OPLOCK. */
	OpOplockBreakV4             /* Break or downgrade OPLOCK. */
	OpLockByteRangeV4           /* Acquire byte range lock. */
	OpUnlockByteRangeV4         /* Release byte range lock. */
	OpQueryEasV4                /* Query extended attributes. */
	OpSetEasV4                  /* Add or modify extended attributes. */
	OpNewHeader          = 0xff /* Header op, must be unique, distinguishes packet headers. */
)

// Status codes
const (
	StatusSuccess = iota
	StatusNoSuchFileOrDir
	StatusInvalidHandle
	StatusOperationNotPermitted
	StatusFileExists
	StatusNotDirectory
	StatusDirNotEmpty
	StatusProtocolError
	StatusAccessDenied
	StatusInvalidName
	StatusGenericError
	StatusSharingViolation
	StatusNoSpace
	StatusOperationNotSupported
	StatusNameTooLong
	StatusInvalidParameter
	StatusNotSameDevice
	StatusStaleSession
	StatusTooManySessions
	StatusTransportError
)

// Flags for attr mask
const (
	AttrValidType = 1 << iota
	AttrValidSize
	AttrValidCreateTime
	AttrValidAccessTime
	AttrValidWriteTime
	AttrValidChangeTime
	AttrValidSpecialPerms
	AttrValidOwnerPerms
	AttrValidGroupPerms
	AttrValidOtherPerms
	AttrValidFlags
	AttrValidAllocationSize
	AttrValidUserID
	AttrValidGroupID
	AttrValidFileID
	AttrValidVolID
	AttrValidNonStaticFileID
	AttrValidEffectivePerms
	AttrValidExtendAttrSize
	AttrValidReparsePoint
	AttrValidShortName
)

// HeaderVersion for HGFS protocol version 4
const HeaderVersion = 0x1

// LargePacketMax is maximum size of an hgfs packet
const LargePacketMax = 0xf800 // HGFS_LARGE_PACKET_MAX

// Packet flags
const (
	PacketFlagRequest = 1 << iota
	PacketFlagReply
	PacketFlagInfoExterror
	PacketFlagValidFlags = 0x7
)

// Status is an error type that encapsulates an error status code and the cause
type Status struct {
	Err  error
	Code uint32
}

func (s *Status) Error() string {
	if s.Err != nil {
		return s.Err.Error()
	}

	return fmt.Sprintf("hgfs.Status=%d", s.Code)
}

// errorStatus maps the given error type to a status code
func errorStatus(err error) uint32 {
	if x, ok := err.(*Status); ok {
		return x.Code
	}

	switch {
	case os.IsNotExist(err):
		return StatusNoSuchFileOrDir
	case os.IsExist(err):
		return StatusFileExists
	case os.IsPermission(err):
		return StatusOperationNotPermitted
	}

	return StatusGenericError
}

// ProtocolError wraps the given error as a Status type
func ProtocolError(err error) error {
	return &Status{
		Err:  err,
		Code: StatusProtocolError,
	}
}

// Request as defined in hgfsProto.h:HgfsRequest
type Request struct {
	Handle uint32
	Op     int32
}

// Reply as defined in hgfsProto.h:HgfsReply
type Reply struct {
	Handle uint32
	Status uint32
}

// Header as defined in hgfsProto.h:HgfsHeader
type Header struct {
	Version     uint8
	Reserved1   [3]uint8
	Dummy       int32
	PacketSize  uint32
	HeaderSize  uint32
	RequestID   uint32
	Op          int32
	Status      uint32
	Flags       uint32
	Information uint32
	SessionID   uint64
	Reserved    uint64
}

var (
	headerSize = uint32(binary.Size(new(Header)))

	packetSize = func(r *Packet) uint32 {
		return headerSize + uint32(len(r.Payload))
	}
)

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (h *Header) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, h)
	if err != nil {
		return fmt.Errorf("reading hgfs header: %s", err)
	}

	if h.Dummy != OpNewHeader {
		return fmt.Errorf("expected hgfs header with OpNewHeader (%#x), got: %#x", OpNewHeader, h.Dummy)
	}

	return nil
}

// Packet encapsulates an hgfs Header and Payload
type Packet struct {
	Header

	Payload []byte
}

// Reply composes a new Packet with the given payload or error
func (r *Packet) Reply(payload interface{}, err error) ([]byte, error) {
	p := new(Packet)

	status := uint32(StatusSuccess)

	if err != nil {
		status = errorStatus(err)
	} else {
		p.Payload, err = MarshalBinary(payload)
		if err != nil {
			return nil, err
		}
	}

	p.Header = Header{
		Version:     HeaderVersion,
		Dummy:       OpNewHeader,
		PacketSize:  headerSize + uint32(len(p.Payload)),
		HeaderSize:  headerSize,
		RequestID:   r.RequestID,
		Op:          r.Op,
		Status:      status,
		Flags:       PacketFlagReply,
		Information: 0,
		SessionID:   r.SessionID,
	}

	if Trace {
		rc := "OK"
		if err != nil {
			rc = err.Error()
		}
		fmt.Fprintf(os.Stderr, "[hgfs] response %#v [%s]\n", p.Header, rc)
	} else if err != nil {
		log.Printf("[hgfs] op=%d error: %s", r.Op, err)
	}

	return p.MarshalBinary()
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *Packet) MarshalBinary() ([]byte, error) {
	r.Header.PacketSize = packetSize(r)

	buf, _ := MarshalBinary(r.Header)

	return append(buf, r.Payload...), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *Packet) UnmarshalBinary(data []byte) error {
	err := r.Header.UnmarshalBinary(data)
	if err != nil {
		return err
	}

	r.Payload = data[r.HeaderSize:r.PacketSize]

	return nil
}

// Capability as defined in hgfsProto.h:HgfsCapability
type Capability struct {
	Op    int32
	Flags uint32
}

// RequestCreateSessionV4 as defined in hgfsProto.h:HgfsRequestCreateSessionV4
type RequestCreateSessionV4 struct {
	NumCapabilities uint32
	MaxPacketSize   uint32
	Flags           uint32
	Reserved        uint32
	Capabilities    []Capability
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestCreateSessionV4) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	r.NumCapabilities = uint32(len(r.Capabilities))

	fields := []*uint32{
		&r.NumCapabilities,
		&r.MaxPacketSize,
		&r.Flags,
		&r.Reserved,
	}

	for _, p := range fields {
		err := binary.Write(buf, binary.LittleEndian, p)
		if err != nil {
			return nil, err
		}
	}

	for i := uint32(0); i < r.NumCapabilities; i++ {
		err := binary.Write(buf, binary.LittleEndian, &r.Capabilities[i])
		if err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestCreateSessionV4) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	fields := []*uint32{
		&r.NumCapabilities,
		&r.MaxPacketSize,
		&r.Flags,
		&r.Reserved,
	}

	for _, p := range fields {
		err := binary.Read(buf, binary.LittleEndian, p)
		if err != nil {
			return err
		}
	}

	for i := uint32(0); i < r.NumCapabilities; i++ {
		var cap Capability
		err := binary.Read(buf, binary.LittleEndian, &cap)
		if err != nil {
			return err
		}

		r.Capabilities = append(r.Capabilities, cap)
	}

	return nil
}

// ReplyCreateSessionV4 as defined in hgfsProto.h:HgfsReplyCreateSessionV4
type ReplyCreateSessionV4 struct {
	SessionID       uint64
	NumCapabilities uint32
	MaxPacketSize   uint32
	IdentityOffset  uint32
	Flags           uint32
	Reserved        uint32
	Capabilities    []Capability
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ReplyCreateSessionV4) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	fields := []interface{}{
		&r.SessionID,
		&r.NumCapabilities,
		&r.MaxPacketSize,
		&r.IdentityOffset,
		&r.Flags,
		&r.Reserved,
	}

	for _, p := range fields {
		err := binary.Write(buf, binary.LittleEndian, p)
		if err != nil {
			return nil, err
		}
	}

	for i := uint32(0); i < r.NumCapabilities; i++ {
		err := binary.Write(buf, binary.LittleEndian, &r.Capabilities[i])
		if err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ReplyCreateSessionV4) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	fields := []interface{}{
		&r.SessionID,
		&r.NumCapabilities,
		&r.MaxPacketSize,
		&r.IdentityOffset,
		&r.Flags,
		&r.Reserved,
	}

	for _, p := range fields {
		err := binary.Read(buf, binary.LittleEndian, p)
		if err != nil {
			return err
		}
	}

	for i := uint32(0); i < r.NumCapabilities; i++ {
		var cap Capability
		err := binary.Read(buf, binary.LittleEndian, &cap)
		if err != nil {
			return err
		}

		r.Capabilities = append(r.Capabilities, cap)
	}

	return nil
}

// RequestDestroySessionV4 as defined in hgfsProto.h:HgfsRequestDestroySessionV4
type RequestDestroySessionV4 struct {
	Reserved uint64
}

// ReplyDestroySessionV4 as defined in hgfsProto.h:HgfsReplyDestroySessionV4
type ReplyDestroySessionV4 struct {
	Reserved uint64
}

// FileName as defined in hgfsProto.h:HgfsFileName
type FileName struct {
	Length uint32
	Name   string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (f *FileName) MarshalBinary() ([]byte, error) {
	name := f.Name
	f.Length = uint32(len(f.Name))
	if f.Length == 0 {
		// field is defined as 'char name[1];', this byte is required for min sizeof() validation
		name = "\x00"
	}
	return MarshalBinary(&f.Length, name)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (f *FileName) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	_ = binary.Read(buf, binary.LittleEndian, &f.Length)

	f.Name = string(buf.Next(int(f.Length)))

	return nil
}

const serverPolicyRootShareName = "root"

// FromString converts name to a FileName
func (f *FileName) FromString(name string) {
	name = strings.TrimPrefix(name, "/")

	cp := strings.Split(name, "/")

	cp = append([]string{serverPolicyRootShareName}, cp...)

	f.Name = strings.Join(cp, "\x00")
	f.Length = uint32(len(f.Name))
}

// Path converts FileName to a string
func (f *FileName) Path() string {
	cp := strings.Split(f.Name, "\x00")

	if len(cp) == 0 || cp[0] != serverPolicyRootShareName {
		return "" // TODO: not happening until if/when we handle Windows shares
	}

	cp[0] = ""

	return strings.Join(cp, "/")
}

// FileNameV3 as defined in hgfsProto.h:HgfsFileNameV3
type FileNameV3 struct {
	Length   uint32
	Flags    uint32
	CaseType int32
	ID       uint32
	Name     string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (f *FileNameV3) MarshalBinary() ([]byte, error) {
	name := f.Name
	f.Length = uint32(len(f.Name))
	if f.Length == 0 {
		// field is defined as 'char name[1];', this byte is required for min sizeof() validation
		name = "\x00"
	}
	return MarshalBinary(&f.Length, &f.Flags, &f.CaseType, &f.ID, name)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (f *FileNameV3) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	fields := []interface{}{
		&f.Length, &f.Flags, &f.CaseType, &f.ID,
	}

	for _, p := range fields {
		if err := binary.Read(buf, binary.LittleEndian, p); err != nil {
			return err
		}
	}

	f.Name = string(buf.Next(int(f.Length)))

	return nil
}

// FromString converts name to a FileNameV3
func (f *FileNameV3) FromString(name string) {
	p := new(FileName)
	p.FromString(name)
	f.Name = p.Name
	f.Length = p.Length
}

// Path converts FileNameV3 to a string
func (f *FileNameV3) Path() string {
	return (&FileName{Name: f.Name, Length: f.Length}).Path()
}

// FileType
const (
	FileTypeRegular = iota
	FileTypeDirectory
	FileTypeSymlink
)

// AttrV2 as defined in hgfsProto.h:HgfsAttrV2
type AttrV2 struct {
	Mask           uint64
	Type           int32
	Size           uint64
	CreationTime   uint64
	AccessTime     uint64
	WriteTime      uint64
	AttrChangeTime uint64
	SpecialPerms   uint8
	OwnerPerms     uint8
	GroupPerms     uint8
	OtherPerms     uint8
	AttrFlags      uint64
	AllocationSize uint64
	UserID         uint32
	GroupID        uint32
	HostFileID     uint64
	VolumeID       uint32
	EffectivePerms uint32
	Reserved2      uint64
}

// RequestGetattrV2 as defined in hgfsProto.h:HgfsRequestGetattrV2
type RequestGetattrV2 struct {
	Request
	AttrHint uint64
	Handle   uint32
	FileName FileName
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestGetattrV2) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Request, &r.AttrHint, &r.Handle, &r.FileName)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestGetattrV2) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Request, &r.AttrHint, &r.Handle, &r.FileName)
}

// ReplyGetattrV2 as defined in hgfsProto.h:HgfsReplyGetattrV2
type ReplyGetattrV2 struct {
	Reply
	Attr          AttrV2
	SymlinkTarget FileName
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ReplyGetattrV2) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Reply, &r.Attr, &r.SymlinkTarget)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ReplyGetattrV2) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Reply, &r.Attr, &r.SymlinkTarget)
}

// RequestSetattrV2 as defined in hgfsProto.h:HgfsRequestSetattrV2
type RequestSetattrV2 struct {
	Request
	Hints    uint64
	Attr     AttrV2
	Handle   uint32
	FileName FileName
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestSetattrV2) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Request, &r.Hints, &r.Attr, &r.Handle, &r.FileName)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestSetattrV2) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Request, &r.Hints, &r.Attr, &r.Handle, &r.FileName)
}

// ReplySetattrV2 as defined in hgfsProto.h:HgfsReplySetattrV2
type ReplySetattrV2 struct {
	Header Reply
}

// OpenMode
const (
	OpenModeReadOnly = iota
	OpenModeWriteOnly
	OpenModeReadWrite
	OpenModeAccmodes
)

// OpenFlags
const (
	Open = iota
	OpenEmpty
	OpenCreate
	OpenCreateSafe
	OpenCreateEmpty
)

// Permissions
const (
	PermRead  = 4
	PermWrite = 2
	PermExec  = 1
)

// RequestOpen as defined in hgfsProto.h:HgfsRequestOpen
type RequestOpen struct {
	Request
	OpenMode    int32
	OpenFlags   int32
	Permissions uint8
	FileName    FileName
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestOpen) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Request, &r.OpenMode, &r.OpenFlags, r.Permissions, &r.FileName)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestOpen) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Request, &r.OpenMode, &r.OpenFlags, &r.Permissions, &r.FileName)
}

// ReplyOpen as defined in hgfsProto.h:HgfsReplyOpen
type ReplyOpen struct {
	Reply
	Handle uint32
}

// RequestClose as defined in hgfsProto.h:HgfsRequestClose
type RequestClose struct {
	Request
	Handle uint32
}

// ReplyClose as defined in hgfsProto.h:HgfsReplyClose
type ReplyClose struct {
	Reply
}

// Lock type
const (
	LockNone = iota
	LockOpportunistic
	LockExclusive
	LockShared
	LockBatch
	LockLease
)

// RequestOpenV3 as defined in hgfsProto.h:HgfsRequestOpenV3
type RequestOpenV3 struct {
	Mask           uint64
	OpenMode       int32
	OpenFlags      int32
	SpecialPerms   uint8
	OwnerPerms     uint8
	GroupPerms     uint8
	OtherPerms     uint8
	AttrFlags      uint64
	AllocationSize uint64
	DesiredAccess  uint32
	ShareAccess    uint32
	DesiredLock    int32
	Reserved1      uint64
	Reserved2      uint64
	FileName       FileNameV3
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestOpenV3) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Mask, &r.OpenMode, &r.OpenFlags,
		&r.SpecialPerms, &r.OwnerPerms, &r.GroupPerms, &r.OtherPerms,
		&r.AttrFlags, &r.AllocationSize, &r.DesiredAccess, &r.ShareAccess,
		&r.DesiredLock, &r.Reserved1, &r.Reserved2, &r.FileName)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestOpenV3) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Mask, &r.OpenMode, &r.OpenFlags,
		&r.SpecialPerms, &r.OwnerPerms, &r.GroupPerms, &r.OtherPerms,
		&r.AttrFlags, &r.AllocationSize, &r.DesiredAccess, &r.ShareAccess,
		&r.DesiredLock, &r.Reserved1, &r.Reserved2, &r.FileName)
}

// ReplyOpenV3 as defined in hgfsProto.h:HgfsReplyOpenV3
type ReplyOpenV3 struct {
	Handle       uint32
	AcquiredLock int32
	Flags        int32
	Reserved     uint32
}

// RequestReadV3 as defined in hgfsProto.h:HgfsRequestReadV3
type RequestReadV3 struct {
	Handle       uint32
	Offset       uint64
	RequiredSize uint32
	Reserved     uint64
}

// ReplyReadV3 as defined in hgfsProto.h:HgfsReplyReadV3
type ReplyReadV3 struct {
	ActualSize uint32
	Reserved   uint64
	Payload    []byte
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ReplyReadV3) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.ActualSize, &r.Reserved, r.Payload)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ReplyReadV3) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.ActualSize, &r.Reserved, &r.Payload)
}

// Write flags
const (
	WriteAppend = 1
)

// RequestWriteV3 as defined in hgfsProto.h:HgfsRequestWriteV3
type RequestWriteV3 struct {
	Handle       uint32
	WriteFlags   uint8
	Offset       uint64
	RequiredSize uint32
	Reserved     uint64
	Payload      []byte
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RequestWriteV3) MarshalBinary() ([]byte, error) {
	return MarshalBinary(&r.Handle, &r.WriteFlags, &r.Offset, &r.RequiredSize, &r.Reserved, r.Payload)
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RequestWriteV3) UnmarshalBinary(data []byte) error {
	return UnmarshalBinary(data, &r.Handle, &r.WriteFlags, &r.Offset, &r.RequiredSize, &r.Reserved, &r.Payload)
}

// ReplyWriteV3 as defined in hgfsProto.h:HgfsReplyWriteV3
type ReplyWriteV3 struct {
	ActualSize uint32
	Reserved   uint64
}
