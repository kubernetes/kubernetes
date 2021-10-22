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

package vix

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

const (
	CommandMagicWord = 0xd00d0001

	CommandGetToolsState = 62

	CommandStartProgram     = 185
	CommandListProcessesEx  = 186
	CommandReadEnvVariables = 187
	CommandTerminateProcess = 193

	CommandCreateDirectoryEx        = 178
	CommandMoveGuestFileEx          = 179
	CommandMoveGuestDirectory       = 180
	CommandCreateTemporaryFileEx    = 181
	CommandCreateTemporaryDirectory = 182
	CommandSetGuestFileAttributes   = 183
	CommandDeleteGuestFileEx        = 194
	CommandDeleteGuestDirectoryEx   = 195

	CommandListFiles                     = 177
	HgfsSendPacketCommand                = 84
	CommandInitiateFileTransferFromGuest = 188
	CommandInitiateFileTransferToGuest   = 189

	// VIX_USER_CREDENTIAL_NAME_PASSWORD
	UserCredentialTypeNamePassword = 1

	// VIX_E_* constants from vix.h
	OK                 = 0
	Fail               = 1
	InvalidArg         = 3
	FileNotFound       = 4
	FileAlreadyExists  = 12
	FileAccessError    = 13
	AuthenticationFail = 35

	UnrecognizedCommandInGuest = 3025
	InvalidMessageHeader       = 10000
	InvalidMessageBody         = 10001
	NotAFile                   = 20001
	NotADirectory              = 20002
	NoSuchProcess              = 20003
	DirectoryNotEmpty          = 20006

	// VIX_COMMAND_* constants from Commands.h
	CommandGuestReturnsBinary = 0x80

	// VIX_FILE_ATTRIBUTES_ constants from vix.h
	FileAttributesDirectory = 0x0001
	FileAttributesSymlink   = 0x0002
)

// SetGuestFileAttributes flags as defined in vixOpenSource.h
const (
	FileAttributeSetAccessDate      = 0x0001
	FileAttributeSetModifyDate      = 0x0002
	FileAttributeSetReadonly        = 0x0004
	FileAttributeSetHidden          = 0x0008
	FileAttributeSetUnixOwnerid     = 0x0010
	FileAttributeSetUnixGroupid     = 0x0020
	FileAttributeSetUnixPermissions = 0x0040
)

type Error int

func (err Error) Error() string {
	return fmt.Sprintf("vix error=%d", err)
}

// ErrorCode does its best to map the given error to a VIX error code.
// See also: Vix_TranslateErrno
func ErrorCode(err error) int {
	switch t := err.(type) {
	case Error:
		return int(t)
	case *os.PathError:
		if errno, ok := t.Err.(syscall.Errno); ok {
			switch errno {
			case syscall.ENOTEMPTY:
				return DirectoryNotEmpty
			}
		}
	case *exec.Error:
		if t.Err == exec.ErrNotFound {
			return FileNotFound
		}
	}

	switch {
	case os.IsNotExist(err):
		return FileNotFound
	case os.IsExist(err):
		return FileAlreadyExists
	case os.IsPermission(err):
		return FileAccessError
	default:
		return Fail
	}
}

type Header struct {
	Magic          uint32
	MessageVersion uint16

	TotalMessageLength uint32
	HeaderLength       uint32
	BodyLength         uint32
	CredentialLength   uint32

	CommonFlags uint8
}

type CommandRequestHeader struct {
	Header

	OpCode       uint32
	RequestFlags uint32

	TimeOut uint32

	Cookie         uint64
	ClientHandleID uint32

	UserCredentialType uint32
}

type StartProgramRequest struct {
	CommandRequestHeader

	Body struct {
		StartMinimized    uint8
		ProgramPathLength uint32
		ArgumentsLength   uint32
		WorkingDirLength  uint32
		NumEnvVars        uint32
		EnvVarLength      uint32
	}

	ProgramPath string
	Arguments   string
	WorkingDir  string
	EnvVars     []string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *StartProgramRequest) MarshalBinary() ([]byte, error) {
	var env bytes.Buffer

	if n := len(r.EnvVars); n != 0 {
		for _, e := range r.EnvVars {
			_, _ = env.Write([]byte(e))
			_ = env.WriteByte(0)
		}
		r.Body.NumEnvVars = uint32(n)
		r.Body.EnvVarLength = uint32(env.Len())
	}

	var fields []string

	add := func(s string, l *uint32) {
		if n := len(s); n != 0 {
			*l = uint32(n) + 1
			fields = append(fields, s)
		}
	}

	add(r.ProgramPath, &r.Body.ProgramPathLength)
	add(r.Arguments, &r.Body.ArgumentsLength)
	add(r.WorkingDir, &r.Body.WorkingDirLength)

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	for _, val := range fields {
		_, _ = buf.Write([]byte(val))
		_ = buf.WriteByte(0)
	}

	if r.Body.EnvVarLength != 0 {
		_, _ = buf.Write(env.Bytes())
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *StartProgramRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	fields := []struct {
		len uint32
		val *string
	}{
		{r.Body.ProgramPathLength, &r.ProgramPath},
		{r.Body.ArgumentsLength, &r.Arguments},
		{r.Body.WorkingDirLength, &r.WorkingDir},
	}

	for _, field := range fields {
		if field.len == 0 {
			continue
		}

		x := buf.Next(int(field.len))
		*field.val = string(bytes.TrimRight(x, "\x00"))
	}

	for i := 0; i < int(r.Body.NumEnvVars); i++ {
		env, rerr := buf.ReadString(0)
		if rerr != nil {
			return rerr
		}

		env = env[:len(env)-1] // discard NULL terminator
		r.EnvVars = append(r.EnvVars, env)
	}

	return nil
}

type KillProcessRequest struct {
	CommandRequestHeader

	Body struct {
		Pid     int64
		Options uint32
	}
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *KillProcessRequest) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *KillProcessRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	return binary.Read(buf, binary.LittleEndian, &r.Body)
}

type ListProcessesRequest struct {
	CommandRequestHeader

	Body struct {
		Key     uint32
		Offset  uint32
		NumPids uint32
	}

	Pids []int64
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ListProcessesRequest) MarshalBinary() ([]byte, error) {
	r.Body.NumPids = uint32(len(r.Pids))

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	for _, pid := range r.Pids {
		_ = binary.Write(buf, binary.LittleEndian, &pid)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ListProcessesRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	r.Pids = make([]int64, r.Body.NumPids)

	for i := uint32(0); i < r.Body.NumPids; i++ {
		err := binary.Read(buf, binary.LittleEndian, &r.Pids[i])
		if err != nil {
			return err
		}
	}

	return nil
}

type ReadEnvironmentVariablesRequest struct {
	CommandRequestHeader

	Body struct {
		NumNames    uint32
		NamesLength uint32
	}

	Names []string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ReadEnvironmentVariablesRequest) MarshalBinary() ([]byte, error) {
	var env bytes.Buffer

	if n := len(r.Names); n != 0 {
		for _, e := range r.Names {
			_, _ = env.Write([]byte(e))
			_ = env.WriteByte(0)
		}
		r.Body.NumNames = uint32(n)
		r.Body.NamesLength = uint32(env.Len())
	}

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	if r.Body.NamesLength != 0 {
		_, _ = buf.Write(env.Bytes())
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ReadEnvironmentVariablesRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	for i := 0; i < int(r.Body.NumNames); i++ {
		env, rerr := buf.ReadString(0)
		if rerr != nil {
			return rerr
		}

		env = env[:len(env)-1] // discard NULL terminator
		r.Names = append(r.Names, env)
	}

	return nil
}

type CreateTempFileRequest struct {
	CommandRequestHeader

	Body struct {
		Options             int32
		FilePrefixLength    uint32
		FileSuffixLength    uint32
		DirectoryPathLength uint32
		PropertyListLength  uint32
	}

	FilePrefix    string
	FileSuffix    string
	DirectoryPath string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *CreateTempFileRequest) MarshalBinary() ([]byte, error) {
	var fields []string

	add := func(s string, l *uint32) {
		*l = uint32(len(s)) // NOTE: NULL byte is not included in the length fields on the wire
		fields = append(fields, s)
	}

	add(r.FilePrefix, &r.Body.FilePrefixLength)
	add(r.FileSuffix, &r.Body.FileSuffixLength)
	add(r.DirectoryPath, &r.Body.DirectoryPathLength)

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	for _, val := range fields {
		_, _ = buf.Write([]byte(val))
		_ = buf.WriteByte(0)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *CreateTempFileRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	fields := []struct {
		len uint32
		val *string
	}{
		{r.Body.FilePrefixLength, &r.FilePrefix},
		{r.Body.FileSuffixLength, &r.FileSuffix},
		{r.Body.DirectoryPathLength, &r.DirectoryPath},
	}

	for _, field := range fields {
		field.len++ // NOTE: NULL byte is not included in the length fields on the wire

		x := buf.Next(int(field.len))
		*field.val = string(bytes.TrimRight(x, "\x00"))
	}

	return nil
}

type FileRequest struct {
	CommandRequestHeader

	Body struct {
		FileOptions         int32
		GuestPathNameLength uint32
	}

	GuestPathName string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *FileRequest) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	r.Body.GuestPathNameLength = uint32(len(r.GuestPathName))

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	_, _ = buf.WriteString(r.GuestPathName)
	_ = buf.WriteByte(0)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *FileRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	name := buf.Next(int(r.Body.GuestPathNameLength))
	r.GuestPathName = string(bytes.TrimRight(name, "\x00"))

	return nil
}

type DirRequest struct {
	CommandRequestHeader

	Body struct {
		FileOptions          int32
		GuestPathNameLength  uint32
		FilePropertiesLength uint32
		Recursive            bool
	}

	GuestPathName string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *DirRequest) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	r.Body.GuestPathNameLength = uint32(len(r.GuestPathName))

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	_, _ = buf.WriteString(r.GuestPathName)
	_ = buf.WriteByte(0)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *DirRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	name := buf.Next(int(r.Body.GuestPathNameLength))
	r.GuestPathName = string(bytes.TrimRight(name, "\x00"))

	return nil
}

type RenameFileRequest struct {
	CommandRequestHeader

	Body struct {
		CopyFileOptions      int32
		OldPathNameLength    uint32
		NewPathNameLength    uint32
		FilePropertiesLength uint32
		Overwrite            bool
	}

	OldPathName string
	NewPathName string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *RenameFileRequest) MarshalBinary() ([]byte, error) {
	var fields []string

	add := func(s string, l *uint32) {
		*l = uint32(len(s)) // NOTE: NULL byte is not included in the length fields on the wire
		fields = append(fields, s)
	}

	add(r.OldPathName, &r.Body.OldPathNameLength)
	add(r.NewPathName, &r.Body.NewPathNameLength)

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	for _, val := range fields {
		_, _ = buf.Write([]byte(val))
		_ = buf.WriteByte(0)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *RenameFileRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	fields := []struct {
		len uint32
		val *string
	}{
		{r.Body.OldPathNameLength, &r.OldPathName},
		{r.Body.NewPathNameLength, &r.NewPathName},
	}

	for _, field := range fields {
		field.len++ // NOTE: NULL byte is not included in the length fields on the wire

		x := buf.Next(int(field.len))
		*field.val = string(bytes.TrimRight(x, "\x00"))
	}

	return nil
}

type ListFilesRequest struct {
	CommandRequestHeader

	Body struct {
		FileOptions         int32
		GuestPathNameLength uint32
		PatternLength       uint32
		Index               int32
		MaxResults          int32
		Offset              uint64
	}

	GuestPathName string
	Pattern       string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *ListFilesRequest) MarshalBinary() ([]byte, error) {
	var fields []string

	add := func(s string, l *uint32) {
		if n := len(s); n != 0 {
			*l = uint32(n) + 1
			fields = append(fields, s)
		}
	}

	add(r.GuestPathName, &r.Body.GuestPathNameLength)
	add(r.Pattern, &r.Body.PatternLength)

	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	for _, val := range fields {
		_, _ = buf.Write([]byte(val))
		_ = buf.WriteByte(0)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *ListFilesRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	fields := []struct {
		len uint32
		val *string
	}{
		{r.Body.GuestPathNameLength, &r.GuestPathName},
		{r.Body.PatternLength, &r.Pattern},
	}

	for _, field := range fields {
		if field.len == 0 {
			continue
		}

		x := buf.Next(int(field.len))
		*field.val = string(bytes.TrimRight(x, "\x00"))
	}

	return nil
}

type SetGuestFileAttributesRequest struct {
	CommandRequestHeader

	Body struct {
		FileOptions         int32
		AccessTime          int64
		ModificationTime    int64
		OwnerID             int32
		GroupID             int32
		Permissions         int32
		Hidden              bool
		ReadOnly            bool
		GuestPathNameLength uint32
	}

	GuestPathName string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *SetGuestFileAttributesRequest) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	r.Body.GuestPathNameLength = uint32(len(r.GuestPathName))

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	_, _ = buf.WriteString(r.GuestPathName)
	_ = buf.WriteByte(0)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *SetGuestFileAttributesRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	name := buf.Next(int(r.Body.GuestPathNameLength))
	r.GuestPathName = string(bytes.TrimRight(name, "\x00"))

	return nil
}

func (r *SetGuestFileAttributesRequest) IsSet(opt int32) bool {
	return r.Body.FileOptions&opt == opt
}

type CommandHgfsSendPacket struct {
	CommandRequestHeader

	Body struct {
		PacketSize uint32
		Timeout    int32
	}

	Packet []byte
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *CommandHgfsSendPacket) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	_, _ = buf.Write(r.Packet)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *CommandHgfsSendPacket) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	r.Packet = buf.Next(int(r.Body.PacketSize))

	return nil
}

type InitiateFileTransferToGuestRequest struct {
	CommandRequestHeader

	Body struct {
		Options             int32
		GuestPathNameLength uint32
		Overwrite           bool
	}

	GuestPathName string
}

// MarshalBinary implements the encoding.BinaryMarshaler interface
func (r *InitiateFileTransferToGuestRequest) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	r.Body.GuestPathNameLength = uint32(len(r.GuestPathName))

	_ = binary.Write(buf, binary.LittleEndian, &r.Body)

	_, _ = buf.WriteString(r.GuestPathName)
	_ = buf.WriteByte(0)

	return buf.Bytes(), nil
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface
func (r *InitiateFileTransferToGuestRequest) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(data)

	err := binary.Read(buf, binary.LittleEndian, &r.Body)
	if err != nil {
		return err
	}

	name := buf.Next(int(r.Body.GuestPathNameLength))
	r.GuestPathName = string(bytes.TrimRight(name, "\x00"))

	return nil
}

type UserCredentialNamePassword struct {
	Body struct {
		NameLength     uint32
		PasswordLength uint32
	}

	Name     string
	Password string
}

func (c *UserCredentialNamePassword) UnmarshalBinary(data []byte) error {
	buf := bytes.NewBuffer(bytes.TrimRight(data, "\x00"))

	err := binary.Read(buf, binary.LittleEndian, &c.Body)
	if err != nil {
		return err
	}

	str, err := base64.StdEncoding.DecodeString(string(buf.Bytes()))
	if err != nil {
		return err
	}

	c.Name = string(str[0:c.Body.NameLength])
	c.Password = string(str[c.Body.NameLength+1 : len(str)-1])

	return nil
}

func (c *UserCredentialNamePassword) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	c.Body.NameLength = uint32(len(c.Name))
	c.Body.PasswordLength = uint32(len(c.Password))

	_ = binary.Write(buf, binary.LittleEndian, &c.Body)

	src := append([]byte(c.Name+"\x00"), []byte(c.Password+"\x00")...)

	enc := base64.StdEncoding
	pwd := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(pwd, src)
	_, _ = buf.Write(pwd)
	_ = buf.WriteByte(0)

	return buf.Bytes(), nil
}
