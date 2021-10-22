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

package toolbox

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/vmware/govmomi/toolbox/hgfs"
	"github.com/vmware/govmomi/toolbox/vix"
)

type CommandHandler func(vix.CommandRequestHeader, []byte) ([]byte, error)

type CommandServer struct {
	Out *ChannelOut

	ProcessManager *ProcessManager

	Authenticate func(vix.CommandRequestHeader, []byte) error

	ProcessStartCommand func(*ProcessManager, *vix.StartProgramRequest) (int64, error)

	handlers map[uint32]CommandHandler

	FileServer *hgfs.Server
}

func registerCommandServer(service *Service) *CommandServer {
	server := &CommandServer{
		Out:            service.out,
		ProcessManager: NewProcessManager(),
	}

	server.handlers = map[uint32]CommandHandler{
		vix.CommandGetToolsState:                 server.GetToolsState,
		vix.CommandStartProgram:                  server.StartCommand,
		vix.CommandTerminateProcess:              server.KillProcess,
		vix.CommandListProcessesEx:               server.ListProcesses,
		vix.CommandReadEnvVariables:              server.ReadEnvironmentVariables,
		vix.CommandCreateTemporaryFileEx:         server.CreateTemporaryFile,
		vix.CommandCreateTemporaryDirectory:      server.CreateTemporaryDirectory,
		vix.CommandDeleteGuestFileEx:             server.DeleteFile,
		vix.CommandCreateDirectoryEx:             server.CreateDirectory,
		vix.CommandDeleteGuestDirectoryEx:        server.DeleteDirectory,
		vix.CommandMoveGuestFileEx:               server.MoveFile,
		vix.CommandMoveGuestDirectory:            server.MoveDirectory,
		vix.CommandListFiles:                     server.ListFiles,
		vix.CommandSetGuestFileAttributes:        server.SetGuestFileAttributes,
		vix.CommandInitiateFileTransferFromGuest: server.InitiateFileTransferFromGuest,
		vix.CommandInitiateFileTransferToGuest:   server.InitiateFileTransferToGuest,
		vix.HgfsSendPacketCommand:                server.ProcessHgfsPacket,
	}

	server.ProcessStartCommand = DefaultStartCommand

	service.RegisterHandler("Vix_1_Relayed_Command", server.Dispatch)

	return server
}

func commandResult(header vix.CommandRequestHeader, rc int, err error, response []byte) []byte {
	// All Foundry tools commands return results that start with a foundry error
	// and a guest-OS-specific error (e.g. errno)
	errno := 0

	if err != nil {
		// TODO: inspect err for system error, setting errno

		response = []byte(err.Error())

		log.Printf("[vix] op=%d error: %s", header.OpCode, err)
	}

	buf := bytes.NewBufferString(fmt.Sprintf("%d %d ", rc, errno))

	if header.CommonFlags&vix.CommandGuestReturnsBinary != 0 {
		// '#' delimits end of ascii and the start of the binary data (see ToolsDaemonTcloReceiveVixCommand)
		_ = buf.WriteByte('#')
	}

	_, _ = buf.Write(response)

	if header.CommonFlags&vix.CommandGuestReturnsBinary == 0 {
		// this is not binary data, so it should be a NULL terminated string (see ToolsDaemonTcloReceiveVixCommand)
		_ = buf.WriteByte(0)
	}

	return buf.Bytes()
}

func (c *CommandServer) Dispatch(data []byte) ([]byte, error) {
	// See ToolsDaemonTcloGetQuotedString
	if data[0] == '"' {
		data = data[1:]
	}

	var name string

	ix := bytes.IndexByte(data, '"')
	if ix > 0 {
		name = string(data[:ix])
		data = data[ix+1:]
	}
	// skip the NULL
	if data[0] == 0 {
		data = data[1:]
	}

	if Trace {
		fmt.Fprintf(os.Stderr, "vix dispatch %q...\n%s\n", name, hex.Dump(data))
	}

	var header vix.CommandRequestHeader
	buf := bytes.NewBuffer(data)
	err := binary.Read(buf, binary.LittleEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != vix.CommandMagicWord {
		return commandResult(header, vix.InvalidMessageHeader, nil, nil), nil
	}

	handler, ok := c.handlers[header.OpCode]
	if !ok {
		return commandResult(header, vix.UnrecognizedCommandInGuest, nil, nil), nil
	}

	if header.OpCode != vix.CommandGetToolsState {
		// Every command expect GetToolsState requires authentication
		creds := buf.Bytes()[header.BodyLength:]

		err = c.authenticate(header, creds[:header.CredentialLength])
		if err != nil {
			return commandResult(header, vix.AuthenticationFail, err, nil), nil
		}
	}

	rc := vix.OK

	response, err := handler(header, buf.Bytes())
	if err != nil {
		rc = vix.ErrorCode(err)
	}

	return commandResult(header, rc, err, response), nil
}

func (c *CommandServer) RegisterHandler(op uint32, handler CommandHandler) {
	c.handlers[op] = handler
}

func (c *CommandServer) GetToolsState(_ vix.CommandRequestHeader, _ []byte) ([]byte, error) {
	hostname, _ := os.Hostname()
	osname := fmt.Sprintf("%s-%s", runtime.GOOS, runtime.GOARCH)

	// Note that vmtoolsd sends back 40 or so of these properties, sticking with the minimal set for now.
	props := vix.PropertyList{
		vix.NewStringProperty(vix.PropertyGuestOsVersion, osname),
		vix.NewStringProperty(vix.PropertyGuestOsVersionShort, osname),
		vix.NewStringProperty(vix.PropertyGuestToolsProductNam, "VMware Tools (Go)"),
		vix.NewStringProperty(vix.PropertyGuestToolsVersion, "10.0.5 build-3227872 (Compatible)"),
		vix.NewStringProperty(vix.PropertyGuestName, hostname),
		vix.NewInt32Property(vix.PropertyGuestToolsAPIOptions, 0x0001), // TODO: const VIX_TOOLSFEATURE_SUPPORT_GET_HANDLE_STATE
		vix.NewInt32Property(vix.PropertyGuestOsFamily, 1),             // TODO: const GUEST_OS_FAMILY_*
		vix.NewBoolProperty(vix.PropertyGuestStartProgramEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestTerminateProcessEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestListProcessesEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestReadEnvironmentVariableEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestMakeDirectoryEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestDeleteFileEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestDeleteDirectoryEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestMoveDirectoryEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestMoveFileEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestCreateTempFileEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestCreateTempDirectoryEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestListFilesEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestChangeFileAttributesEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestInitiateFileTransferFromGuestEnabled, true),
		vix.NewBoolProperty(vix.PropertyGuestInitiateFileTransferToGuestEnabled, true),
	}

	src, _ := props.MarshalBinary()
	enc := base64.StdEncoding
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)

	return buf, nil
}

func (c *CommandServer) StartCommand(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.StartProgramRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	pid, err := c.ProcessStartCommand(c.ProcessManager, r)
	if err != nil {
		return nil, err
	}

	return append([]byte(fmt.Sprintf("%d", pid)), 0), nil
}

func DefaultStartCommand(m *ProcessManager, r *vix.StartProgramRequest) (int64, error) {
	p := NewProcess()

	switch r.ProgramPath {
	case "http.RoundTrip":
		p = NewProcessRoundTrip()
	default:
		// Standard vmware-tools requires an absolute path,
		// we'll enable IO redirection by default without an absolute path.
		if !strings.Contains(r.ProgramPath, "/") {
			p = p.WithIO()
		}
	}

	return m.Start(r, p)
}

func (c *CommandServer) KillProcess(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.KillProcessRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	if c.ProcessManager.Kill(r.Body.Pid) {
		return nil, err
	}

	// TODO: could kill process started outside of toolbox

	return nil, vix.Error(vix.NoSuchProcess)
}

func (c *CommandServer) ListProcesses(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.ListProcessesRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	state := c.ProcessManager.ListProcesses(r.Pids)

	return state, nil
}

func (c *CommandServer) ReadEnvironmentVariables(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.ReadEnvironmentVariablesRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	buf := new(bytes.Buffer)

	if len(r.Names) == 0 {
		for _, e := range os.Environ() {
			_, _ = buf.WriteString(fmt.Sprintf("<ev>%s</ev>", xmlEscape.Replace(e)))
		}
	} else {
		for _, key := range r.Names {
			val := os.Getenv(key)
			if val == "" {
				continue
			}
			_, _ = buf.WriteString(fmt.Sprintf("<ev>%s=%s</ev>", xmlEscape.Replace(key), xmlEscape.Replace(val)))
		}
	}

	return buf.Bytes(), nil
}

func (c *CommandServer) CreateTemporaryFile(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.CreateTempFileRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	f, err := ioutil.TempFile(r.DirectoryPath, r.FilePrefix+"vmware")
	if err != nil {
		return nil, err
	}

	_ = f.Close()

	return []byte(f.Name()), nil
}

func (c *CommandServer) CreateTemporaryDirectory(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.CreateTempFileRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	name, err := ioutil.TempDir(r.DirectoryPath, r.FilePrefix+"vmware")
	if err != nil {
		return nil, err
	}

	return []byte(name), nil
}

func (c *CommandServer) DeleteFile(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.FileRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(r.GuestPathName)
	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, vix.Error(vix.NotAFile)
	}

	err = os.Remove(r.GuestPathName)

	return nil, err
}

func (c *CommandServer) DeleteDirectory(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.DirRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(r.GuestPathName)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		return nil, vix.Error(vix.NotADirectory)
	}

	if r.Body.Recursive {
		err = os.RemoveAll(r.GuestPathName)
	} else {
		err = os.Remove(r.GuestPathName)
	}

	return nil, err
}

func (c *CommandServer) CreateDirectory(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.DirRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	mkdir := os.Mkdir

	if r.Body.Recursive {
		mkdir = os.MkdirAll
	}

	err = mkdir(r.GuestPathName, 0700)

	return nil, err
}

func (c *CommandServer) MoveDirectory(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.RenameFileRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(r.OldPathName)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		return nil, vix.Error(vix.NotADirectory)
	}

	if !r.Body.Overwrite {
		info, err = os.Stat(r.NewPathName)
		if err == nil {
			return nil, vix.Error(vix.FileAlreadyExists)
		}
	}

	return nil, os.Rename(r.OldPathName, r.NewPathName)
}

func (c *CommandServer) MoveFile(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.RenameFileRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(r.OldPathName)
	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, vix.Error(vix.NotAFile)
	}

	if !r.Body.Overwrite {
		info, err = os.Stat(r.NewPathName)
		if err == nil {
			return nil, vix.Error(vix.FileAlreadyExists)
		}
	}

	return nil, os.Rename(r.OldPathName, r.NewPathName)
}

func (c *CommandServer) ListFiles(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.ListFilesRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := os.Lstat(r.GuestPathName)
	if err != nil {
		return nil, err
	}

	var dir string
	var files []os.FileInfo

	if info.IsDir() {
		dir = r.GuestPathName
		files, err = ioutil.ReadDir(r.GuestPathName)
		if err != nil {
			return nil, err
		}
	} else {
		dir = filepath.Dir(r.GuestPathName)
		files = append(files, info)
	}

	offset := r.Body.Offset + uint64(r.Body.Index)
	total := uint64(len(files)) - offset
	if int(offset) < len(files) {
		files = files[offset:]
	} else {
		total = 0 // offset is not valid (open-vm-tools behaves the same in this case)
	}

	var remaining uint64

	if r.Body.MaxResults > 0 && total > uint64(r.Body.MaxResults) {
		remaining = total - uint64(r.Body.MaxResults)
		files = files[:r.Body.MaxResults]
	}

	buf := new(bytes.Buffer)
	buf.WriteString(fmt.Sprintf("<rem>%d</rem>", remaining))

	for _, info = range files {
		buf.WriteString(fileExtendedInfoFormat(dir, info))
	}

	return buf.Bytes(), nil
}

func chtimes(r *vix.SetGuestFileAttributesRequest) error {
	var mtime, atime *time.Time

	if r.IsSet(vix.FileAttributeSetModifyDate) {
		t := time.Unix(r.Body.ModificationTime, 0)
		mtime = &t
	}

	if r.IsSet(vix.FileAttributeSetAccessDate) {
		t := time.Unix(r.Body.AccessTime, 0)
		atime = &t
	}

	if mtime == nil && atime == nil {
		return nil
	}

	info, err := os.Stat(r.GuestPathName)
	if err != nil {
		return err
	}

	if mtime == nil {
		t := info.ModTime()
		mtime = &t
	}

	if atime == nil {
		t := info.ModTime()
		atime = &t
	}

	return os.Chtimes(r.GuestPathName, *atime, *mtime)
}

func chown(r *vix.SetGuestFileAttributesRequest) error {
	uid := -1
	gid := -1

	if r.IsSet(vix.FileAttributeSetUnixOwnerid) {
		uid = int(r.Body.OwnerID)
	}

	if r.IsSet(vix.FileAttributeSetUnixGroupid) {
		gid = int(r.Body.GroupID)
	}

	if uid == -1 && gid == -1 {
		return nil
	}

	return os.Chown(r.GuestPathName, uid, gid)
}

func chmod(r *vix.SetGuestFileAttributesRequest) error {
	if r.IsSet(vix.FileAttributeSetUnixPermissions) {
		return os.Chmod(r.GuestPathName, os.FileMode(r.Body.Permissions).Perm())
	}

	return nil
}

func (c *CommandServer) SetGuestFileAttributes(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.SetGuestFileAttributesRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	for _, set := range []func(*vix.SetGuestFileAttributesRequest) error{chtimes, chown, chmod} {
		err = set(r)
		if err != nil {
			return nil, err
		}
	}

	return nil, nil
}

func (c *CommandServer) InitiateFileTransferFromGuest(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.ListFilesRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := c.FileServer.Stat(r.GuestPathName)
	if err != nil {
		return nil, err
	}

	if info.Mode()&os.ModeSymlink == os.ModeSymlink {
		return nil, vix.Error(vix.InvalidArg)
	}

	if info.IsDir() {
		return nil, vix.Error(vix.NotAFile)
	}

	return []byte(fileExtendedInfoFormat("", info)), nil
}

func (c *CommandServer) InitiateFileTransferToGuest(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.InitiateFileTransferToGuestRequest{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	info, err := c.FileServer.Stat(r.GuestPathName)
	if err == nil {
		if info.Mode()&os.ModeSymlink == os.ModeSymlink {
			return nil, vix.Error(vix.InvalidArg)
		}

		if info.IsDir() {
			return nil, vix.Error(vix.NotAFile)
		}

		if !r.Body.Overwrite {
			return nil, vix.Error(vix.FileAlreadyExists)
		}
	} else {
		if !os.IsNotExist(err) {
			return nil, err
		}
	}

	return nil, nil
}

func (c *CommandServer) ProcessHgfsPacket(header vix.CommandRequestHeader, data []byte) ([]byte, error) {
	r := &vix.CommandHgfsSendPacket{
		CommandRequestHeader: header,
	}

	err := r.UnmarshalBinary(data)
	if err != nil {
		return nil, err
	}

	return c.FileServer.Dispatch(r.Packet)
}

func (c *CommandServer) authenticate(r vix.CommandRequestHeader, data []byte) error {
	if c.Authenticate != nil {
		return c.Authenticate(r, data)
	}

	switch r.UserCredentialType {
	case vix.UserCredentialTypeNamePassword:
		var c vix.UserCredentialNamePassword

		if err := c.UnmarshalBinary(data); err != nil {
			return err
		}

		if Trace {
			fmt.Fprintf(traceLog, "ignoring credentials: %q:%q\n", c.Name, c.Password)
		}

		return nil
	default:
		return fmt.Errorf("unsupported UserCredentialType=%d", r.UserCredentialType)
	}
}
