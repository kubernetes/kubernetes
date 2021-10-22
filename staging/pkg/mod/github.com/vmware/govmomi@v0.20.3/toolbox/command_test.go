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
	"context"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vmware/govmomi/toolbox/hgfs"
	"github.com/vmware/govmomi/toolbox/vix"
)

type CommandClient struct {
	Service *Service
	Header  *vix.CommandRequestHeader
	creds   []byte
}

func NewCommandClient() *CommandClient {
	Trace = testing.Verbose()
	hgfs.Trace = Trace

	creds, _ := (&vix.UserCredentialNamePassword{
		Name:     "user",
		Password: "pass",
	}).MarshalBinary()

	header := new(vix.CommandRequestHeader)
	header.Magic = vix.CommandMagicWord

	header.UserCredentialType = vix.UserCredentialTypeNamePassword
	header.CredentialLength = uint32(len(creds))

	in := new(mockChannelIn)
	out := new(mockChannelOut)

	return &CommandClient{
		creds:   creds,
		Header:  header,
		Service: NewService(in, out),
	}
}

func (c *CommandClient) Request(op uint32, m encoding.BinaryMarshaler) []byte {
	b, err := m.MarshalBinary()
	if err != nil {
		panic(err)
	}

	c.Header.OpCode = op
	c.Header.BodyLength = uint32(len(b))

	var buf bytes.Buffer
	_, _ = buf.Write([]byte("\"reqname\"\x00"))
	_ = binary.Write(&buf, binary.LittleEndian, c.Header)

	_, _ = buf.Write(b)

	data := append(buf.Bytes(), c.creds...)
	reply, err := c.Service.Command.Dispatch(data)
	if err != nil {
		panic(err)
	}

	return reply
}

func vixRC(buf []byte) int {
	args := bytes.SplitN(buf, []byte{' '}, 2)
	rc, err := strconv.Atoi(string(args[0]))
	if err != nil {
		panic(err)
	}
	return rc
}

func TestVixRelayedCommandHandler(t *testing.T) {
	Trace = true
	if !testing.Verbose() {
		// cover Trace paths but discard output
		traceLog = ioutil.Discard
	}

	in := new(mockChannelIn)
	out := new(mockChannelOut)

	service := NewService(in, out)

	cmd := service.Command

	msg := []byte("\"reqname\"\x00")

	_, err := cmd.Dispatch(msg) // io.EOF
	if err == nil {
		t.Fatal("expected error")
	}

	header := new(vix.CommandRequestHeader)

	marshal := func(m ...encoding.BinaryMarshaler) []byte {
		var buf bytes.Buffer
		_, _ = buf.Write(msg)
		_ = binary.Write(&buf, binary.LittleEndian, header)

		for _, e := range m {
			b, err := e.MarshalBinary()
			if err != nil {
				panic(err)
			}
			_, _ = buf.Write(b)
		}

		return buf.Bytes()
	}

	// header.Magic not set
	reply, _ := cmd.Dispatch(marshal())
	rc := vixRC(reply)
	if rc != vix.InvalidMessageHeader {
		t.Fatalf("%q", reply)
	}

	// header.OpCode not set
	header.Magic = vix.CommandMagicWord
	reply, _ = cmd.Dispatch(marshal())
	rc = vixRC(reply)
	if rc != vix.UnrecognizedCommandInGuest {
		t.Fatalf("%q", reply)
	}

	// valid request for GetToolsState
	header.OpCode = vix.CommandGetToolsState
	reply, _ = cmd.Dispatch(marshal())
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("%q", reply)
	}

	// header.UserCredentialType not set
	header.OpCode = vix.CommandStartProgram
	request := new(vix.StartProgramRequest)
	buf := marshal(request)
	reply, _ = cmd.Dispatch(marshal())
	rc = vixRC(reply)
	if rc != vix.AuthenticationFail {
		t.Fatalf("%q", reply)
	}

	creds, _ := (&vix.UserCredentialNamePassword{
		Name:     "user",
		Password: "pass",
	}).MarshalBinary()

	header.BodyLength = uint32(binary.Size(request.Body))
	header.UserCredentialType = vix.UserCredentialTypeNamePassword
	header.CredentialLength = uint32(len(creds))

	// ProgramPath not set
	buf = append(marshal(request), creds...)
	reply, _ = cmd.Dispatch(buf)
	rc = vixRC(reply)
	if rc != vix.FileNotFound {
		t.Fatalf("%q", reply)
	}

	cmd.ProcessStartCommand = func(pm *ProcessManager, r *vix.StartProgramRequest) (int64, error) {
		return -1, nil
	}

	// valid request for StartProgram
	buf = append(marshal(request), creds...)
	reply, _ = cmd.Dispatch(buf)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("%q", reply)
	}

	cmd.Authenticate = func(_ vix.CommandRequestHeader, data []byte) error {
		var c vix.UserCredentialNamePassword
		if err := c.UnmarshalBinary(data); err != nil {
			panic(err)
		}

		return errors.New("you shall not pass")
	}

	// fail auth with our own handler
	buf = append(marshal(request), creds...)
	reply, _ = cmd.Dispatch(buf)
	rc = vixRC(reply)
	if rc != vix.AuthenticationFail {
		t.Fatalf("%q", reply)
	}

	cmd.Authenticate = nil

	// cause Vix.UserCredentialNamePassword.UnmarshalBinary to error
	// first by EOF reading header, second in base64 decode
	for _, l := range []uint32{1, 10} {
		header.CredentialLength = l
		buf = append(marshal(request), creds...)
		reply, _ = cmd.Dispatch(buf)
		rc = vixRC(reply)
		if rc != vix.AuthenticationFail {
			t.Fatalf("%q", reply)
		}
	}
}

// cover misc error paths
func TestVixCommandErrors(t *testing.T) {
	r := new(vix.StartProgramRequest)
	err := r.UnmarshalBinary(nil)
	if err == nil {
		t.Error("expected error")
	}

	r.Body.NumEnvVars = 1
	buf, _ := r.MarshalBinary()
	err = r.UnmarshalBinary(buf)
	if err == nil {
		t.Error("expected error")
	}

	c := new(CommandServer)
	_, err = c.StartCommand(r.CommandRequestHeader, nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestVixInitiateDirTransfer(t *testing.T) {
	c := NewCommandClient()

	dir := os.TempDir()

	for _, enable := range []bool{true, false} {
		expect := vix.NotAFile
		if enable {
			expect = vix.OK
		} else {
			// validate we behave as open-vm-tools does when the directory archive feature is disabled
			c.Service.Command.FileServer.RegisterFileHandler(hgfs.ArchiveScheme, nil)
		}

		fromGuest := &vix.ListFilesRequest{GuestPathName: dir}
		toGuest := &vix.InitiateFileTransferToGuestRequest{GuestPathName: dir}
		toGuest.Body.Overwrite = true

		tests := []struct {
			op      uint32
			request encoding.BinaryMarshaler
		}{
			{vix.CommandInitiateFileTransferFromGuest, fromGuest},
			{vix.CommandInitiateFileTransferToGuest, toGuest},
		}

		for _, test := range tests {
			reply := c.Request(test.op, test.request)

			rc := vixRC(reply)

			if rc != expect {
				t.Errorf("rc=%d", rc)
			}
		}
	}
}

func TestVixInitiateFileTransfer(t *testing.T) {
	c := NewCommandClient()

	request := new(vix.ListFilesRequest)

	f, err := ioutil.TempFile("", "toolbox")
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range []string{"a", "b", "c", "d", "e"} {
		_, _ = f.WriteString(strings.Repeat(s, 40))
	}

	_ = f.Close()

	name := f.Name()

	// 1st pass file exists == OK, 2nd pass does not exist == FAIL
	for _, fail := range []bool{false, true} {
		request.GuestPathName = name

		reply := c.Request(vix.CommandInitiateFileTransferFromGuest, request)

		rc := vixRC(reply)

		if Trace {
			fmt.Fprintf(os.Stderr, "%s: %s\n", name, string(reply))
		}

		if fail {
			if rc == vix.OK {
				t.Errorf("%s: %d", name, rc)
			}
		} else {
			if rc != vix.OK {
				t.Errorf("%s: %d", name, rc)
			}

			err = os.Remove(name)
			if err != nil {
				t.Error(err)
			}
		}
	}
}

func TestVixInitiateFileTransferWrite(t *testing.T) {
	c := NewCommandClient()

	request := new(vix.InitiateFileTransferToGuestRequest)

	f, err := ioutil.TempFile("", "toolbox")
	if err != nil {
		t.Fatal(err)
	}

	_ = f.Close()

	name := f.Name()

	tests := []struct {
		force bool
		fail  bool
	}{
		{false, true},  // exists == FAIL
		{true, false},  // exists, but overwrite == OK
		{false, false}, // does not exist == OK
	}

	for i, test := range tests {
		request.GuestPathName = name
		request.Body.Overwrite = test.force

		reply := c.Request(vix.CommandInitiateFileTransferToGuest, request)

		rc := vixRC(reply)

		if Trace {
			fmt.Fprintf(os.Stderr, "%s: %s\n", name, string(reply))
		}

		if test.fail {
			if rc == vix.OK {
				t.Errorf("%d: %d", i, rc)
			}
		} else {
			if rc != vix.OK {
				t.Errorf("%d: %d", i, rc)
			}
			if test.force {
				_ = os.Remove(name)
			}
		}
	}
}

func TestVixProcessHgfsPacket(t *testing.T) {
	c := NewCommandClient()

	c.Header.CommonFlags = vix.CommandGuestReturnsBinary

	request := new(vix.CommandHgfsSendPacket)

	op := new(hgfs.RequestCreateSessionV4)
	packet := new(hgfs.Packet)
	packet.Payload, _ = op.MarshalBinary()
	packet.Header.Version = hgfs.HeaderVersion
	packet.Header.Dummy = hgfs.OpNewHeader
	packet.Header.HeaderSize = uint32(binary.Size(&packet.Header))
	packet.Header.PacketSize = packet.Header.HeaderSize + uint32(len(packet.Payload))
	packet.Header.Op = hgfs.OpCreateSessionV4

	request.Packet, _ = packet.MarshalBinary()
	request.Body.PacketSize = uint32(len(request.Packet))

	reply := c.Request(vix.HgfsSendPacketCommand, request)

	rc := vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	ix := bytes.IndexByte(reply, '#')
	reply = reply[ix+1:]
	err := packet.UnmarshalBinary(reply)
	if err != nil {
		t.Fatal(err)
	}

	if packet.Status != hgfs.StatusSuccess {
		t.Errorf("status=%d", packet.Status)
	}

	if packet.Dummy != hgfs.OpNewHeader {
		t.Errorf("dummy=%d", packet.Dummy)
	}

	session := new(hgfs.ReplyCreateSessionV4)
	err = session.UnmarshalBinary(packet.Payload)
	if err != nil {
		t.Fatal(err)
	}

	if session.NumCapabilities == 0 || int(session.NumCapabilities) != len(session.Capabilities) {
		t.Errorf("NumCapabilities=%d", session.NumCapabilities)
	}
}

func TestVixListProcessesEx(t *testing.T) {
	c := NewCommandClient()
	pm := c.Service.Command.ProcessManager

	c.Service.Command.ProcessStartCommand = func(pm *ProcessManager, r *vix.StartProgramRequest) (int64, error) {
		var p *Process
		switch r.ProgramPath {
		case "foo":
			p = NewProcessFunc(func(ctx context.Context, arg string) error {
				return nil
			})
		default:
			return -1, os.ErrNotExist
		}

		return pm.Start(r, p)
	}

	exec := &vix.StartProgramRequest{
		ProgramPath: "foo",
	}

	reply := c.Request(vix.CommandStartProgram, exec)
	rc := vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	r := bytes.Trim(bytes.Split(reply, []byte{' '})[2], "\x00")
	pid, _ := strconv.Atoi(string(r))

	exec.ProgramPath = "bar"
	reply = c.Request(vix.CommandStartProgram, exec)
	rc = vixRC(reply)
	t.Log(vix.Error(rc).Error())
	if rc != vix.FileNotFound {
		t.Fatalf("rc: %d", rc)
	}
	if vix.ErrorCode(os.ErrNotExist) != rc {
		t.Fatalf("rc: %d", rc)
	}

	pm.wg.Wait()

	ps := new(vix.ListProcessesRequest)

	ps.Pids = []int64{int64(pid)}

	reply = c.Request(vix.CommandListProcessesEx, ps)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	n := bytes.Count(reply, []byte("<proc>"))
	if n != len(ps.Pids) {
		t.Errorf("ps -p %d=%d", pid, n)
	}

	kill := new(vix.KillProcessRequest)
	kill.Body.Pid = ps.Pids[0]

	reply = c.Request(vix.CommandTerminateProcess, kill)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	kill.Body.Pid = 33333
	reply = c.Request(vix.CommandTerminateProcess, kill)
	rc = vixRC(reply)
	if rc != vix.NoSuchProcess {
		t.Fatalf("rc: %d", rc)
	}
}

func TestVixGetenv(t *testing.T) {
	c := NewCommandClient()

	env := os.Environ()
	key := strings.SplitN(env[0], "=", 2)[0]

	tests := []struct {
		names  []string
		expect int
	}{
		{nil, len(env)},              // all env
		{[]string{key, "ENOENT"}, 1}, // specific vars, 1 exists 1 does not
	}

	for i, test := range tests {
		env := &vix.ReadEnvironmentVariablesRequest{
			Names: test.names,
		}
		reply := c.Request(vix.CommandReadEnvVariables, env)
		rc := vixRC(reply)
		if rc != vix.OK {
			t.Fatalf("%d) rc: %d", i, rc)
		}

		num := bytes.Count(reply, []byte("<ev>"))
		if num != test.expect {
			t.Errorf("%d) getenv(%v): %d", i, test.names, num)
		}
	}
}

func TestVixDirectories(t *testing.T) {
	c := NewCommandClient()

	mktemp := &vix.CreateTempFileRequest{
		FilePrefix: "toolbox-",
	}

	// mktemp -d
	reply := c.Request(vix.CommandCreateTemporaryDirectory, mktemp)
	rc := vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	dir := strings.TrimSuffix(string(reply[4:]), "\x00")

	mkdir := &vix.DirRequest{
		GuestPathName: dir,
	}

	// mkdir $dir == EEXIST
	reply = c.Request(vix.CommandCreateDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.FileAlreadyExists {
		t.Fatalf("rc: %d", rc)
	}

	// mkdir $dir/ok == OK
	mkdir.GuestPathName = dir + "/ok"
	reply = c.Request(vix.CommandCreateDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	// rm of a dir should fail, regardless if empty or not
	reply = c.Request(vix.CommandDeleteGuestFileEx, &vix.FileRequest{
		GuestPathName: mkdir.GuestPathName,
	})
	rc = vixRC(reply)
	if rc != vix.NotAFile {
		t.Errorf("rc: %d", rc)
	}

	// rmdir $dir/ok == OK
	reply = c.Request(vix.CommandDeleteGuestDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	// rmdir $dir/ok == ENOENT
	reply = c.Request(vix.CommandDeleteGuestDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.FileNotFound {
		t.Fatalf("rc: %d", rc)
	}

	// mkdir $dir/1/2 == ENOENT (parent directory does not exist)
	mkdir.GuestPathName = dir + "/1/2"
	reply = c.Request(vix.CommandCreateDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.FileNotFound {
		t.Fatalf("rc: %d", rc)
	}

	// mkdir -p $dir/1/2 == OK
	mkdir.Body.Recursive = true
	reply = c.Request(vix.CommandCreateDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	// rmdir $dir == ENOTEMPTY
	mkdir.GuestPathName = dir
	mkdir.Body.Recursive = false
	reply = c.Request(vix.CommandDeleteGuestDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.DirectoryNotEmpty {
		t.Fatalf("rc: %d", rc)
	}

	// rm -rf $dir == OK
	mkdir.Body.Recursive = true
	reply = c.Request(vix.CommandDeleteGuestDirectoryEx, mkdir)
	rc = vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}
}

func TestVixFiles(t *testing.T) {
	c := NewCommandClient()

	mktemp := &vix.CreateTempFileRequest{
		FilePrefix: "toolbox-",
	}

	// mktemp -d
	reply := c.Request(vix.CommandCreateTemporaryDirectory, mktemp)
	rc := vixRC(reply)
	if rc != vix.OK {
		t.Fatalf("rc: %d", rc)
	}

	dir := strings.TrimSuffix(string(reply[4:]), "\x00")

	max := 12
	var total int

	// mktemp
	for i := 0; i <= max; i++ {
		mktemp = &vix.CreateTempFileRequest{
			DirectoryPath: dir,
		}

		reply = c.Request(vix.CommandCreateTemporaryFileEx, mktemp)
		rc = vixRC(reply)
		if rc != vix.OK {
			t.Fatalf("rc: %d", rc)
		}
	}

	// name of the last file temp file we created, we'll mess around with it then delete it
	name := strings.TrimSuffix(string(reply[4:]), "\x00")
	// for testing symlinks
	link := filepath.Join(dir, "a-link")
	err := os.Symlink(name, link)
	if err != nil {
		t.Fatal(err)
	}

	for _, fpath := range []string{name, link} {
		// test ls of a single file
		ls := &vix.ListFilesRequest{
			GuestPathName: fpath,
		}

		reply = c.Request(vix.CommandListFiles, ls)

		rc = vixRC(reply)
		if rc != vix.OK {
			t.Fatalf("rc: %d", rc)
		}

		num := bytes.Count(reply, []byte("<fxi>"))
		if num != 1 {
			t.Errorf("ls %s: %d", name, num)
		}

		num = bytes.Count(reply, []byte("<rem>0</rem>"))
		if num != 1 {
			t.Errorf("ls %s: %d", name, num)
		}

		ft := 0
		target := ""
		if fpath == link {
			target = name
			ft = vix.FileAttributesSymlink
		}

		num = bytes.Count(reply, []byte(fmt.Sprintf("<slt>%s</slt>", target)))
		if num != 1 {
			t.Errorf("ls %s: %d", name, num)
		}

		num = bytes.Count(reply, []byte(fmt.Sprintf("<ft>%d</ft>", ft)))
		if num != 1 {
			t.Errorf("ls %s: %d", name, num)
		}
	}

	mv := &vix.RenameFileRequest{
		OldPathName: name,
		NewPathName: name + "-new",
	}

	for _, expect := range []int{vix.OK, vix.FileNotFound} {
		reply = c.Request(vix.CommandMoveGuestFileEx, mv)
		rc = vixRC(reply)
		if rc != expect {
			t.Errorf("rc: %d", rc)
		}

		if expect == vix.OK {
			// test file type is properly checked
			reply = c.Request(vix.CommandMoveGuestDirectory, &vix.RenameFileRequest{
				OldPathName: mv.NewPathName,
				NewPathName: name,
			})
			rc = vixRC(reply)
			if rc != vix.NotADirectory {
				t.Errorf("rc: %d", rc)
			}

			// test Overwrite flag is properly checked
			reply = c.Request(vix.CommandMoveGuestFileEx, &vix.RenameFileRequest{
				OldPathName: mv.NewPathName,
				NewPathName: mv.NewPathName,
			})
			rc = vixRC(reply)
			if rc != vix.FileAlreadyExists {
				t.Errorf("rc: %d", rc)
			}
		}
	}

	// rmdir of a file should fail
	reply = c.Request(vix.CommandDeleteGuestDirectoryEx, &vix.DirRequest{
		GuestPathName: mv.NewPathName,
	})

	rc = vixRC(reply)
	if rc != vix.NotADirectory {
		t.Errorf("rc: %d", rc)
	}

	file := &vix.FileRequest{
		GuestPathName: mv.NewPathName,
	}

	for _, expect := range []int{vix.OK, vix.FileNotFound} {
		reply = c.Request(vix.CommandDeleteGuestFileEx, file)
		rc = vixRC(reply)
		if rc != expect {
			t.Errorf("rc: %d", rc)
		}
	}

	// ls again now that file is gone
	reply = c.Request(vix.CommandListFiles, &vix.ListFilesRequest{
		GuestPathName: name,
	})

	rc = vixRC(reply)
	if rc != vix.FileNotFound {
		t.Errorf("rc: %d", rc)
	}

	// ls
	ls := &vix.ListFilesRequest{
		GuestPathName: dir,
	}
	ls.Body.MaxResults = 5 // default is 50

	for i := 0; i < 5; i++ {
		reply = c.Request(vix.CommandListFiles, ls)

		if Trace {
			fmt.Fprintf(os.Stderr, "%s: %q\n", dir, string(reply[4:]))
		}

		var rem int
		_, err := fmt.Fscanf(bytes.NewReader(reply[4:]), "<rem>%d</rem>", &rem)
		if err != nil {
			t.Fatal(err)
		}

		num := bytes.Count(reply, []byte("<fxi>"))
		total += num
		ls.Body.Offset += uint64(num)

		if rem == 0 {
			break
		}
	}

	if total != max+1 {
		t.Errorf("expected %d, got %d", max, total)
	}

	// Test invalid offset, making sure it doesn't cause panic (issue #934)
	ls.Body.Offset += 10
	_ = c.Request(vix.CommandListFiles, ls)

	// mv $dir ${dir}-old
	mv = &vix.RenameFileRequest{
		OldPathName: dir,
		NewPathName: dir + "-old",
	}

	for _, expect := range []int{vix.OK, vix.FileNotFound} {
		reply = c.Request(vix.CommandMoveGuestDirectory, mv)
		rc = vixRC(reply)
		if rc != expect {
			t.Errorf("rc: %d", rc)
		}

		if expect == vix.OK {
			// test file type is properly checked
			reply = c.Request(vix.CommandMoveGuestFileEx, &vix.RenameFileRequest{
				OldPathName: mv.NewPathName,
				NewPathName: dir,
			})
			rc = vixRC(reply)
			if rc != vix.NotAFile {
				t.Errorf("rc: %d", rc)
			}

			// test Overwrite flag is properly checked
			reply = c.Request(vix.CommandMoveGuestDirectory, &vix.RenameFileRequest{
				OldPathName: mv.NewPathName,
				NewPathName: mv.NewPathName,
			})
			rc = vixRC(reply)
			if rc != vix.FileAlreadyExists {
				t.Errorf("rc: %d", rc)
			}
		}
	}

	rmdir := &vix.DirRequest{
		GuestPathName: mv.NewPathName,
	}

	// rm -rm $dir
	for _, rmr := range []bool{false, true} {
		rmdir.Body.Recursive = rmr

		reply = c.Request(vix.CommandDeleteGuestDirectoryEx, rmdir)
		rc = vixRC(reply)
		if rmr {
			if rc != vix.OK {
				t.Fatalf("rc: %d", rc)
			}
		} else {
			if rc != vix.DirectoryNotEmpty {
				t.Fatalf("rc: %d", rc)
			}
		}
	}
}

func TestVixFileChangeAttributes(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("running as root")
	}

	c := NewCommandClient()

	f, err := ioutil.TempFile("", "toolbox-")
	if err != nil {
		t.Fatal(err)
	}
	_ = f.Close()
	name := f.Name()

	// touch,chown,chmod
	chattr := &vix.SetGuestFileAttributesRequest{
		GuestPathName: name,
	}

	h := &chattr.Body

	tests := []struct {
		expect int
		f      func()
	}{
		{
			vix.OK, func() {},
		},
		{
			vix.OK, func() {
				h.FileOptions = vix.FileAttributeSetModifyDate
				h.ModificationTime = time.Now().Unix()
			},
		},
		{
			vix.OK, func() {
				h.FileOptions = vix.FileAttributeSetAccessDate
				h.AccessTime = time.Now().Unix()
			},
		},
		{
			vix.FileAccessError, func() {
				h.FileOptions = vix.FileAttributeSetUnixOwnerid
				h.OwnerID = 0 // fails as we are not root
			},
		},
		{
			vix.FileAccessError, func() {
				h.FileOptions = vix.FileAttributeSetUnixGroupid
				h.GroupID = 0 // fails as we are not root
			},
		},
		{
			vix.OK, func() {
				h.FileOptions = vix.FileAttributeSetUnixOwnerid
				h.OwnerID = int32(os.Getuid())
			},
		},
		{
			vix.OK, func() {
				h.FileOptions = vix.FileAttributeSetUnixGroupid
				h.GroupID = int32(os.Getgid())
			},
		},
		{
			vix.OK, func() {
				h.FileOptions = vix.FileAttributeSetUnixPermissions
				h.Permissions = int32(os.FileMode(0755).Perm())
			},
		},
		{
			vix.FileNotFound, func() {
				_ = os.Remove(name)
			},
		},
	}

	for i, test := range tests {
		test.f()
		reply := c.Request(vix.CommandSetGuestFileAttributes, chattr)
		rc := vixRC(reply)

		if rc != test.expect {
			t.Errorf("%d: rc=%d", i, rc)
		}
	}
}
