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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"testing"
)

type Client struct {
	s         *Server
	SessionID uint64
}

func NewClient() *Client {
	s := NewServer()

	return &Client{
		s: s,
	}
}

func (c *Client) Dispatch(op int32, req interface{}, res interface{}) *Packet {
	var err error
	p := new(Packet)
	p.Payload, err = MarshalBinary(req)
	if err != nil {
		panic(err)
	}

	p.Header.Version = 0x1
	p.Header.Dummy = OpNewHeader
	p.Header.HeaderSize = headerSize
	p.Header.PacketSize = headerSize + uint32(len(p.Payload))
	p.Header.SessionID = c.SessionID
	p.Header.Op = op

	data, err := p.MarshalBinary()
	if err != nil {
		panic(err)
	}

	data, err = c.s.Dispatch(data)
	if err != nil {
		panic(err)
	}

	p = new(Packet)
	err = p.UnmarshalBinary(data)
	if err != nil {
		panic(err)
	}

	if p.Status == StatusSuccess {
		err = UnmarshalBinary(p.Payload, res)
		if err != nil {
			panic(err)
		}
	}

	return p
}

func (c *Client) CreateSession() uint32 {
	req := new(RequestCreateSessionV4)
	res := new(ReplyCreateSessionV4)

	p := c.Dispatch(OpCreateSessionV4, req, res)

	if p.Status == StatusSuccess {
		c.SessionID = res.SessionID
	}

	return p.Status
}

func (c *Client) DestroySession() uint32 {
	req := new(RequestDestroySessionV4)
	res := new(ReplyDestroySessionV4)

	return c.Dispatch(OpDestroySessionV4, req, res).Status
}

func (c *Client) GetAttr(name string) (*AttrV2, uint32) {
	req := new(RequestGetattrV2)
	res := new(ReplyGetattrV2)

	req.FileName.FromString(name)

	p := c.Dispatch(OpGetattrV2, req, res)

	if p.Status != StatusSuccess {
		return nil, p.Status
	}

	return &res.Attr, p.Status
}

func (c *Client) SetAttr(name string, attr AttrV2) uint32 {
	req := new(RequestSetattrV2)
	res := new(ReplySetattrV2)

	req.FileName.FromString(name)

	req.Attr = attr

	p := c.Dispatch(OpSetattrV2, req, res)

	if p.Status != StatusSuccess {
		return p.Status
	}

	return p.Status
}

func (c *Client) Open(name string, write ...bool) (uint32, uint32) {
	req := new(RequestOpen)
	res := new(ReplyOpen)

	if len(write) == 1 && write[0] {
		req.OpenMode = OpenModeWriteOnly
	}

	req.FileName.FromString(name)

	p := c.Dispatch(OpOpen, req, res)
	if p.Status != StatusSuccess {
		return 0, p.Status
	}

	return res.Handle, p.Status
}

func (c *Client) OpenWrite(name string) (uint32, uint32) {
	req := new(RequestOpenV3)
	res := new(ReplyOpenV3)

	req.OpenMode = OpenModeWriteOnly
	req.OpenFlags = OpenCreateEmpty
	req.FileName.FromString(name)

	p := c.Dispatch(OpOpenV3, req, res)
	if p.Status != StatusSuccess {
		return 0, p.Status
	}

	// cover the unsupported lock type path
	req.DesiredLock = LockOpportunistic
	status := c.Dispatch(OpOpenV3, req, res).Status
	if status != StatusOperationNotSupported {
		return 0, status
	}

	// cover the unsupported open mode path
	req.DesiredLock = LockNone
	req.OpenMode = OpenCreateSafe
	status = c.Dispatch(OpOpenV3, req, res).Status
	if status != StatusAccessDenied {
		return 0, status
	}

	return res.Handle, p.Status
}

func (c *Client) Close(handle uint32) uint32 {
	req := new(RequestClose)
	res := new(ReplyClose)

	req.Handle = handle

	return c.Dispatch(OpClose, req, res).Status
}

func TestStaleSession(t *testing.T) {
	c := NewClient()

	// list of methods that can return StatusStaleSession
	invalid := []func() uint32{
		func() uint32 { _, status := c.Open("enoent"); return status },
		func() uint32 { return c.Dispatch(OpReadV3, new(RequestReadV3), new(ReplyReadV3)).Status },
		func() uint32 { return c.Dispatch(OpWriteV3, new(RequestWriteV3), new(ReplyWriteV3)).Status },
		func() uint32 { return c.Close(0) },
		c.DestroySession,
	}

	for i, f := range invalid {
		status := f()
		if status != StatusStaleSession {
			t.Errorf("%d: status=%d", i, status)
		}
	}
}

func TestSessionMax(t *testing.T) {
	c := NewClient()
	var status uint32

	for i := 0; i <= maxSessions+1; i++ {
		status = c.CreateSession()
	}

	if status != StatusTooManySessions {
		t.Errorf("status=%d", status)
	}
}

func TestSessionDestroy(t *testing.T) {
	Trace = true
	c := NewClient()
	c.CreateSession()
	_, status := c.Open("/etc/resolv.conf")
	if status != StatusSuccess {
		t.Errorf("status=%d", status)
	}
	c.DestroySession()

	if c.s.removeSession(c.SessionID) {
		t.Error("session was not removed")
	}
}

func TestInvalidOp(t *testing.T) {
	c := NewClient()
	status := c.Dispatch(1024, new(RequestClose), new(ReplyClose)).Status
	if status != StatusOperationNotSupported {
		t.Errorf("status=%d", status)
	}
}

func TestReadV3(t *testing.T) {
	Trace = testing.Verbose()

	c := NewClient()
	c.CreateSession()

	_, status := c.GetAttr("enoent")

	if status != StatusNoSuchFileOrDir {
		t.Errorf("status=%d", status)
	}

	_, status = c.Open("enoent")
	if status != StatusNoSuchFileOrDir {
		t.Errorf("status=%d", status)
	}

	fname := "/etc/resolv.conf"

	attr, _ := c.GetAttr(path.Dir(fname))
	if attr.Type != FileTypeDirectory {
		t.Errorf("type=%d", attr.Type)
	}

	attr, _ = c.GetAttr(fname)
	if attr.Type != FileTypeRegular {
		t.Errorf("type=%d", attr.Type)
	}

	if attr.Size <= 0 {
		t.Errorf("size=%d", attr.Size)
	}

	handle, status := c.Open(fname)
	if status != StatusSuccess {
		t.Fatalf("status=%d", status)
	}

	var req *RequestReadV3
	var offset uint64
	size := uint32(attr.Size / 2)

	for offset = 0; offset < attr.Size; {
		req = &RequestReadV3{
			Offset:       offset,
			Handle:       handle,
			RequiredSize: size,
		}

		res := new(ReplyReadV3)

		status = c.Dispatch(OpReadV3, req, res).Status

		if status != StatusSuccess {
			t.Fatalf("status=%d", status)
		}

		if Trace {
			fmt.Fprintf(os.Stderr, "read %d: %q\n", res.ActualSize, string(res.Payload))
		}

		offset += uint64(res.ActualSize)
	}

	if uint64(offset) != attr.Size {
		t.Errorf("size %d vs %d", offset, attr.Size)
	}

	status = c.Dispatch(OpReadV3, new(RequestReadV3), new(ReplyReadV3)).Status
	if status != StatusInvalidHandle {
		t.Fatalf("status=%d", status)
	}

	status = c.Close(0)
	if status != StatusInvalidHandle {
		t.Fatalf("status=%d", status)
	}

	status = c.Close(handle)
	if status != StatusSuccess {
		t.Fatalf("status=%d", status)
	}

	status = c.DestroySession()
	if status != StatusSuccess {
		t.Fatalf("status=%d", status)
	}
}

func TestWriteV3(t *testing.T) {
	Trace = testing.Verbose()

	f, err := ioutil.TempFile("", "toolbox")
	if err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	name := f.Name()

	c := NewClient()
	c.CreateSession()

	_, status := c.Open("enoent", true)
	// write not supported yet
	if status != StatusAccessDenied {
		t.Errorf("status=%d", status)
	}

	handle, status := c.OpenWrite(name)
	if status != StatusSuccess {
		t.Fatalf("status=%d", status)
	}

	payload := []byte("one two three\n")
	size := uint32(len(payload))

	req := &RequestWriteV3{
		Handle:       handle,
		WriteFlags:   WriteAppend,
		Offset:       0,
		RequiredSize: size,
		Payload:      payload,
	}

	res := new(ReplyReadV3)

	status = c.Dispatch(OpWriteV3, req, res).Status

	if status != StatusSuccess {
		t.Errorf("status=%d", status)
	}

	if size != res.ActualSize {
		t.Errorf("%d vs %d", size, res.ActualSize)
	}

	status = c.Dispatch(OpWriteV3, new(RequestWriteV3), new(ReplyWriteV3)).Status
	if status != StatusInvalidHandle {
		t.Fatalf("status=%d", status)
	}

	status = c.Close(handle)
	if status != StatusSuccess {
		t.Errorf("status=%d", status)
	}

	attr, _ := c.GetAttr(name)
	if attr.Size != uint64(size) {
		t.Errorf("%d vs %d", size, attr.Size)
	}

	attr.OwnerPerms |= PermExec

	errors := []struct {
		err    error
		status uint32
	}{
		{os.ErrPermission, StatusOperationNotPermitted},
		{os.ErrNotExist, StatusNoSuchFileOrDir},
		{os.ErrExist, StatusFileExists},
		{nil, StatusSuccess},
	}

	for _, e := range errors {
		c.s.chown = func(_ string, _ int, _ int) error {
			return e.err
		}

		status = c.SetAttr(name, *attr)
		if status != e.status {
			t.Errorf("status=%d", status)
		}
	}

	c.s.chown = func(_ string, _ int, _ int) error {
		return nil
	}

	for _, e := range errors {
		c.s.chmod = func(_ string, _ os.FileMode) error {
			return e.err
		}

		status = c.SetAttr(name, *attr)
		if status != e.status {
			t.Errorf("status=%d", status)
		}
	}

	status = c.DestroySession()
	if status != StatusSuccess {
		t.Errorf("status=%d", status)
	}

	_ = os.Remove(name)
}
