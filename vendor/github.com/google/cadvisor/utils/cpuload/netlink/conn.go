// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build linux

package netlink

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"os"
	"syscall"
)

type Connection struct {
	// netlink socket
	fd int
	// cache pid to use in every netlink request.
	pid uint32
	// sequence number for netlink messages.
	seq  uint32
	addr syscall.SockaddrNetlink
	rbuf *bufio.Reader
}

// Create and bind a new netlink socket.
func newConnection() (*Connection, error) {

	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_DGRAM, syscall.NETLINK_GENERIC)
	if err != nil {
		return nil, err
	}

	conn := new(Connection)
	conn.fd = fd
	conn.seq = 0
	conn.pid = uint32(os.Getpid())
	conn.addr.Family = syscall.AF_NETLINK
	conn.rbuf = bufio.NewReader(conn)
	err = syscall.Bind(fd, &conn.addr)
	if err != nil {
		syscall.Close(fd)
		return nil, err
	}
	return conn, err
}

func (c *Connection) Read(b []byte) (n int, err error) {
	n, _, err = syscall.Recvfrom(c.fd, b, 0)
	return n, err
}

func (c *Connection) Write(b []byte) (n int, err error) {
	err = syscall.Sendto(c.fd, b, 0, &c.addr)
	return len(b), err
}

func (c *Connection) Close() error {
	return syscall.Close(c.fd)
}

func (c *Connection) WriteMessage(msg syscall.NetlinkMessage) error {
	w := bytes.NewBuffer(nil)
	msg.Header.Len = uint32(syscall.NLMSG_HDRLEN + len(msg.Data))
	msg.Header.Seq = c.seq
	c.seq++
	msg.Header.Pid = c.pid
	err := binary.Write(w, binary.LittleEndian, msg.Header)
	if err != nil {
		return err
	}
	_, err = w.Write(msg.Data)
	if err != nil {
		return err
	}
	_, err = c.Write(w.Bytes())
	return err
}

func (c *Connection) ReadMessage() (msg syscall.NetlinkMessage, err error) {
	err = binary.Read(c.rbuf, binary.LittleEndian, &msg.Header)
	if err != nil {
		return msg, err
	}
	msg.Data = make([]byte, msg.Header.Len-syscall.NLMSG_HDRLEN)
	_, err = c.rbuf.Read(msg.Data)
	return msg, err
}
