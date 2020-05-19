/*
   Copyright The containerd Authors.

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

package ttrpc

import (
	"context"
	"net"
	"os"
	"syscall"

	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

type UnixCredentialsFunc func(*unix.Ucred) error

func (fn UnixCredentialsFunc) Handshake(ctx context.Context, conn net.Conn) (net.Conn, interface{}, error) {
	uc, err := requireUnixSocket(conn)
	if err != nil {
		return nil, nil, errors.Wrap(err, "ttrpc.UnixCredentialsFunc: require unix socket")
	}

	rs, err := uc.SyscallConn()
	if err != nil {
		return nil, nil, errors.Wrap(err, "ttrpc.UnixCredentialsFunc: (net.UnixConn).SyscallConn failed")
	}
	var (
		ucred    *unix.Ucred
		ucredErr error
	)
	if err := rs.Control(func(fd uintptr) {
		ucred, ucredErr = unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED)
	}); err != nil {
		return nil, nil, errors.Wrapf(err, "ttrpc.UnixCredentialsFunc: (*syscall.RawConn).Control failed")
	}

	if ucredErr != nil {
		return nil, nil, errors.Wrapf(err, "ttrpc.UnixCredentialsFunc: failed to retrieve socket peer credentials")
	}

	if err := fn(ucred); err != nil {
		return nil, nil, errors.Wrapf(err, "ttrpc.UnixCredentialsFunc: credential check failed")
	}

	return uc, ucred, nil
}

// UnixSocketRequireUidGid requires specific *effective* UID/GID, rather than the real UID/GID.
//
// For example, if a daemon binary is owned by the root (UID 0) with SUID bit but running as an
// unprivileged user (UID 1001), the effective UID becomes 0, and the real UID becomes 1001.
// So calling this function with uid=0 allows a connection from effective UID 0 but rejects
// a connection from effective UID 1001.
//
// See socket(7), SO_PEERCRED: "The returned credentials are those that were in effect at the time of the call to connect(2) or socketpair(2)."
func UnixSocketRequireUidGid(uid, gid int) UnixCredentialsFunc {
	return func(ucred *unix.Ucred) error {
		return requireUidGid(ucred, uid, gid)
	}
}

func UnixSocketRequireRoot() UnixCredentialsFunc {
	return UnixSocketRequireUidGid(0, 0)
}

// UnixSocketRequireSameUser resolves the current effective unix user and returns a
// UnixCredentialsFunc that will validate incoming unix connections against the
// current credentials.
//
// This is useful when using abstract sockets that are accessible by all users.
func UnixSocketRequireSameUser() UnixCredentialsFunc {
	euid, egid := os.Geteuid(), os.Getegid()
	return UnixSocketRequireUidGid(euid, egid)
}

func requireRoot(ucred *unix.Ucred) error {
	return requireUidGid(ucred, 0, 0)
}

func requireUidGid(ucred *unix.Ucred, uid, gid int) error {
	if (uid != -1 && uint32(uid) != ucred.Uid) || (gid != -1 && uint32(gid) != ucred.Gid) {
		return errors.Wrap(syscall.EPERM, "ttrpc: invalid credentials")
	}
	return nil
}

func requireUnixSocket(conn net.Conn) (*net.UnixConn, error) {
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return nil, errors.New("a unix socket connection is required")
	}

	return uc, nil
}
