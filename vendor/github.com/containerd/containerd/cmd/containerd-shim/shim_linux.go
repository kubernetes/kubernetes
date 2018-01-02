package main

import (
	"net"
	"os"
	"os/signal"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"golang.org/x/net/context"
	"golang.org/x/sys/unix"

	"github.com/containerd/containerd/reaper"
	"github.com/containerd/containerd/sys"
	runc "github.com/containerd/go-runc"
	"github.com/pkg/errors"
)

// setupSignals creates a new signal handler for all signals and sets the shim as a
// sub-reaper so that the container processes are reparented
func setupSignals() (chan os.Signal, error) {
	signals := make(chan os.Signal, 2048)
	signal.Notify(signals)
	// make sure runc is setup to use the monitor
	// for waiting on processes
	runc.Monitor = reaper.Default
	// set the shim as the subreaper for all orphaned processes created by the container
	if err := sys.SetSubreaper(1); err != nil {
		return nil, err
	}
	return signals, nil
}

func newServer() *grpc.Server {
	return grpc.NewServer(grpc.Creds(NewUnixSocketCredentials(0, 0)))
}

type unixSocketCredentials struct {
	uid        int
	gid        int
	serverName string
}

// NewUnixSocketCredentials returns TransportCredentials for a local unix socket
func NewUnixSocketCredentials(uid, gid int) credentials.TransportCredentials {
	return &unixSocketCredentials{uid, gid, "locahost"}
}

func (u *unixSocketCredentials) ClientHandshake(ctx context.Context, addr string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errors.New("ClientHandshake is not supported by unixSocketCredentials")
}

func (u *unixSocketCredentials) ServerHandshake(c net.Conn) (net.Conn, credentials.AuthInfo, error) {
	uc, ok := c.(*net.UnixConn)
	if !ok {
		return nil, nil, errors.New("unixSocketCredentials only supports unix socket")
	}

	f, err := uc.File()
	if err != nil {
		return nil, nil, errors.Wrap(err, "unixSocketCredentials: failed to retrieve connection underlying fd")
	}
	pcred, err := unix.GetsockoptUcred(int(f.Fd()), unix.SOL_SOCKET, unix.SO_PEERCRED)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unixSocketCredentials: failed to retrieve socket peer credentials")
	}

	if (u.uid != -1 && uint32(u.uid) != pcred.Uid) || (u.gid != -1 && uint32(u.gid) != pcred.Gid) {
		return nil, nil, errors.New("unixSocketCredentials: invalid credentials")
	}

	return c, u, nil
}

func (u *unixSocketCredentials) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{
		SecurityProtocol: "unix-socket-peer-creds",
		SecurityVersion:  "1.0",
		ServerName:       u.serverName,
	}
}

func (u *unixSocketCredentials) Clone() credentials.TransportCredentials {
	return &unixSocketCredentials{u.uid, u.gid, u.serverName}
}

func (u *unixSocketCredentials) OverrideServerName(serverName string) error {
	u.serverName = serverName
	return nil
}

func (u *unixSocketCredentials) AuthType() string {
	return "unix-socket-peer-creds"
}
