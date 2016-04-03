// +build windows

package server

import (
	"errors"
	"net"
	"net/http"

	"github.com/docker/docker/daemon"
	"github.com/docker/docker/pkg/version"
	"github.com/docker/docker/runconfig"
)

// NewServer sets up the required Server and does protocol specific checking.
func (s *Server) newServer(proto, addr string) ([]serverCloser, error) {
	var (
		ls []net.Listener
	)
	switch proto {
	case "tcp":
		l, err := s.initTcpSocket(addr)
		if err != nil {
			return nil, err
		}
		ls = append(ls, l)

	default:
		return nil, errors.New("Invalid protocol format. Windows only supports tcp.")
	}

	var res []serverCloser
	for _, l := range ls {
		res = append(res, &HttpServer{
			&http.Server{
				Addr:    addr,
				Handler: s.router,
			},
			l,
		})
	}
	return res, nil

}

func (s *Server) AcceptConnections(d *daemon.Daemon) {
	s.daemon = d
	s.registerSubRouter()
	// close the lock so the listeners start accepting connections
	select {
	case <-s.start:
	default:
		close(s.start)
	}
}

func allocateDaemonPort(addr string) error {
	return nil
}

func adjustCpuShares(version version.Version, hostConfig *runconfig.HostConfig) {
}
