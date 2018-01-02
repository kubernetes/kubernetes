// +build linux darwin freebsd solaris

package main

import (
	"context"
	"os"

	"golang.org/x/sys/unix"

	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/reaper"
	"github.com/containerd/containerd/server"
)

const defaultConfigPath = "/etc/containerd/config.toml"

var handledSignals = []os.Signal{
	unix.SIGTERM,
	unix.SIGINT,
	unix.SIGUSR1,
	unix.SIGCHLD,
	unix.SIGPIPE,
}

func handleSignals(ctx context.Context, signals chan os.Signal, serverC chan *server.Server) chan struct{} {
	done := make(chan struct{}, 1)
	go func() {
		var server *server.Server
		for {
			select {
			case s := <-serverC:
				server = s
			case s := <-signals:
				log.G(ctx).WithField("signal", s).Debug("received signal")
				switch s {
				case unix.SIGCHLD:
					if err := reaper.Reap(); err != nil {
						log.G(ctx).WithError(err).Error("reap containerd processes")
					}
				case unix.SIGUSR1:
					dumpStacks()
				case unix.SIGPIPE:
					continue
				default:
					if server == nil {
						close(done)
						return
					}
					server.Stop()
					close(done)
				}
			}
		}
	}()
	return done
}
