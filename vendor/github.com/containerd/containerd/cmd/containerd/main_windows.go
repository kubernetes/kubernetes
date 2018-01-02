package main

import (
	"context"
	"os"
	"path/filepath"

	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/server"

	"golang.org/x/sys/windows"
)

var (
	defaultConfigPath = filepath.Join(os.Getenv("programfiles"), "containerd", "config.toml")
	handledSignals    = []os.Signal{
		windows.SIGTERM,
		windows.SIGINT,
	}
)

func handleSignals(ctx context.Context, signals chan os.Signal, serverC chan *server.Server) chan struct{} {
	done := make(chan struct{})
	go func() {
		var server *server.Server
		for {
			select {
			case s := <-serverC:
				server = s
			case s := <-signals:
				log.G(ctx).WithField("signal", s).Debug("received signal")
				if server == nil {
					close(done)
					return
				}
				server.Stop()
				close(done)
			}
		}
	}()
	return done
}
