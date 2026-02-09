/*
Copyright 2025 The Kubernetes Authors.

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

package kubeletplugin

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path"
)

// endpoint defines where and how to listen for incoming connections.
// The listener always gets closed when shutting down.
//
// If the listen function is not set, a new listener for a Unix domain socket gets
// created at the path.
type endpoint struct {
	dir, file  string
	listenFunc func(ctx context.Context, socketpath string) (net.Listener, error)
}

func (e endpoint) path() string {
	return path.Join(e.dir, e.file)
}

func (e endpoint) listen(ctx context.Context) (net.Listener, error) {
	socketpath := e.path()

	if e.listenFunc != nil {
		return e.listenFunc(ctx, socketpath)
	}

	// Remove stale sockets, listen would fail otherwise.
	if err := e.removeSocket(); err != nil {
		return nil, err
	}
	cfg := net.ListenConfig{}
	listener, err := cfg.Listen(ctx, "unix", socketpath)
	if err != nil {
		if removeErr := e.removeSocket(); removeErr != nil {
			err = errors.Join(err, err)
		}
		return nil, err
	}
	return &unixListener{Listener: listener, endpoint: e}, nil
}

func (e endpoint) removeSocket() error {
	if err := os.Remove(e.path()); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove Unix domain socket: %w", err)
	}
	return nil
}

// unixListener adds removing of the Unix domain socket on Close.
type unixListener struct {
	net.Listener
	endpoint endpoint
}

func (l *unixListener) Close() error {
	err := l.Listener.Close()
	if removeErr := l.endpoint.removeSocket(); removeErr != nil {
		err = errors.Join(err, err)
	}
	return err
}
