// Copyright 2016 The etcd Authors
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

package transport

import (
	"net"
	"os"
)

type unixListener struct{ net.Listener }

func NewUnixListener(addr string) (net.Listener, error) {
	if err := os.RemoveAll(addr); err != nil {
		return nil, err
	}
	l, err := net.Listen("unix", addr)
	if err != nil {
		return nil, err
	}
	return &unixListener{l}, nil
}

func (ul *unixListener) Close() error {
	if err := os.RemoveAll(ul.Addr().String()); err != nil {
		return err
	}
	return ul.Listener.Close()
}
