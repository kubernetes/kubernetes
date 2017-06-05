// Copyright 2015 The etcd Authors
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

package main

import (
	"flag"
	"os"
	"path/filepath"

	"github.com/coreos/pkg/capnslog"
)

var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcd-agent")

func main() {
	etcdPath := flag.String("etcd-path", filepath.Join(os.Getenv("GOPATH"), "bin/etcd"), "the path to etcd binary")
	etcdLogDir := flag.String("etcd-log-dir", "etcd-log", "directory to store etcd logs")
	port := flag.String("port", ":9027", "port to serve agent server")
	flag.Parse()

	a, err := newAgent(*etcdPath, *etcdLogDir)
	if err != nil {
		plog.Fatal(err)
	}
	a.serveRPC(*port)

	var done chan struct{}
	<-done
}
