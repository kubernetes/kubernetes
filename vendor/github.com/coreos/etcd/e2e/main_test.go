// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package e2e

import (
	"flag"
	"os"
	"runtime"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

var binDir string
var certDir string

func TestMain(m *testing.M) {
	os.Setenv("ETCD_UNSUPPORTED_ARCH", runtime.GOARCH)
	os.Unsetenv("ETCDCTL_API")

	flag.StringVar(&binDir, "bin-dir", "../bin", "The directory for store etcd and etcdctl binaries.")
	flag.StringVar(&certDir, "cert-dir", "../integration/fixtures", "The directory for store certificate files.")
	flag.Parse()

	v := m.Run()
	if v == 0 && testutil.CheckLeakedGoroutine() {
		os.Exit(1)
	}
	os.Exit(v)
}
