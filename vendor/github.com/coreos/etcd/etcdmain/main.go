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

package etcdmain

import (
	"fmt"
	"os"

	"github.com/coreos/go-systemd/daemon"
	systemdutil "github.com/coreos/go-systemd/util"
)

func Main() {
	checkSupportArch()

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "gateway", "grpc-proxy":
			if err := rootCmd.Execute(); err != nil {
				fmt.Fprint(os.Stderr, err)
				os.Exit(1)
			}
			return
		}
	}

	startEtcdOrProxyV2()
}

func notifySystemd() {
	if !systemdutil.IsRunningSystemd() {
		return
	}
	sent, err := daemon.SdNotify(false, "READY=1")
	if err != nil {
		plog.Errorf("failed to notify systemd for readiness: %v", err)
	}
	if !sent {
		plog.Errorf("forgot to set Type=notify in systemd service file?")
	}
}
