// Copyright 2016 The rkt Authors
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

package common

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_common "github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/fs"
)

/*
 Bind-mount the hosts /etc/resolv.conf in to the stage1's /etc/rkt-resolv.conf.
 That file will then be bind-mounted in to the stage2 by perpare-app.c
*/
func UseHostResolv(mnt fs.MountUnmounter, podRoot string) error {
	return BindMount(
		mnt,
		"/etc/resolv.conf",
		filepath.Join(_common.Stage1RootfsPath(podRoot), "etc/rkt-resolv.conf"),
		true)
}

/*
 Bind-mount the hosts /etc/hosts in to the stage1's /etc/rkt-hosts
 That file will then be bind-mounted in to the stage2 by perpare-app.c
*/
func UseHostHosts(mnt fs.MountUnmounter, podRoot string) error {
	return BindMount(
		mnt,
		"/etc/hosts",
		filepath.Join(_common.Stage1RootfsPath(podRoot), "etc/rkt-hosts"),
		true)
}

// AddHostsEntry adds an entry to an *existing* hosts file, appending
// to the existing IP if needed
func AddHostsEntry(filename string, ip string, hostname string) error {
	fp, err := os.OpenFile(filename, os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer fp.Close()

	out := ""
	found := false

	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Fields(line)
		if !found && len(words) > 0 && words[0] == ip {
			found = true
			out += fmt.Sprintf("%s %s\n", line, hostname)
		} else {
			out += line
			out += "\n"
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	// If that IP was not found, add a new line
	if !found {
		out += fmt.Sprintf("%s\t%s\n", ip, hostname)
	}

	// Seek to the beginning, truncate, and write again
	if _, err := fp.Seek(0, 0); err != nil {
		return err
	}
	if err := fp.Truncate(0); err != nil {
		return err
	}
	if _, err := fp.Write([]byte(out)); err != nil {
		return err
	}

	return nil
}
