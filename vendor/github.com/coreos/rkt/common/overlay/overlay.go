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

package overlay

import (
	"fmt"
	"strings"
	"syscall"

	"github.com/coreos/rkt/pkg/label"
	"github.com/hashicorp/errwrap"
)

// sanitizer defines a string translator used to escape colon and comma
// characters in the directories names.
var sanitizer = strings.NewReplacer(`:`, `\:`, `,`, `\,`)

// MountCfg contains the needed data to construct the overlay mount syscall.
// The Lower and Upper fields are paths to the filesystems to be merged. The
// Work field should be an empty directory. Dest is where the mount will be
// located. Lbl is an SELinux label.
type MountCfg struct {
	Lower,
	Upper,
	Work,
	Dest,
	Lbl string
}

// sanitize escapes the colon and comma symbols in order to support the dir
// names with these characters, otherwise they will be treated as separators
// between the directory names.
func sanitize(dir string) string {
	return sanitizer.Replace(dir)
}

// Opts returns options for mount system call.
func (cfg *MountCfg) Opts() string {
	opts := fmt.Sprintf(
		"lowerdir=%s,upperdir=%s,workdir=%s",
		sanitize(cfg.Lower), sanitize(cfg.Upper), sanitize(cfg.Work),
	)

	return label.FormatMountLabel(opts, cfg.Lbl)
}

// Mount mounts the upper and lower directories to the destination directory.
// The MountCfg struct supplies information required to build the mount system
// call.
func Mount(cfg *MountCfg) error {
	err := syscall.Mount("overlay", cfg.Dest, "overlay", 0, cfg.Opts())
	if err != nil {
		const text = "error mounting overlay with options '%s' and dest '%s'"
		return errwrap.Wrap(fmt.Errorf(text, cfg.Opts(), cfg.Dest), err)
	}

	return nil
}
