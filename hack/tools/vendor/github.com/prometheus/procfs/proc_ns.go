// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Namespace represents a single namespace of a process.
type Namespace struct {
	Type  string // Namespace type.
	Inode uint32 // Inode number of the namespace. If two processes are in the same namespace their inodes will match.
}

// Namespaces contains all of the namespaces that the process is contained in.
type Namespaces map[string]Namespace

// Namespaces reads from /proc/<pid>/ns/* to get the namespaces of which the
// process is a member.
func (p Proc) Namespaces() (Namespaces, error) {
	d, err := os.Open(p.path("ns"))
	if err != nil {
		return nil, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return nil, fmt.Errorf("failed to read contents of ns dir: %w", err)
	}

	ns := make(Namespaces, len(names))
	for _, name := range names {
		target, err := os.Readlink(p.path("ns", name))
		if err != nil {
			return nil, err
		}

		fields := strings.SplitN(target, ":", 2)
		if len(fields) != 2 {
			return nil, fmt.Errorf("failed to parse namespace type and inode from %q", target)
		}

		typ := fields[0]
		inode, err := strconv.ParseUint(strings.Trim(fields[1], "[]"), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse inode from %q: %w", fields[1], err)
		}

		ns[name] = Namespace{typ, uint32(inode)}
	}

	return ns, nil
}
