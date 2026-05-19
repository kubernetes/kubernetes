// Copyright 2022 The Prometheus Authors
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
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

func sysctlToPath(sysctl string) string {
	return strings.ReplaceAll(sysctl, ".", "/")
}

func (fs FS) SysctlStrings(sysctl string) ([]string, error) {
	value, err := util.SysReadFile(fs.proc.Path("sys", sysctlToPath(sysctl)))
	if err != nil {
		return nil, err
	}
	return strings.Fields(value), nil

}

func (fs FS) SysctlInts(sysctl string) ([]int, error) {
	fields, err := fs.SysctlStrings(sysctl)
	if err != nil {
		return nil, err
	}

	values := make([]int, len(fields))
	for i, f := range fields {
		vp := util.NewValueParser(f)
		values[i] = vp.Int()
		if err := vp.Err(); err != nil {
			return nil, fmt.Errorf("%w: field %d in sysctl %s is not a valid int: %w", ErrFileParse, i, sysctl, err)
		}
	}
	return values, nil
}
