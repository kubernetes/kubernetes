/*
   Copyright The containerd Authors.

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

package cgroups

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	v1 "github.com/containerd/cgroups/stats/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func NewHugetlb(root string) (*hugetlbController, error) {
	sizes, err := hugePageSizes()
	if err != nil {
		return nil, err
	}

	return &hugetlbController{
		root:  filepath.Join(root, string(Hugetlb)),
		sizes: sizes,
	}, nil
}

type hugetlbController struct {
	root  string
	sizes []string
}

func (h *hugetlbController) Name() Name {
	return Hugetlb
}

func (h *hugetlbController) Path(path string) string {
	return filepath.Join(h.root, path)
}

func (h *hugetlbController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(h.Path(path), defaultDirPerm); err != nil {
		return err
	}
	for _, limit := range resources.HugepageLimits {
		if err := retryingWriteFile(
			filepath.Join(h.Path(path), strings.Join([]string{"hugetlb", limit.Pagesize, "limit_in_bytes"}, ".")),
			[]byte(strconv.FormatUint(limit.Limit, 10)),
			defaultFilePerm,
		); err != nil {
			return err
		}
	}
	return nil
}

func (h *hugetlbController) Stat(path string, stats *v1.Metrics) error {
	for _, size := range h.sizes {
		s, err := h.readSizeStat(path, size)
		if err != nil {
			return err
		}
		stats.Hugetlb = append(stats.Hugetlb, s)
	}
	return nil
}

func (h *hugetlbController) readSizeStat(path, size string) (*v1.HugetlbStat, error) {
	s := v1.HugetlbStat{
		Pagesize: size,
	}
	for _, t := range []struct {
		name  string
		value *uint64
	}{
		{
			name:  "usage_in_bytes",
			value: &s.Usage,
		},
		{
			name:  "max_usage_in_bytes",
			value: &s.Max,
		},
		{
			name:  "failcnt",
			value: &s.Failcnt,
		},
	} {
		v, err := readUint(filepath.Join(h.Path(path), strings.Join([]string{"hugetlb", size, t.name}, ".")))
		if err != nil {
			return nil, err
		}
		*t.value = v
	}
	return &s, nil
}
