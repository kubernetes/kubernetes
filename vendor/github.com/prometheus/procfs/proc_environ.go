// Copyright 2019 The Prometheus Authors
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
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Environ reads process environments from `/proc/<pid>/environ`.
func (p Proc) Environ() ([]string, error) {
	environments := make([]string, 0)

	data, err := util.ReadFileNoStat(p.path("environ"))
	if err != nil {
		return environments, err
	}

	environments = strings.Split(string(data), "\000")
	if len(environments) > 0 {
		environments = environments[:len(environments)-1]
	}

	return environments, nil
}
