/*
Copyright 2019 The Kubernetes Authors.

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

package pidlimit

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
)

const (
	// PIDs is the (internal) name for this resource
	PIDs v1.ResourceName = "pid"
)

func parseRunningTaskCount(loadavg string) (int64, error) {
	fields := strings.Fields(loadavg)
	if len(fields) < 5 {
		return 0, fmt.Errorf("not enough fields in /proc/loadavg")
	}
	subfields := strings.Split(fields[3], "/")
	if len(subfields) != 2 {
		return 0, fmt.Errorf("error parsing fourth field of /proc/loadavg")
	}
	return strconv.ParseInt(subfields[1], 10, 64)
}
