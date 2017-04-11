// Copyright 2016 The etcd Authors
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

package clientv3_test

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
)

// TestMain sets up an etcd cluster if running the examples.
func TestMain(m *testing.M) {
	useCluster := true // default to running all tests
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.run=") {
			exp := strings.Split(arg, "=")[1]
			match, err := regexp.MatchString(exp, "Example")
			useCluster = (err == nil && match) || strings.Contains(exp, "Example")
			break
		}
	}

	v := 0
	if useCluster {
		cfg := integration.ClusterConfig{Size: 3}
		clus := integration.NewClusterV3(nil, &cfg)
		endpoints = make([]string, 3)
		for i := range endpoints {
			endpoints[i] = clus.Client(i).Endpoints()[0]
		}
		v = m.Run()
		clus.Terminate(nil)
	} else {
		v = m.Run()
	}

	if v == 0 && testutil.CheckLeakedGoroutine() {
		os.Exit(1)
	}
	os.Exit(v)
}
