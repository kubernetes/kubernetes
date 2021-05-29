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

package auth

import (
	"github.com/prometheus/client_golang/prometheus"
	"sync"
)

var (
	currentAuthRevision = prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "etcd_debugging",
		Subsystem: "auth",
		Name:      "revision",
		Help:      "The current revision of auth store.",
	},
		func() float64 {
			reportCurrentAuthRevMu.RLock()
			defer reportCurrentAuthRevMu.RUnlock()
			return reportCurrentAuthRev()
		},
	)
	// overridden by auth store initialization
	reportCurrentAuthRevMu sync.RWMutex
	reportCurrentAuthRev   = func() float64 { return 0 }
)

func init() {
	prometheus.MustRegister(currentAuthRevision)
}
