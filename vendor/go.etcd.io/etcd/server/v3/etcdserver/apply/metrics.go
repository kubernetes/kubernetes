// Copyright 2022 The etcd Authors
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

package apply

import "github.com/prometheus/client_golang/prometheus"

var alarms = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Namespace: "etcd_debugging",
		Subsystem: "server",
		Name:      "alarms",
		Help:      "Alarms for every member in cluster. 1 for 'server_id' label with current ID. 2 for 'alarm_type' label with type of this alarm",
	},
	[]string{"server_id", "alarm_type"},
)

func init() {
	prometheus.MustRegister(alarms)
}
