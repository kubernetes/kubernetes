// Copyright 2015 Google Inc. All Rights Reserved.
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

package gcm

type rateMetric struct {
	name        string
	description string
}

// Map of base metrics to corresponding rateMetrics.
var gcmRateMetrics = map[string]rateMetric{
	"uptime": {
		name:        "uptime_rate",
		description: "Rate of change of time since start in seconds per second",
	},
	"cpu/usage": {
		name:        "cpu/usage_rate",
		description: "Rate of total CPU usage in millicores per second",
	},
	"memory/page_faults": {
		name:        "memory/page_faults_rate",
		description: "Rate of major page faults in counts per second",
	},
	"network/rx": {
		name:        "network/rx_rate",
		description: "Rate of bytes received over the network in bytes per second",
	},
	"network/rx_errors": {
		name:        "network/rx_errors_rate",
		description: "Rate of errors sending over the network in errors per second",
	},
	"network/tx": {
		name:        "network/tx_rate",
		description: "Rate of bytes transmitted over the network in bytes per second",
	},
	"network/tx_errors": {
		name:        "network/tx_errors_rate",
		description: "Rate of errors transmitting over the network in errors per second",
	},
}
