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

package common

import (
	"os"
	"strconv"
	"strings"
)

// a string set of known rkt experiments, gated stage0 features
var stage0Experiments = map[string]struct{}{
	"app":    {}, // rkt app subcommands for CRI
	"attach": {}, // rkt attach subcommands and streaming options
}

// IsExperimentEnabled returns true if the given rkt experiment is enabled.
// The given name is converted to upper case and a bool RKT_EXPERIMENT_{NAME}
// environment variable is retrieved.
// If the experiment name is unknown, false is returned.
// If the environment variable does not contain a valid bool value
// according to strconv.ParseBool, false is returned.
func IsExperimentEnabled(name string) bool {
	if _, ok := stage0Experiments[name]; !ok {
		return false
	}

	v := os.Getenv("RKT_EXPERIMENT_" + strings.ToUpper(name))

	enabled, err := strconv.ParseBool(v)
	if err != nil {
		return false // ignore errors from bool conversion
	}

	return enabled
}
