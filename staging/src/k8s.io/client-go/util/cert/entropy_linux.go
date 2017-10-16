/*
Copyright 2017 The Kubernetes Authors.

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

package cert

import (
	"io/ioutil"
	"strconv"
	"strings"

	"fmt"
)

const EntropyGoodLevel = 100

// EntropyWarning checks for enough entropy on the machine for SSL operations and returns a warning string if yes.
func EntropyWarning() (string, error) {
	bs, err := ioutil.ReadFile("/proc/sys/kernel/random/entropy_avail")
	if err != nil {
		return "", err
	}

	s := strings.SplitN(string(bs), "\n", 2)[0]
	entropy, err := strconv.Atoi(s)
	if err != nil {
		return "", err
	}

	if entropy < EntropyGoodLevel {
		return fmt.Sprintf("Entropy of the system is below %d which is considered low: %d. SSL operations might take long.", EntropyGoodLevel, entropy), nil
	}

	return "", nil
}
