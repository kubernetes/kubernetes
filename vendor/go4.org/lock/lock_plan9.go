/*
Copyright 2013 The Go Authors

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

package lock

import (
	"fmt"
	"io"
	"os"
)

func init() {
	lockFn = lockPlan9
}

func lockPlan9(name string) (io.Closer, error) {
	fi, err := os.Stat(name)
	if err == nil && fi.Size() > 0 {
		return nil, fmt.Errorf("can't Lock file %q: has non-zero size", name)
	}

	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE, os.ModeExclusive|0644)
	if err != nil {
		return nil, fmt.Errorf("Lock Create of %s failed: %v", name, err)
	}

	return &unlocker{f: f, abs: name}, nil
}
