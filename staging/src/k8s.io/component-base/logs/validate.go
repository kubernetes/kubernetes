/*
Copyright 2021 The Kubernetes Authors.

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

package logs

import (
	"flag"
	"fmt"
	"strings"

	"github.com/spf13/pflag"
)

func ValidateLoggingConfiguration(o *Options) []error {
	errs := []error{}
	if o.LogFormat != DefaultLogFormat {
		allFlags := UnsupportedLoggingFlags(hyphensToUnderscores)
		for _, fname := range allFlags {
			if flagIsSet(fname, hyphensToUnderscores) {
				errs = append(errs, fmt.Errorf("non-default logging format doesn't honor flag: %s", fname))
			}
		}
	}
	if _, err := o.Get(); err != nil {
		errs = append(errs, fmt.Errorf("unsupported log format: %s", o.LogFormat))
	}
	return errs
}

// hyphensToUnderscores replaces hyphens with underscores
// we should always use underscores instead of hyphens when validate flags
func hyphensToUnderscores(s string) string {
	return strings.Replace(s, "-", "_", -1)
}

func flagIsSet(name string, normalizeFunc func(name string) string) bool {
	f := flag.Lookup(name)
	if f != nil {
		return f.DefValue != f.Value.String()
	}
	if normalizeFunc != nil {
		f = flag.Lookup(normalizeFunc(name))
		if f != nil {
			return f.DefValue != f.Value.String()
		}
	}
	pf := pflag.Lookup(name)
	if pf != nil {
		return pf.DefValue != pf.Value.String()
	}
	panic("failed to lookup unsupported log flag")
}
