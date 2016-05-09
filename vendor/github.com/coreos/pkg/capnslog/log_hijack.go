// Copyright 2015 CoreOS, Inc.
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

package capnslog

import (
	"log"
)

func init() {
	pkg := NewPackageLogger("log", "")
	w := packageWriter{pkg}
	log.SetFlags(0)
	log.SetPrefix("")
	log.SetOutput(w)
}

type packageWriter struct {
	pl *PackageLogger
}

func (p packageWriter) Write(b []byte) (int, error) {
	if p.pl.level < INFO {
		return 0, nil
	}
	p.pl.internalLog(calldepth+2, INFO, string(b))
	return len(b), nil
}
