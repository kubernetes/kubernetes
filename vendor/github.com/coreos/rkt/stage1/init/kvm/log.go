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

package kvm

import (
	"io/ioutil"
	"os"

	rktlog "github.com/coreos/rkt/pkg/log"
)

var diag *rktlog.Logger

func init() {
	diag = rktlog.New(os.Stderr, "kvm", false)
}

func InitDebug(debug bool) {
	diag.SetDebug(debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}
}
