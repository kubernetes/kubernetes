//go:build !race

/*
Copyright 2025 The Kubernetes Authors.

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

package junit

import (
	"github.com/onsi/ginkgo/v2"
)

func detectDataRaces(report ginkgo.Report) {
	// This is a NOP variant of this function which is used when the test binary is compiled
	// without race detection. In that case there cannot be any data race reports and therefore
	// we don't need to check for them.
}
