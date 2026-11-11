//go:build !windows

/*
Copyright The Kubernetes Authors.

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

package stats

// capUsageNanoCores is a no-op on non-Windows platforms. The Windows build
// overrides it to clamp CRI-reported usageNanoCores to the node's CPU capacity,
// preventing a Windows runtime measurement artifact from leaking into the
// summary API (see capWindowsUsageNanoCores).
func (p *criStatsProvider) capUsageNanoCores(usageNanoCores *uint64) *uint64 {
	return usageNanoCores
}
