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

package healthz

import (
	"net/http"

	"k8s.io/apiserver/pkg/server/healthz"
)

// NamedPingChecker returns a health check with given name
// that returns no error when checked.
func NamedPingChecker(name string) healthz.HealthChecker {
	return NamedHealthChecker(name, healthz.PingHealthz)
}

// NamedHealthChecker creates a named health check from
// an unnamed one.
func NamedHealthChecker(name string, check UnnamedHealthChecker) healthz.HealthChecker {
	return healthz.NamedCheck(name, check.Check)
}

// UnnamedHealthChecker is an unnamed healthz checker.
// The name of the check can be set by the controller manager.
type UnnamedHealthChecker interface {
	Check(req *http.Request) error
}

var _ UnnamedHealthChecker = (healthz.HealthChecker)(nil)
