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

package api

type Level string

const (
	LevelPrivileged Level = "privileged"
	LevelBaseline   Level = "baseline"
	LevelRestricted Level = "restricted"
)

var validLevels = []string{
	string(LevelPrivileged),
	string(LevelBaseline),
	string(LevelRestricted),
}

const VersionLatest = "latest"

const AuditAnnotationPrefix = labelPrefix

const (
	labelPrefix = "pod-security.kubernetes.io/"

	EnforceLevelLabel   = labelPrefix + "enforce"
	EnforceVersionLabel = labelPrefix + "enforce-version"
	AuditLevelLabel     = labelPrefix + "audit"
	AuditVersionLabel   = labelPrefix + "audit-version"
	WarnLevelLabel      = labelPrefix + "warn"
	WarnVersionLabel    = labelPrefix + "warn-version"

	ExemptionReasonAnnotationKey = "exempt"
	AuditViolationsAnnotationKey = "audit-violations"
	EnforcedPolicyAnnotationKey  = "enforce-policy"
)
