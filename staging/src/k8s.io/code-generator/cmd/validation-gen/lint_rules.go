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

package main

var ruleOptionalAndRequired = conflictingTagsRule(
	"fields cannot be both optional and required",
	"+k8s:optional", "+k8s:required")

var ruleRequiredAndDefault = conflictingTagsRule(
	"fields with default values must be optional",
	"+k8s:required", "+default")

var ruleUnionMemberAndOptional = dependentTagsRule(
	"fields which are union members must be optional",
	"+k8s:unionMember", "+k8s:optional")

var defaultLintRules = []lintRule{
	ruleOptionalAndRequired,
	ruleRequiredAndDefault,
	ruleUnionMemberAndOptional,
}
