/*
Copyright 2018 The Kubernetes Authors.

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

// Package podtolerationrestriction is a plugin that first verifies
// any conflict between a pod's tolerations and its namespace's
// tolerations, and rejects the pod if there's a conflict.  If there's
// no conflict, the pod's tolerations are merged with its namespace's
// toleration. Resulting pod's tolerations are verified against its
// namespace's whitelist of tolerations. If the verification is
// successful, the pod is admitted otherwise rejected. If a namespace
// does not have associated default or whitelist of tolerations, then
// cluster level default or whitelist of tolerations are used instead
// if specified. Tolerations to a namespace are assigned via
// scheduler.alpha.kubernetes.io/defaultTolerations and
// scheduler.alpha.kubernetes.io/tolerationsWhitelist annotations
// keys.
package podtolerationrestriction
