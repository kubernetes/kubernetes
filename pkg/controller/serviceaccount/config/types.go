/*
Copyright 2019 The Kubernetes Authors.

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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SAControllerConfiguration contains elements describing ServiceAccountController.
type SAControllerConfiguration struct {
	// serviceAccountKeyFile is the filename containing a PEM-encoded private RSA key
	// used to sign service account tokens.
	ServiceAccountKeyFile string
	// concurrentSATokenSyncs is the number of service account token syncing operations
	// that will be done concurrently.
	ConcurrentSATokenSyncs int32
	// rootCAFile is the root certificate authority will be included in service
	// account's token secret. This must be a valid PEM-encoded CA bundle.
	RootCAFile string
}

type LegacySATokenCleanerConfiguration struct {
	// CleanUpPeriod is the period of time since the last usage of an
	// auto-generated service account token before it can be deleted.
	CleanUpPeriod metav1.Duration
}
