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

package webhook

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/pod-security-admission/admission"
)

// responsibilities:
// - serving webhook
// - decoding AdmissionReview
// - converting unstructured object/oldObject to typed object
//   - use a scheme that is pluggable (can default to client-go scheme, but allow custom resource runner to provide a scheme)
// - construct AdmissionReview response

type Config struct {
	TLSCertFile string
	TLSKeyFile  string
	Port        string
}

// Config.AddFlags
// transform Config to Server

type Server struct {
	// cert, key

	// TODO: scope down to decoder?
	// included so extenders can plug in their own unstructured->typed conversion
	Scheme runtime.Scheme

	// TODO: make this an interface so extenders can wrap/delegate
	Admission admission.Admission
}

// Server.Run()
