/*
Copyright 2020 The Kubernetes Authors.

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

// Package exec contains helper utilities for exec credential plugins.
package exec

import (
	"errors"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/pkg/apis/clientauthentication/install"
	"k8s.io/client-go/rest"
)

const execInfoEnv = "KUBERNETES_EXEC_INFO"

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

func init() {
	install.Install(scheme)
}

// LoadExecCredentialFromEnv is a helper-wrapper around LoadExecCredential that loads from the
// well-known KUBERNETES_EXEC_INFO environment variable.
//
// When the KUBERNETES_EXEC_INFO environment variable is not set or is empty, then this function
// will immediately return an error.
func LoadExecCredentialFromEnv() (runtime.Object, *rest.Config, error) {
	env := os.Getenv(execInfoEnv)
	if env == "" {
		return nil, nil, errors.New("KUBERNETES_EXEC_INFO env var is unset or empty")
	}
	return LoadExecCredential([]byte(env))
}

// LoadExecCredential loads the configuration needed for an exec plugin to communicate with a
// cluster.
//
// LoadExecCredential expects the provided data to be a serialized client.authentication.k8s.io
// ExecCredential object (of any version). If the provided data is invalid (i.e., it cannot be
// unmarshalled into any known client.authentication.k8s.io ExecCredential version), an error will
// be returned. A successfully unmarshalled ExecCredential will be returned as the first return
// value.
//
// If the provided data is successfully unmarshalled, but it does not contain cluster information
// (i.e., ExecCredential.Spec.Cluster == nil), then an error will be returned.
//
// Note that the returned rest.Config will use anonymous authentication, since the exec plugin has
// not returned credentials for this cluster yet.
func LoadExecCredential(data []byte) (runtime.Object, *rest.Config, error) {
	obj, gvk, err := codecs.UniversalDeserializer().Decode(data, nil, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("decode: %w", err)
	}

	expectedGK := schema.GroupKind{
		Group: clientauthentication.SchemeGroupVersion.Group,
		Kind:  "ExecCredential",
	}
	if gvk.GroupKind() != expectedGK {
		return nil, nil, fmt.Errorf(
			"invalid group/kind: wanted %s, got %s",
			expectedGK.String(),
			gvk.GroupKind().String(),
		)
	}

	// Explicitly convert object here so that we can return a nicer error message above for when the
	// data represents an invalid type.
	var execCredential clientauthentication.ExecCredential
	if err := scheme.Convert(obj, &execCredential, nil); err != nil {
		return nil, nil, fmt.Errorf("cannot convert to ExecCredential: %w", err)
	}

	if execCredential.Spec.Cluster == nil {
		return nil, nil, errors.New("ExecCredential does not contain cluster information")
	}

	restConfig, err := rest.ExecClusterToConfig(execCredential.Spec.Cluster)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot create rest.Config: %w", err)
	}

	return obj, restConfig, nil
}
