/*
Copyright 2014 Google Inc. All rights reserved.

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

package clientcmd

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var ErrNoContext = errors.New("no context chosen")

type errContextNotFound struct {
	ContextName string
}

func (e *errContextNotFound) Error() string {
	return fmt.Sprintf("context was not found for specified context: %v", e.ContextName)
}

// IsContextNotFound returns a boolean indicating whether the error is known to
// report that a context was not found
func IsContextNotFound(err error) bool {
	if err == nil {
		return false
	}

	return strings.Contains(err.Error(), "context was not found for specified context")
}

// Validate checks for errors in the Config.  It does not return early so that it can find as many errors as possible.
func Validate(config Config) error {
	validationErrors := make([]error, 0)

	if len(config.CurrentContext) != 0 {
		if _, exists := config.Contexts[config.CurrentContext]; !exists {
			validationErrors = append(validationErrors, &errContextNotFound{config.CurrentContext})
		}
	}

	for contextName, context := range config.Contexts {
		validationErrors = append(validationErrors, validateContext(contextName, context, config)...)
	}

	for authInfoName, authInfo := range config.AuthInfos {
		validationErrors = append(validationErrors, validateAuthInfo(authInfoName, authInfo)...)
	}

	for clusterName, clusterInfo := range config.Clusters {
		validationErrors = append(validationErrors, validateClusterInfo(clusterName, clusterInfo)...)
	}

	return util.SliceToError(validationErrors)
}

// ConfirmUsable looks a particular context and determines if that particular part of the config is useable.  There might still be errors in the config,
// but no errors in the sections requested or referenced.  It does not return early so that it can find as many errors as possible.
func ConfirmUsable(config Config, passedContextName string) error {
	validationErrors := make([]error, 0)

	var contextName string
	if len(passedContextName) != 0 {
		contextName = passedContextName
	} else {
		contextName = config.CurrentContext
	}

	if len(contextName) == 0 {
		return ErrNoContext
	}

	context, exists := config.Contexts[contextName]
	if !exists {
		validationErrors = append(validationErrors, &errContextNotFound{contextName})
	}

	if exists {
		validationErrors = append(validationErrors, validateContext(contextName, context, config)...)
		validationErrors = append(validationErrors, validateAuthInfo(context.AuthInfo, config.AuthInfos[context.AuthInfo])...)
		validationErrors = append(validationErrors, validateClusterInfo(context.Cluster, config.Clusters[context.Cluster])...)
	}

	return util.SliceToError(validationErrors)
}

// validateClusterInfo looks for conflicts and errors in the cluster info
func validateClusterInfo(clusterName string, clusterInfo Cluster) []error {
	validationErrors := make([]error, 0)

	if len(clusterInfo.Server) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("no server found for %v", clusterName))
	}
	if len(clusterInfo.CertificateAuthority) != 0 {
		clientCertCA, err := os.Open(clusterInfo.CertificateAuthority)
		defer clientCertCA.Close()
		if err != nil {
			validationErrors = append(validationErrors, fmt.Errorf("unable to read certificate-authority %v for %v due to %v", clusterInfo.CertificateAuthority, clusterName, err))
		}
	}

	return validationErrors
}

// validateAuthInfo looks for conflicts and errors in the auth info
func validateAuthInfo(authInfoName string, authInfo AuthInfo) []error {
	validationErrors := make([]error, 0)

	methods := make([]string, 0, 3)
	if len(authInfo.Token) != 0 {
		methods = append(methods, "token")
	}
	if len(authInfo.AuthPath) != 0 {
		methods = append(methods, "authFile")

		file, err := os.Open(authInfo.AuthPath)
		os.IsNotExist(err)
		defer file.Close()
		if err != nil {
			validationErrors = append(validationErrors, fmt.Errorf("unable to read auth-path %v for %v due to %v", authInfo.AuthPath, authInfoName, err))
		}
	}
	if len(authInfo.ClientCertificate) != 0 {
		methods = append(methods, "clientCert")

		clientCertFile, err := os.Open(authInfo.ClientCertificate)
		defer clientCertFile.Close()
		if err != nil {
			validationErrors = append(validationErrors, fmt.Errorf("unable to read client-cert %v for %v due to %v", authInfo.ClientCertificate, authInfoName, err))
		}
		clientKeyFile, err := os.Open(authInfo.ClientKey)
		defer clientKeyFile.Close()
		if err != nil {
			validationErrors = append(validationErrors, fmt.Errorf("unable to read client-key %v for %v due to %v", authInfo.ClientKey, authInfoName, err))
		}
	}

	if (len(methods)) > 1 {
		validationErrors = append(validationErrors, fmt.Errorf("more than one authentication method found for  %v.  Found %v, only one is allowed", authInfoName, methods))
	}

	return validationErrors
}

// validateContext looks for errors in the context.  It is not transitive, so errors in the reference authInfo or cluster configs are not included in this return
func validateContext(contextName string, context Context, config Config) []error {
	validationErrors := make([]error, 0)

	if len(context.AuthInfo) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("user was not specified for Context %v", contextName))
	} else if _, exists := config.AuthInfos[context.AuthInfo]; !exists {
		validationErrors = append(validationErrors, fmt.Errorf("user, %v, was not found for Context %v", context.AuthInfo, contextName))
	}

	if len(context.Cluster) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("cluster was not specified for Context %v", contextName))
	} else if _, exists := config.Clusters[context.Cluster]; !exists {
		validationErrors = append(validationErrors, fmt.Errorf("cluster, %v, was not found for Context %v", context.Cluster, contextName))
	}

	if (len(context.Namespace) != 0) && !util.IsDNS952Label(context.Namespace) {
		validationErrors = append(validationErrors, fmt.Errorf("namespace, %v, for context %v, does not conform to the kubernetest DNS952 rules", context.Namespace, contextName))
	}

	return validationErrors
}
