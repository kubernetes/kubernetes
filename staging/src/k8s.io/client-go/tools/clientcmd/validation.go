/*
Copyright 2014 The Kubernetes Authors.

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
	"reflect"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

var (
	ErrNoContext   = errors.New("no context chosen")
	ErrEmptyConfig = errors.New("no configuration has been provided")
	// message is for consistency with old behavior
	ErrEmptyCluster = errors.New("cluster has no server defined")
)

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
	if _, ok := err.(*errContextNotFound); ok || err == ErrNoContext {
		return true
	}
	return strings.Contains(err.Error(), "context was not found for specified context")
}

// IsEmptyConfig returns true if the provided error indicates the provided configuration
// is empty.
func IsEmptyConfig(err error) bool {
	switch t := err.(type) {
	case errConfigurationInvalid:
		return len(t) == 1 && t[0] == ErrEmptyConfig
	}
	return err == ErrEmptyConfig
}

// errConfigurationInvalid is a set of errors indicating the configuration is invalid.
type errConfigurationInvalid []error

// errConfigurationInvalid implements error and Aggregate
var _ error = errConfigurationInvalid{}
var _ utilerrors.Aggregate = errConfigurationInvalid{}

func newErrConfigurationInvalid(errs []error) error {
	switch len(errs) {
	case 0:
		return nil
	default:
		return errConfigurationInvalid(errs)
	}
}

// Error implements the error interface
func (e errConfigurationInvalid) Error() string {
	return fmt.Sprintf("invalid configuration: %v", utilerrors.NewAggregate(e).Error())
}

// Errors implements the AggregateError interface
func (e errConfigurationInvalid) Errors() []error {
	return e
}

// IsConfigurationInvalid returns true if the provided error indicates the configuration is invalid.
func IsConfigurationInvalid(err error) bool {
	switch err.(type) {
	case *errContextNotFound, errConfigurationInvalid:
		return true
	}
	return IsContextNotFound(err)
}

// Validate checks for errors in the Config.  It does not return early so that it can find as many errors as possible.
func Validate(config clientcmdapi.Config) error {
	validationErrors := make([]error, 0)

	if clientcmdapi.IsConfigEmpty(&config) {
		return newErrConfigurationInvalid([]error{ErrEmptyConfig})
	}

	if len(config.CurrentContext) != 0 {
		if _, exists := config.Contexts[config.CurrentContext]; !exists {
			validationErrors = append(validationErrors, &errContextNotFound{config.CurrentContext})
		}
	}

	for contextName, context := range config.Contexts {
		validationErrors = append(validationErrors, validateContext(contextName, *context, config)...)
	}

	for authInfoName, authInfo := range config.AuthInfos {
		validationErrors = append(validationErrors, validateAuthInfo(authInfoName, *authInfo)...)
	}

	for clusterName, clusterInfo := range config.Clusters {
		validationErrors = append(validationErrors, validateClusterInfo(clusterName, *clusterInfo)...)
	}

	return newErrConfigurationInvalid(validationErrors)
}

// ConfirmUsable looks a particular context and determines if that particular part of the config is useable.  There might still be errors in the config,
// but no errors in the sections requested or referenced.  It does not return early so that it can find as many errors as possible.
func ConfirmUsable(config clientcmdapi.Config, passedContextName string) error {
	validationErrors := make([]error, 0)

	if clientcmdapi.IsConfigEmpty(&config) {
		return newErrConfigurationInvalid([]error{ErrEmptyConfig})
	}

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
		validationErrors = append(validationErrors, validateContext(contextName, *context, config)...)
		validationErrors = append(validationErrors, validateAuthInfo(context.AuthInfo, *config.AuthInfos[context.AuthInfo])...)
		validationErrors = append(validationErrors, validateClusterInfo(context.Cluster, *config.Clusters[context.Cluster])...)
	}

	return newErrConfigurationInvalid(validationErrors)
}

// validateClusterInfo looks for conflicts and errors in the cluster info
func validateClusterInfo(clusterName string, clusterInfo clientcmdapi.Cluster) []error {
	validationErrors := make([]error, 0)

	emptyCluster := clientcmdapi.NewCluster()
	if reflect.DeepEqual(*emptyCluster, clusterInfo) {
		return []error{ErrEmptyCluster}
	}

	if len(clusterInfo.Server) == 0 {
		if len(clusterName) == 0 {
			validationErrors = append(validationErrors, fmt.Errorf("default cluster has no server defined"))
		} else {
			validationErrors = append(validationErrors, fmt.Errorf("no server found for cluster %q", clusterName))
		}
	}
	// Make sure CA data and CA file aren't both specified
	if len(clusterInfo.CertificateAuthority) != 0 && len(clusterInfo.CertificateAuthorityData) != 0 {
		validationErrors = append(validationErrors, fmt.Errorf("certificate-authority-data and certificate-authority are both specified for %v. certificate-authority-data will override.", clusterName))
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
func validateAuthInfo(authInfoName string, authInfo clientcmdapi.AuthInfo) []error {
	validationErrors := make([]error, 0)

	usingAuthPath := false
	methods := make([]string, 0, 3)
	if len(authInfo.Token) != 0 {
		methods = append(methods, "token")
	}
	if len(authInfo.Username) != 0 || len(authInfo.Password) != 0 {
		methods = append(methods, "basicAuth")
	}

	if len(authInfo.ClientCertificate) != 0 || len(authInfo.ClientCertificateData) != 0 {
		// Make sure cert data and file aren't both specified
		if len(authInfo.ClientCertificate) != 0 && len(authInfo.ClientCertificateData) != 0 {
			validationErrors = append(validationErrors, fmt.Errorf("client-cert-data and client-cert are both specified for %v. client-cert-data will override.", authInfoName))
		}
		// Make sure key data and file aren't both specified
		if len(authInfo.ClientKey) != 0 && len(authInfo.ClientKeyData) != 0 {
			validationErrors = append(validationErrors, fmt.Errorf("client-key-data and client-key are both specified for %v; client-key-data will override", authInfoName))
		}
		// Make sure a key is specified
		if len(authInfo.ClientKey) == 0 && len(authInfo.ClientKeyData) == 0 {
			validationErrors = append(validationErrors, fmt.Errorf("client-key-data or client-key must be specified for %v to use the clientCert authentication method.", authInfoName))
		}

		if len(authInfo.ClientCertificate) != 0 {
			clientCertFile, err := os.Open(authInfo.ClientCertificate)
			defer clientCertFile.Close()
			if err != nil {
				validationErrors = append(validationErrors, fmt.Errorf("unable to read client-cert %v for %v due to %v", authInfo.ClientCertificate, authInfoName, err))
			}
		}
		if len(authInfo.ClientKey) != 0 {
			clientKeyFile, err := os.Open(authInfo.ClientKey)
			defer clientKeyFile.Close()
			if err != nil {
				validationErrors = append(validationErrors, fmt.Errorf("unable to read client-key %v for %v due to %v", authInfo.ClientKey, authInfoName, err))
			}
		}
	}

	// authPath also provides information for the client to identify the server, so allow multiple auth methods in that case
	if (len(methods) > 1) && (!usingAuthPath) {
		validationErrors = append(validationErrors, fmt.Errorf("more than one authentication method found for %v; found %v, only one is allowed", authInfoName, methods))
	}

	// ImpersonateGroups or ImpersonateUserExtra should be requested with a user
	if (len(authInfo.ImpersonateGroups) > 0 || len(authInfo.ImpersonateUserExtra) > 0) && (len(authInfo.Impersonate) == 0) {
		validationErrors = append(validationErrors, fmt.Errorf("requesting groups or user-extra for %v without impersonating a user", authInfoName))
	}
	return validationErrors
}

// validateContext looks for errors in the context.  It is not transitive, so errors in the reference authInfo or cluster configs are not included in this return
func validateContext(contextName string, context clientcmdapi.Context, config clientcmdapi.Config) []error {
	validationErrors := make([]error, 0)

	if len(context.AuthInfo) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("user was not specified for context %q", contextName))
	} else if _, exists := config.AuthInfos[context.AuthInfo]; !exists {
		validationErrors = append(validationErrors, fmt.Errorf("user %q was not found for context %q", context.AuthInfo, contextName))
	}

	if len(context.Cluster) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("cluster was not specified for context %q", contextName))
	} else if _, exists := config.Clusters[context.Cluster]; !exists {
		validationErrors = append(validationErrors, fmt.Errorf("cluster %q was not found for context %q", context.Cluster, contextName))
	}

	if len(context.Namespace) != 0 {
		if len(validation.IsDNS1123Label(context.Namespace)) != 0 {
			validationErrors = append(validationErrors, fmt.Errorf("namespace %q for context %q does not conform to the kubernetes DNS_LABEL rules", context.Namespace, contextName))
		}
	}

	return validationErrors
}
