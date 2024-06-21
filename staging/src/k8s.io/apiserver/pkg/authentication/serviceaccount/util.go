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

package serviceaccount

import (
	"context"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"

	"k8s.io/klog/v2"
)

const (
	ServiceAccountUsernamePrefix    = "system:serviceaccount:"
	ServiceAccountUsernameSeparator = ":"
	ServiceAccountGroupPrefix       = "system:serviceaccounts:"
	AllServiceAccountsGroup         = "system:serviceaccounts"
	// IssuedCredentialIDAuditAnnotationKey is the annotation key used in the audit event that is persisted to the
	// '/token' endpoint for service accounts.
	// This annotation indicates the generated credential identifier for the service account token being issued.
	// This is useful when tracing back the origin of tokens that have gone on to make request that have persisted
	// their credential-identifier into the audit log via the user's extra info stored on subsequent audit events.
	IssuedCredentialIDAuditAnnotationKey = "authentication.kubernetes.io/issued-credential-id"
	// PodNameKey is the key used in a user's "extra" to specify the pod name of
	// the authenticating request.
	PodNameKey = "authentication.kubernetes.io/pod-name"
	// PodUIDKey is the key used in a user's "extra" to specify the pod UID of
	// the authenticating request.
	PodUIDKey = "authentication.kubernetes.io/pod-uid"
	// NodeNameKey is the key used in a user's "extra" to specify the node name of
	// the authenticating request.
	NodeNameKey = "authentication.kubernetes.io/node-name"
	// NodeUIDKey is the key used in a user's "extra" to specify the node UID of
	// the authenticating request.
	NodeUIDKey = "authentication.kubernetes.io/node-uid"
)

// MakeUsername generates a username from the given namespace and ServiceAccount name.
// The resulting username can be passed to SplitUsername to extract the original namespace and ServiceAccount name.
func MakeUsername(namespace, name string) string {
	return ServiceAccountUsernamePrefix + namespace + ServiceAccountUsernameSeparator + name
}

// MatchesUsername checks whether the provided username matches the namespace and name without
// allocating. Use this when checking a service account namespace and name against a known string.
func MatchesUsername(namespace, name string, username string) bool {
	if !strings.HasPrefix(username, ServiceAccountUsernamePrefix) {
		return false
	}
	username = username[len(ServiceAccountUsernamePrefix):]

	if !strings.HasPrefix(username, namespace) {
		return false
	}
	username = username[len(namespace):]

	if !strings.HasPrefix(username, ServiceAccountUsernameSeparator) {
		return false
	}
	username = username[len(ServiceAccountUsernameSeparator):]

	return username == name
}

var invalidUsernameErr = fmt.Errorf("Username must be in the form %s", MakeUsername("namespace", "name"))

// SplitUsername returns the namespace and ServiceAccount name embedded in the given username,
// or an error if the username is not a valid name produced by MakeUsername
func SplitUsername(username string) (string, string, error) {
	if !strings.HasPrefix(username, ServiceAccountUsernamePrefix) {
		return "", "", invalidUsernameErr
	}
	trimmed := strings.TrimPrefix(username, ServiceAccountUsernamePrefix)
	parts := strings.Split(trimmed, ServiceAccountUsernameSeparator)
	if len(parts) != 2 {
		return "", "", invalidUsernameErr
	}
	namespace, name := parts[0], parts[1]
	if len(apimachineryvalidation.ValidateNamespaceName(namespace, false)) != 0 {
		return "", "", invalidUsernameErr
	}
	if len(apimachineryvalidation.ValidateServiceAccountName(name, false)) != 0 {
		return "", "", invalidUsernameErr
	}
	return namespace, name, nil
}

// MakeGroupNames generates service account group names for the given namespace
func MakeGroupNames(namespace string) []string {
	return []string{
		AllServiceAccountsGroup,
		MakeNamespaceGroupName(namespace),
	}
}

// MakeNamespaceGroupName returns the name of the group all service accounts in the namespace are included in
func MakeNamespaceGroupName(namespace string) string {
	return ServiceAccountGroupPrefix + namespace
}

// UserInfo returns a user.Info interface for the given namespace, service account name and UID
func UserInfo(namespace, name, uid string) user.Info {
	return (&ServiceAccountInfo{
		Name:      name,
		Namespace: namespace,
		UID:       uid,
	}).UserInfo()
}

type ServiceAccountInfo struct {
	Name, Namespace, UID string
	PodName, PodUID      string
	CredentialID         string
	NodeName, NodeUID    string
}

func (sa *ServiceAccountInfo) UserInfo() user.Info {
	info := &user.DefaultInfo{
		Name:   MakeUsername(sa.Namespace, sa.Name),
		UID:    sa.UID,
		Groups: MakeGroupNames(sa.Namespace),
	}

	if sa.PodName != "" && sa.PodUID != "" {
		if info.Extra == nil {
			info.Extra = make(map[string][]string)
		}
		info.Extra[PodNameKey] = []string{sa.PodName}
		info.Extra[PodUIDKey] = []string{sa.PodUID}
	}
	if sa.CredentialID != "" {
		if info.Extra == nil {
			info.Extra = make(map[string][]string)
		}
		info.Extra[user.CredentialIDKey] = []string{sa.CredentialID}
	}
	if sa.NodeName != "" {
		if info.Extra == nil {
			info.Extra = make(map[string][]string)
		}
		info.Extra[NodeNameKey] = []string{sa.NodeName}
		// node UID is optional and will only be set if the node name is set
		if sa.NodeUID != "" {
			info.Extra[NodeUIDKey] = []string{sa.NodeUID}
		}
	}

	return info
}

// CredentialIDForJTI converts a given JTI string into a credential identifier for use in a
// users 'extra' info.
func CredentialIDForJTI(jti string) string {
	if len(jti) == 0 {
		return ""
	}
	return "JTI=" + jti
}

// IsServiceAccountToken returns true if the secret is a valid api token for the service account
func IsServiceAccountToken(secret *v1.Secret, sa *v1.ServiceAccount) bool {
	if secret.Type != v1.SecretTypeServiceAccountToken {
		return false
	}

	name := secret.Annotations[v1.ServiceAccountNameKey]
	uid := secret.Annotations[v1.ServiceAccountUIDKey]
	if name != sa.Name {
		// Name must match
		return false
	}
	if len(uid) > 0 && uid != string(sa.UID) {
		// If UID is specified, it must match
		return false
	}

	return true
}

func GetOrCreateServiceAccount(coreClient v1core.CoreV1Interface, namespace, name string) (*v1.ServiceAccount, error) {
	sa, err := coreClient.ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err == nil {
		return sa, nil
	}
	if !apierrors.IsNotFound(err) {
		return nil, err
	}

	// Create the namespace if we can't verify it exists.
	// Tolerate errors, since we don't know whether this component has namespace creation permissions.
	if _, err := coreClient.Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{}); apierrors.IsNotFound(err) {
		if _, err = coreClient.Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			klog.Warningf("create non-exist namespace %s failed:%v", namespace, err)
		}
	}

	// Create the service account
	sa, err = coreClient.ServiceAccounts(namespace).Create(context.TODO(), &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}}, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		// If we're racing to init and someone else already created it, re-fetch
		return coreClient.ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	}
	return sa, err
}
