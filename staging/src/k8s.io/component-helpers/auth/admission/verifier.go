/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/coreos/go-oidc/v3/oidc"
	jose "github.com/go-jose/go-jose/v4"

	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/klog/v2"
)

var (
	// Keep in sync with allowedSigningAlgs in k8s.io/apiserver's OIDC JWT authenticator;
	// component-helpers cannot import k8s.io/apiserver.
	// https://github.com/kubernetes/kubernetes/blob/0bf39fd17169a6c933a6cb403c88656cd020aabf/staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/oidc.go#L261-L274
	allowedSigningAlgs = []jose.SignatureAlgorithm{
		jose.RS256,
		jose.RS384,
		jose.RS512,
		jose.ES256,
		jose.ES384,
		jose.ES512,
		jose.PS256,
		jose.PS384,
		jose.PS512,
		jose.MLDSA44,
		jose.MLDSA65,
		jose.MLDSA87,
	}
	supportedSigningAlgs = func() []string {
		algs := make([]string, len(allowedSigningAlgs))
		for i, alg := range allowedSigningAlgs {
			algs[i] = string(alg)
		}
		return algs
	}()
)

type webhookPrivateClaims struct {
	Kubernetes struct {
		Attestations map[string][]string `json:"attestations"`
	} `json:"kubernetes.io"`
}

// tokenGroups checks the Kubernetes claims of an authenticated token.
func tokenGroups(token *oidc.IDToken) ([]string, error) {
	var claims webhookPrivateClaims
	if err := token.Claims(&claims); err != nil {
		return nil, fmt.Errorf("decoding Kubernetes token claims: %w", err)
	}
	groups := claims.Kubernetes.Attestations[authenticationv1.AttestationAdmissionReviewAPIGroups]
	if len(groups) == 0 {
		return nil, fmt.Errorf("token must carry at least one %q attestation", authenticationv1.AttestationAdmissionReviewAPIGroups)
	}
	return groups, nil
}

// allowsAPIGroup checks the token's attestation against the admitted API group.
func allowsAPIGroup(groups []string, group string) bool {
	return slices.Contains(groups, group) || slices.Contains(groups, "*")
}

// verifier is unexported so callers must enter through [WithTokenVerification],
// which authenticates the Authorization header before reading or decoding the
// HTTP request body.
type verifier struct {
	verifier                      *oidc.IDTokenVerifier
	expectedSubresourceMismatches ExpectedSubresourceGroupMismatchMap
}

// SubresourceGroupMismatchKey identifies a subresource whose admitted object
// Kind may belong to a different API group than the parent resource attested by
// the webhook token. A zero Resource matches the named subresource on any
// parent resource (for example, "/scale").
type SubresourceGroupMismatchKey struct {
	Resource    metav1.GroupVersionResource
	Subresource string
}

// ExpectedSubresourceGroupMismatchMap maps a parent resource and subresource to
// the set of expected (and therefore allowed) cross-group object Kinds for that
// subresource.
type ExpectedSubresourceGroupMismatchMap map[SubresourceGroupMismatchKey]map[schema.GroupVersionKind]struct{}

// WebhookSubresourceGroupAllowedMismatch builds the map of expected cross-group
// subresource object Kinds for a webhook.
type WebhookSubresourceGroupAllowedMismatch interface {
	Build(ctx context.Context) (ExpectedSubresourceGroupMismatchMap, error)
}

func (v *verifier) verify(ctx context.Context, req *admissionv1.AdmissionRequest, token string) error {
	if req == nil {
		return errVerificationFailed
	}
	groups, err := v.authenticateToken(ctx, token)
	if err != nil {
		return err
	}
	return v.checkDecodedRequest(ctx, req, groups)
}

func (v *verifier) authenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	if ctx.Err() != nil {
		return nil, errVerificationFailed
	}
	logger := klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier")
	token, err := v.verifier.Verify(ctx, rawToken)
	if err != nil {
		logger.V(2).Info("Webhook token verification failed", "reason", "token authentication failed", "detail", err.Error())
		return nil, errVerificationFailed
	}
	groups, err := tokenGroups(token)
	if err != nil {
		logger.V(2).Info("Webhook token verification failed", "reason", "invalid Kubernetes token claims", "detail", err.Error())
		return nil, errVerificationFailed
	}
	return groups, nil
}

func (v *verifier) checkDecodedRequest(ctx context.Context, req *admissionv1.AdmissionRequest, groups []string) error {
	// Reject requests whose caller context was cancelled while reading or decoding the body.
	if ctx.Err() != nil || req == nil {
		return errVerificationFailed
	}
	logger := klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier")
	if !allowsAPIGroup(groups, req.Resource.Group) {
		logger.V(2).Info("Webhook token verification failed", "reason", "token admissionReviewAPIGroups does not authorize the review's API group")
		return errVerificationFailed
	}
	if err := checkRequestObjectKinds(req, v.expectedSubresourceMismatches); err != nil {
		logger.V(2).Info("Admission request rejected", "reason", "invalid admission request",
			"requestUID", req.UID, "operation", req.Operation, "resource", schema.GroupVersionResource(req.Resource).String(),
			"detail", err.Error())
		return errVerificationFailed
	}
	return nil
}

func checkRequestObjectKinds(req *admissionv1.AdmissionRequest, expectedMismatches ExpectedSubresourceGroupMismatchMap) error {
	if req == nil {
		return errors.New("admission request is nil")
	}
	kindOrResourceIncomplete := req.Kind.Version == "" || req.Kind.Kind == "" || req.Resource.Version == "" || req.Resource.Resource == ""
	if kindOrResourceIncomplete {
		return errors.New("admission request kind or resource is incomplete")
	}
	if req.Kind.Group != req.Resource.Group && !allowedKindResourceGroupMismatch(req, expectedMismatches) {
		return fmt.Errorf("object kind group %q differs from authorized resource group %q", req.Kind.Group, req.Resource.Group)
	}
	admittedKind := schema.GroupVersionKind(req.Kind)
	if err := checkObjectMatchesAdmittedKind("object", req.Object, admittedKind); err != nil {
		return err
	}
	return checkObjectMatchesAdmittedKind("oldObject", req.OldObject, admittedKind)
}

// allowedKindResourceGroupMismatch reports whether req is a known subresource
// whose admitted object Kind is in a different API group than the parent
// resource attested by the token.
func allowedKindResourceGroupMismatch(req *admissionv1.AdmissionRequest, expectedMismatches ExpectedSubresourceGroupMismatchMap) bool {
	if req == nil || req.SubResource == "" {
		return false
	}
	if expectedMismatches == nil {
		expectedMismatches = defaultSubresourceGroupMismatches
	}
	admittedKind := schema.GroupVersionKind(req.Kind)
	if kinds, ok := expectedMismatches[SubresourceGroupMismatchKey{Resource: req.Resource, Subresource: req.SubResource}]; ok {
		if _, allowed := kinds[admittedKind]; allowed {
			return true
		}
	}
	if kinds, ok := expectedMismatches[SubresourceGroupMismatchKey{Subresource: req.SubResource}]; ok {
		if _, allowed := kinds[admittedKind]; allowed {
			return true
		}
	}
	return false
}

// AllowedSubresourceGroupMismatchList returns a copy of the built-in Kubernetes
// subresources whose admitted object Kind belongs to a different API group than
// the parent resource attested by the webhook token ("/scale", "pods/eviction",
// and "serviceaccounts/token"). Webhooks implementing
// [WebhookSubresourceGroupAllowedMismatch] can call this function to seed their
// map before adding custom cross-group subresources.
func AllowedSubresourceGroupMismatchList() ExpectedSubresourceGroupMismatchMap {
	return ExpectedSubresourceGroupMismatchMap{
		{Subresource: "scale"}: {
			autoscalingSchemeGVKScale: {},
		},
		{Resource: coreV1PodsGVR, Subresource: "eviction"}: {
			policySchemeGVKEviction: {},
		},
		{Resource: coreV1ServiceAccountsGVR, Subresource: "token"}: {
			authenticationV1TokenRequestGVK: {},
		},
	}
}

// DiscoverSubresourceGroupMismatches queries cluster discovery via a
// [discovery.CachedDiscoveryInterface] and returns the built-in
// [AllowedSubresourceGroupMismatchList] extended with any discovered
// subresources whose response Kind group differs from their parent resource
// group.
func DiscoverSubresourceGroupMismatches(ctx context.Context, client discovery.CachedDiscoveryInterface) (ExpectedSubresourceGroupMismatchMap, error) {
	if client == nil {
		return nil, errors.New("admission: cached discovery client is nil")
	}
	_, resources, err := discovery.ServerGroupsAndResourcesWithContext(ctx, discovery.ToDiscoveryInterfaceWithContext(client))
	if err != nil {
		return nil, fmt.Errorf("admission: discovering subresource group mismatches: %w", err)
	}
	expectedMismatches := AllowedSubresourceGroupMismatchList()
	for _, list := range resources {
		if list == nil {
			continue
		}
		gv, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			return nil, fmt.Errorf("admission: parsing discovered groupVersion %q: %w", list.GroupVersion, err)
		}
		for _, resource := range list.APIResources {
			parent, subresource, isSubresource := strings.Cut(resource.Name, "/")
			if !isSubresource || parent == "" || subresource == "" || resource.Kind == "" {
				continue
			}
			kind := schema.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: resource.Kind}
			if resource.Group != "" {
				kind.Group = resource.Group
			}
			if resource.Version != "" {
				kind.Version = resource.Version
			}
			if kind.Group == gv.Group {
				continue
			}
			gvr := metav1.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: parent}
			if allowedKindResourceGroupMismatch(&admissionv1.AdmissionRequest{
				Kind:        metav1.GroupVersionKind(kind),
				Resource:    gvr,
				SubResource: subresource,
			}, expectedMismatches) {
				continue
			}
			key := SubresourceGroupMismatchKey{Resource: gvr, Subresource: subresource}
			if expectedMismatches[key] == nil {
				expectedMismatches[key] = make(map[schema.GroupVersionKind]struct{}, 1)
			}
			expectedMismatches[key][kind] = struct{}{}
		}
	}
	return expectedMismatches, nil
}

var (
	autoscalingSchemeGVKScale       = schema.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}
	policySchemeGVKEviction         = schema.GroupVersionKind{Group: "policy", Version: "v1", Kind: "Eviction"}
	authenticationV1TokenRequestGVK = authenticationv1.SchemeGroupVersion.WithKind("TokenRequest")
	coreV1PodsGVR                   = metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	coreV1ServiceAccountsGVR        = metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"}

	defaultSubresourceGroupMismatches = AllowedSubresourceGroupMismatchList()
)

// checkObjectMatchesAdmittedKind validates both RawExtension representations against the admitted GVK.
func checkObjectMatchesAdmittedKind(name string, extension runtime.RawExtension, admittedKind schema.GroupVersionKind) error {
	if err := checkRawObjectMatchesAdmittedKind(name, extension.Raw, admittedKind); err != nil {
		return err
	}
	return checkRuntimeObjectMatchesAdmittedKind(name, extension.Object, admittedKind)
}

// checkRuntimeObjectMatchesAdmittedKind reads in-memory GVK without marshaling object contents.
func checkRuntimeObjectMatchesAdmittedKind(name string, object runtime.Object, admittedKind schema.GroupVersionKind) error {
	if object == nil {
		return nil
	}
	objectKind := object.GetObjectKind()
	if objectKind == nil {
		return fmt.Errorf("%s has no kind", name)
	}
	return checkGVKMatchesAdmittedKind(name, objectKind.GroupVersionKind(), admittedKind)
}

// checkRawObjectMatchesAdmittedKind reads JSON TypeMeta; empty data is absent.
func checkRawObjectMatchesAdmittedKind(name string, raw []byte, admittedKind schema.GroupVersionKind) error {
	if len(raw) == 0 {
		return nil
	}
	var tm metav1.TypeMeta
	if err := json.Unmarshal(raw, &tm); err != nil {
		return fmt.Errorf("%s is undecodable: %w", name, err)
	}
	gv, err := schema.ParseGroupVersion(tm.APIVersion)
	if err != nil {
		return fmt.Errorf("%s apiVersion %q is invalid: %w", name, tm.APIVersion, err)
	}
	return checkGVKMatchesAdmittedKind(name, gv.WithKind(tm.Kind), admittedKind)
}

func checkGVKMatchesAdmittedKind(name string, gvk schema.GroupVersionKind, admittedKind schema.GroupVersionKind) error {
	if gvk.Version == "" || gvk.Kind == "" {
		return fmt.Errorf("%s has no apiVersion or kind", name)
	}
	if gvk != admittedKind {
		return fmt.Errorf("%s GVK %s differs from the admitted kind", name, gvk.String())
	}
	return nil
}
