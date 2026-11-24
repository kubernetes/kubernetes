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
	"sync"
	"time"

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

func allowedSigningAlgs() []jose.SignatureAlgorithm {
	// Keep in sync with allowedSigningAlgs in k8s.io/apiserver's OIDC JWT authenticator;
	// component-helpers cannot import k8s.io/apiserver.
	// https://github.com/kubernetes/kubernetes/blob/0bf39fd17169a6c933a6cb403c88656cd020aabf/staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/oidc.go#L261-L274
	return []jose.SignatureAlgorithm{
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
}

func supportedSigningAlgs() []string {
	allowed := allowedSigningAlgs()
	algs := make([]string, len(allowed))
	for i, alg := range allowed {
		algs[i] = string(alg)
	}
	return algs
}

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

// verifier is unexported so callers must enter through [WithTokenVerification],
// which authenticates the Authorization header before reading or decoding the
// HTTP request body.
type verifier struct {
	verifier     *oidc.IDTokenVerifier
	mismatchFunc SubresourceGroupMismatchFunc
}

// subresourceGroupMismatchKey identifies a subresource whose admitted object
// Kind may belong to a different API group than the parent resource attested by
// the webhook token.
type subresourceGroupMismatchKey struct {
	resource    schema.GroupVersionResource
	subresource string
}

// expectedSubresourceGroupMismatchMap maps a parent resource and subresource to
// the expected (and therefore allowed) cross-group object Kind for that
// subresource.
type expectedSubresourceGroupMismatchMap map[subresourceGroupMismatchKey]schema.GroupVersionKind

// SubresourceGroupMismatchFunc reports whether req is a subresource whose
// admitted object Kind is expected to belong to a different API group than the
// parent resource attested by the webhook token.
type SubresourceGroupMismatchFunc func(ctx context.Context, req *admissionv1.AdmissionRequest) (bool, error)

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
	if req == nil {
		return errVerificationFailed
	}
	// A token carrying the wildcard "*" API group attestation is authorized for
	// all API groups, so per-group and object-kind group checks do not apply.
	if slices.Contains(groups, "*") {
		return nil
	}
	logger := klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier")
	if !slices.Contains(groups, req.Resource.Group) {
		logger.V(2).Info("Webhook token verification failed", "reason", "token admissionReviewAPIGroups does not authorize the review's API group")
		return errVerificationFailed
	}
	if err := checkRequestObjectKinds(ctx, req, v.mismatchFunc); err != nil {
		logger.V(2).Info("Admission request rejected", "reason", "invalid admission request",
			"requestUID", req.UID, "operation", req.Operation, "resource", schema.GroupVersionResource(req.Resource).String(),
			"detail", err.Error())
		return errVerificationFailed
	}
	return nil
}

func checkRequestObjectKinds(ctx context.Context, req *admissionv1.AdmissionRequest, mismatchFunc SubresourceGroupMismatchFunc) error {
	if req == nil {
		return errors.New("admission request is nil")
	}
	kindOrResourceIncomplete := req.Kind.Version == "" || req.Kind.Kind == "" || req.Resource.Version == "" || req.Resource.Resource == ""
	if kindOrResourceIncomplete {
		return errors.New("admission request kind or resource is incomplete")
	}
	if req.Kind.Group != req.Resource.Group {
		if req.SubResource == "" {
			return fmt.Errorf("object kind group %q differs from authorized resource group %q, no subresource", req.Kind.Group, req.Resource.Group)
		}
		allowed, err := mismatchFunc(ctx, req)
		if err != nil {
			return fmt.Errorf("checking subresource group mismatch: %w", err)
		}
		if !allowed {
			return fmt.Errorf("object kind group %q differs from authorized resource group %q", req.Kind.Group, req.Resource.Group)
		}
	}
	admittedKind := schema.GroupVersionKind(req.Kind)
	if err := checkObjectMatchesAdmittedKind("object", req.Object, admittedKind); err != nil {
		return err
	}
	return checkObjectMatchesAdmittedKind("oldObject", req.OldObject, admittedKind)
}

// DefaultStaticAllowedSubresourceGroupMismatch reports whether req is one of
// the built-in Kubernetes subresources whose admitted object Kind belongs to a
// different API group than the parent resource attested by the webhook token
// ("/scale", "pods/eviction", and "serviceaccounts/token"). It is the default
// [SubresourceGroupMismatchFunc] used by [WithIssuer] when none is supplied,
// and custom [SubresourceGroupMismatchFunc] implementations can call it before
// checking additional cross-group subresources.
func DefaultStaticAllowedSubresourceGroupMismatch(_ context.Context, req *admissionv1.AdmissionRequest) (bool, error) {
	if req == nil {
		return false, nil
	}

	switch schema.GroupVersionKind(req.Kind) {
	case schema.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}:
		return req.SubResource == "scale", nil
	case schema.GroupVersionKind{Group: "policy", Version: "v1", Kind: "Eviction"}:
		return req.Resource == metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"} && req.SubResource == "eviction", nil
	case authenticationv1.SchemeGroupVersion.WithKind("TokenRequest"):
		return req.Resource == metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"} && req.SubResource == "token", nil
	default:
		return false, nil
	}
}

// allowedKindResourceGroupMismatch reports whether req is a known subresource
// whose admitted object Kind is in a different API group than the parent
// resource attested by the token.
func allowedKindResourceGroupMismatch(req *admissionv1.AdmissionRequest, expectedMismatches expectedSubresourceGroupMismatchMap) bool {
	if req == nil {
		return false
	}
	expectedKind, ok := expectedMismatches[subresourceGroupMismatchKey{
		resource:    schema.GroupVersionResource(req.Resource),
		subresource: req.SubResource,
	}]
	return ok && expectedKind == schema.GroupVersionKind(req.Kind)
}

const minDiscoveryRefreshInterval = time.Minute

type cachedAggregatedDiscoveryClient interface {
	discovery.CachedDiscoveryInterfaceWithContext
	discovery.AggregatedDiscoveryInterfaceWithContext
}

// discoverSubresourceGroupMismatches queries cluster discovery via a cached
// aggregated discovery client and returns a [SubresourceGroupMismatchFunc]
// backed by the discovered subresources whose response Kind group differs from
// their parent resource group, refreshing at most once per minute on a miss.
func discoverSubresourceGroupMismatches(ctx context.Context, client cachedAggregatedDiscoveryClient) (SubresourceGroupMismatchFunc, error) {
	expectedMismatches, err := discoverSubresourceGroupMismatchMap(ctx, client)
	if err != nil {
		return nil, err
	}
	var (
		mu          sync.RWMutex
		lastRefresh = time.Now()
	)
	return func(ctx context.Context, req *admissionv1.AdmissionRequest) (bool, error) {
		mu.RLock()
		allowed := allowedKindResourceGroupMismatch(req, expectedMismatches)
		mu.RUnlock()
		if allowed {
			return true, nil
		}

		mu.Lock()
		defer mu.Unlock()
		if allowedKindResourceGroupMismatch(req, expectedMismatches) {
			return true, nil
		}
		if time.Since(lastRefresh) < minDiscoveryRefreshInterval {
			return false, nil
		}
		lastRefresh = time.Now()
		client.InvalidateWithContext(ctx)
		refreshed, err := discoverSubresourceGroupMismatchMap(ctx, client)
		if err != nil {
			return false, err
		}
		expectedMismatches = refreshed
		return allowedKindResourceGroupMismatch(req, expectedMismatches), nil
	}, nil
}

func discoverSubresourceGroupMismatchMap(ctx context.Context, client discovery.AggregatedDiscoveryInterfaceWithContext) (expectedSubresourceGroupMismatchMap, error) {
	logger := klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier")

	_, apiResourceListMap, mapOfFailures, err := client.GroupsAndMaybeResourcesWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("admission: discovering subresource group mismatches: %w", err)
	}
	if len(mapOfFailures) > 0 {
		logger.V(2).Info("admission: API resource list discovery partially failed", "failures", mapOfFailures)
	}
	if len(apiResourceListMap) == 0 {
		return nil, fmt.Errorf("admission: no API resource lists discovered")
	}

	expectedMismatches := expectedSubresourceGroupMismatchMap{}

	for listGV, list := range apiResourceListMap {
		for _, apiResource := range list.APIResources {
			parent, subresource, isSubresource := strings.Cut(apiResource.Name, "/")
			if !isSubresource || parent == "" || subresource == "" || apiResource.Kind == "" {
				continue
			}

			// empty implies group of the containing resource list, which means they are the same.
			// nothing to do here.
			subresourceGV := listGV
			if apiResource.Group != "" {
				subresourceGV.Group = apiResource.Group
			}
			if apiResource.Version != "" {
				subresourceGV.Version = apiResource.Version
			}
			if subresourceGV == listGV {
				continue
			}

			// record the expected group, version, and kind for this subresource
			expectedMismatches[subresourceGroupMismatchKey{
				resource:    listGV.WithResource(parent),
				subresource: subresource,
			}] = subresourceGV.WithKind(apiResource.Kind)
		}
	}
	return expectedMismatches, nil
}

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
