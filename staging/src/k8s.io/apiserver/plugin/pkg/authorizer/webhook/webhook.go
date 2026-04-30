/*
Copyright 2016 The Kubernetes Authors.

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

// Package webhook implements the authorizer.Authorizer interface using HTTP webhooks.
package webhook

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"time"

	v1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/cache"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/apiserver"
	apiservervalidation "k8s.io/apiserver/pkg/apis/apiserver/validation"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authorizationcel "k8s.io/apiserver/pkg/authorization/cel"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics"
	"k8s.io/client-go/kubernetes/scheme"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
	authorizationv1alpha1client "k8s.io/client-go/kubernetes/typed/authorization/v1alpha1"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

const (
	// The maximum length of requester-controlled attributes to allow caching.
	maxControlledAttrCacheSize = 10000
)

// DefaultRetryBackoff returns the default backoff parameters for webhook retry.
func DefaultRetryBackoff() *wait.Backoff {
	backoff := webhook.DefaultRetryBackoffWithInitialDelay(500 * time.Millisecond)
	return &backoff
}

// Ensure Webhook implements the authorizer.Authorizer interface.
var _ authorizer.Authorizer = (*WebhookAuthorizer)(nil)

type subjectAccessReviewer interface {
	Create(context.Context, *authorizationv1.SubjectAccessReview, metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, int, error)
}

type authorizationConditionsReviewer interface {
	Create(context.Context, *authorizationv1alpha1.AuthorizationConditionsReview, metav1.CreateOptions) (*authorizationv1alpha1.AuthorizationConditionsReview, int, error)
}

type WebhookAuthorizer struct {
	subjectAccessReview             subjectAccessReviewer
	authorizationConditionsReviewer authorizationConditionsReviewer
	responseCache                   *cache.LRUExpireCache
	authorizedTTL                   time.Duration
	unauthorizedTTL                 time.Duration
	retryBackoff                    wait.Backoff
	decisionOnError                 authorizer.Decision
	metrics                         metrics.AuthorizerMetrics
	celMatcher                      *authorizationcel.CELMatcher
	name                            string
}

// NewFromInterface creates a WebhookAuthorizer using the given subjectAccessReview client.
// If conditionsWebhookConfig is non-nil, an AuthorizationConditionsReview client will be
// built from it to support conditional authorization evaluation.
func NewFromInterface(subjectAccessReview authorizationv1client.AuthorizationV1Interface, authorizedTTL, unauthorizedTTL time.Duration, retryBackoff wait.Backoff, decisionOnError authorizer.Decision, metrics metrics.AuthorizerMetrics, compiler authorizationcel.Compiler, authorizationV1Alpha1Interface authorizationv1alpha1client.AuthorizationV1alpha1Interface) (*WebhookAuthorizer, error) {
	var conditionsReviewer authorizationConditionsReviewer
	if authorizationV1Alpha1Interface != nil {
		conditionsReviewer = &authorizationConditionsReviewV1Alpha1Client{authorizationV1Alpha1Interface.RESTClient()}
	}
	return newWithBackoff(&subjectAccessReviewV1Client{subjectAccessReview.RESTClient()}, authorizedTTL, unauthorizedTTL, retryBackoff, decisionOnError, nil, metrics, compiler, "", conditionsReviewer)
}

// New creates a new WebhookAuthorizer from the provided kubeconfig file.
// TODO(luxas): This does not _actually_ build from the KubeConfig file, should we move the logic from reloadableAuthorizerResolver here?
// The config's cluster field is used to refer to the remote service, user refers to the returned authorizer.
//
//	# clusters refers to the remote service.
//	clusters:
//	- name: name-of-remote-authz-service
//	  cluster:
//	    certificate-authority: /path/to/ca.pem      # CA for verifying the remote service.
//	    server: https://authz.example.com/authorize # URL of remote service to query. Must use 'https'.
//
//	# users refers to the API server's webhook configuration.
//	users:
//	- name: name-of-api-server
//	  user:
//	    client-certificate: /path/to/cert.pem # cert for the webhook plugin to use
//	    client-key: /path/to/key.pem          # key matching the cert
//
// For additional HTTP configuration, refer to the kubeconfig documentation
// https://kubernetes.io/docs/user-guide/kubeconfig-file/.
func New(sarConfig *rest.Config, sarVersion string, authorizedTTL, unauthorizedTTL time.Duration, retryBackoff wait.Backoff, decisionOnError authorizer.Decision, matchConditions []apiserver.WebhookMatchCondition, name string, metrics metrics.AuthorizerMetrics, compiler authorizationcel.Compiler, conditionsReviewConfig *rest.Config, conditionsReviewVersion string) (*WebhookAuthorizer, error) {
	subjectAccessReview, err := subjectAccessReviewInterfaceFromConfig(sarConfig, sarVersion, retryBackoff)
	if err != nil {
		return nil, err
	}
	var conditionsReviewer authorizationConditionsReviewer
	if conditionsReviewConfig != nil {
		conditionsReviewer, err = authorizationConditionsReviewInterfaceFromConfig(conditionsReviewConfig, conditionsReviewVersion, retryBackoff)
		if err != nil {
			return nil, err
		}
	}

	return newWithBackoff(subjectAccessReview, authorizedTTL, unauthorizedTTL, retryBackoff, decisionOnError, matchConditions, metrics, compiler, name, conditionsReviewer)
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(subjectAccessReview subjectAccessReviewer, authorizedTTL, unauthorizedTTL time.Duration, retryBackoff wait.Backoff, decisionOnError authorizer.Decision, matchConditions []apiserver.WebhookMatchCondition, am metrics.AuthorizerMetrics, compiler authorizationcel.Compiler, name string, authorizationConditionsReviewer authorizationConditionsReviewer) (*WebhookAuthorizer, error) {
	// compile all expressions once in validation and save the results to be used for eval later
	cm, fieldErr := apiservervalidation.ValidateAndCompileMatchConditions(compiler, matchConditions)
	if err := fieldErr.ToAggregate(); err != nil {
		return nil, err
	}
	if cm != nil {
		cm.AuthorizerType = "Webhook"
		cm.AuthorizerName = name
		cm.Metrics = am
	}
	return &WebhookAuthorizer{
		subjectAccessReview:             subjectAccessReview,
		authorizationConditionsReviewer: authorizationConditionsReviewer,
		responseCache:                   cache.NewLRUExpireCache(8192),
		authorizedTTL:                   authorizedTTL,
		unauthorizedTTL:                 unauthorizedTTL,
		retryBackoff:                    retryBackoff,
		decisionOnError:                 decisionOnError,
		metrics:                         am,
		celMatcher:                      cm,
		name:                            name,
	}, nil
}

// Authorize makes a REST request to the remote service describing the attempted action as a JSON
// serialized api.authorization.v1beta1.SubjectAccessReview object. An example request body is
// provided below.
//
//	{
//	  "apiVersion": "authorization.k8s.io/v1beta1",
//	  "kind": "SubjectAccessReview",
//	  "spec": {
//	    "resourceAttributes": {
//	      "namespace": "kittensandponies",
//	      "verb": "GET",
//	      "group": "group3",
//	      "resource": "pods"
//	    },
//	    "user": "jane",
//	    "group": [
//	      "group1",
//	      "group2"
//	    ]
//	  }
//	}
//
// The remote service is expected to fill the SubjectAccessReviewStatus field to either allow or
// disallow access. A permissive response would return:
//
//	{
//	  "apiVersion": "authorization.k8s.io/v1beta1",
//	  "kind": "SubjectAccessReview",
//	  "status": {
//	    "allowed": true
//	  }
//	}
//
// To disallow access, the remote service would return:
//
//	{
//	  "apiVersion": "authorization.k8s.io/v1beta1",
//	  "kind": "SubjectAccessReview",
//	  "status": {
//	    "allowed": false,
//	    "reason": "user does not have read access to the namespace"
//	  }
//	}
//
// TODO(mikedanese): We should eventually support failing closed when we
// encounter an error. We are failing open now to preserve backwards compatible
// behavior.
func (w *WebhookAuthorizer) Authorize(ctx context.Context, attr authorizer.Attributes) (decision authorizer.Decision, reason string, err error) {
	r := &authorizationv1.SubjectAccessReview{}
	if user := attr.GetUser(); user != nil {
		r.Spec = authorizationv1.SubjectAccessReviewSpec{
			User:   user.GetName(),
			UID:    user.GetUID(),
			Groups: user.GetGroups(),
			Extra:  convertToSARExtra(user.GetExtra()),
		}
	}

	if attr.IsResourceRequest() {
		r.Spec.ResourceAttributes = resourceAttributesFrom(attr)
	} else {
		r.Spec.NonResourceAttributes = &authorizationv1.NonResourceAttributes{
			Path: attr.GetPath(),
			Verb: attr.GetVerb(),
		}
	}
	// Process Match Conditions before calling the webhook
	matches, err := w.match(ctx, r)
	// If at least one matchCondition evaluates to an error (but none are FALSE):
	// If failurePolicy=Deny, then the webhook rejects the request
	// If failurePolicy=NoOpinion, then the error is ignored and the webhook is skipped
	if err != nil {
		return w.decisionOnError, "", err
	}
	// If at least one matchCondition successfully evaluates to FALSE,
	// then the webhook is skipped.
	if !matches {
		return authorizer.DecisionNoOpinion, "", nil
	}

	r.Status, err = w.sendSARWebhook(ctx, r, attr)
	if err != nil {
		return w.decisionOnError, "", err
	}

	switch {
	case r.Status.Denied && r.Status.Allowed:
		return authorizer.DecisionDeny, r.Status.Reason, fmt.Errorf("webhook subject access review returned both allow and deny response")
	case r.Status.ConditionalDecision != nil:
		// Fail with Deny if there is at least one Deny condition or decision.
		if shouldFailWithDeny(*r.Status.ConditionalDecision) {
			return authorizer.DecisionDeny, "webhook authorizer tried to return conditional decision although client does not support it", nil
		}
		return authorizer.DecisionNoOpinion, "webhook authorizer tried to return conditional decision although client does not support it", nil
	case r.Status.Denied:
		return authorizer.DecisionDeny, r.Status.Reason, nil
	case r.Status.Allowed:
		return authorizer.DecisionAllow, r.Status.Reason, nil
	default:
		return authorizer.DecisionNoOpinion, r.Status.Reason, nil
	}
}

func shouldFailWithDeny(decision authorizationv1.ConditionsAwareDecision) bool {
	switch decision.Type {
	case authorizationv1.ConditionsAwareDecisionTypeAllow, authorizationv1.ConditionsAwareDecisionTypeNoOpinion:
		return false
	case authorizationv1.ConditionsAwareDecisionTypeDeny:
		return true
	case authorizationv1.ConditionsAwareDecisionTypeConditionsMap:
		if decision.ConditionsMap != nil {
			for _, cond := range decision.ConditionsMap.Conditions {
				if cond.Effect == authorizationv1.ConditionEffectDeny {
					return true
				}
			}
		}
		return false
	case authorizationv1.ConditionsAwareDecisionTypeUnion:
		// Note: This is slightly stricter than it needs to be, as what is returned is not necessarily pruned
		// the way the union constructor would do it. For example, Union[NoOpinion, Allow, ConditionsMap],
		// where the ConditionsMap has some Deny condition, yields shouldFailWithDeny == true, even though
		// that whole Union is equal to an Allow, in fact. However, the approximation is stricter than it needs
		// to be here, which is good for safety.
		return slices.ContainsFunc(decision.Union, shouldFailWithDeny)
	default:
		return true
	}
}

func (w *WebhookAuthorizer) ConditionsAwareAuthorize(ctx context.Context, attr authorizer.Attributes) authorizer.ConditionsAwareDecision {
	r := &authorizationv1.SubjectAccessReview{}
	if user := attr.GetUser(); user != nil {
		r.Spec = authorizationv1.SubjectAccessReviewSpec{
			User:   user.GetName(),
			UID:    user.GetUID(),
			Groups: user.GetGroups(),
			Extra:  convertToSARExtra(user.GetExtra()),
		}
	}
	r.Spec.ConditionalAuthorization = &authorizationv1.ConditionalAuthorizationOptions{
		Enabled: true,
	}

	if attr.IsResourceRequest() {
		r.Spec.ResourceAttributes = resourceAttributesFrom(attr)
	} else {
		r.Spec.NonResourceAttributes = &authorizationv1.NonResourceAttributes{
			Path: attr.GetPath(),
			Verb: attr.GetVerb(),
		}
	}

	// Process Match Conditions before calling the webhook
	matches, err := w.match(ctx, r)
	// If at least one matchCondition evaluates to an error (but none are FALSE):
	// If failurePolicy=Deny, then the webhook rejects the request
	// If failurePolicy=NoOpinion, then the error is ignored and the webhook is skipped
	if err != nil {
		return w.conditionsAwareFailClosedDecision(err)
	}
	// If at least one matchCondition successfully evaluates to FALSE,
	// then the webhook is skipped.
	if !matches {
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	r.Status, err = w.sendSARWebhook(ctx, r, attr)
	if err != nil {
		return w.conditionsAwareFailClosedDecision(err)
	}

	switch {
	case r.Status.Denied && r.Status.Allowed:
		return authorizer.ConditionsAwareDecisionDeny(r.Status.Reason, fmt.Errorf("webhook subject access review returned both allow and deny response"))
	case r.Status.Denied && r.Status.ConditionalDecision != nil:
		return authorizer.ConditionsAwareDecisionDeny(r.Status.Reason, fmt.Errorf("webhook subject access review returned both conditional and deny response"))
	case r.Status.Allowed && r.Status.ConditionalDecision != nil:
		return authorizer.ConditionsAwareDecisionDeny(r.Status.Reason, fmt.Errorf("webhook subject access review returned both conditional and allow response"))
	case r.Status.Denied:
		return authorizer.ConditionsAwareDecisionDeny(r.Status.Reason, nil)
	case r.Status.Allowed:
		return authorizer.ConditionsAwareDecisionAllow(r.Status.Reason, nil)
	case r.Status.ConditionalDecision != nil:
		return deserializeDecision(*r.Status.ConditionalDecision, w.conditionsAwareFailClosedDecision)
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion(r.Status.Reason, nil)
	}
}

func (w *WebhookAuthorizer) conditionsAwareFailClosedDecision(err error) authorizer.ConditionsAwareDecision {
	if w.decisionOnError == authorizer.DecisionNoOpinion {
		return authorizer.ConditionsAwareDecisionNoOpinion("", err)
	}
	return authorizer.ConditionsAwareDecisionDeny("", err)
}

func (w *WebhookAuthorizer) sendSARWebhook(ctx context.Context, r *authorizationv1.SubjectAccessReview, attr authorizer.Attributes) (authorizationv1.SubjectAccessReviewStatus, error) {
	// If all evaluated successfully and ALL matchConditions evaluate to TRUE,
	// then the webhook is called.
	key, err := json.Marshal(r.Spec)
	if err != nil {
		return authorizationv1.SubjectAccessReviewStatus{}, err
	}
	if entry, ok := w.responseCache.Get(string(key)); ok {
		return entry.(authorizationv1.SubjectAccessReviewStatus), nil
	} else {
		var result *authorizationv1.SubjectAccessReview
		var metricsResult string
		// WithExponentialBackoff will return SAR create error (sarErr) if any.
		if err := webhook.WithExponentialBackoff(ctx, w.retryBackoff, func() error {
			var sarErr error
			var statusCode int

			start := time.Now()
			result, statusCode, sarErr = w.subjectAccessReview.Create(ctx, r, metav1.CreateOptions{})
			latency := time.Since(start)

			switch {
			case sarErr == nil:
				metricsResult = "success"
			case ctx.Err() != nil:
				metricsResult = "canceled"
			case utilnet.IsTimeout(sarErr) || errors.Is(sarErr, context.DeadlineExceeded) || apierrors.IsTimeout(sarErr) || statusCode == http.StatusGatewayTimeout:
				metricsResult = "timeout"
			default:
				metricsResult = "error"
			}
			w.metrics.RecordWebhookEvaluation(ctx, w.name, metricsResult)
			w.metrics.RecordWebhookDuration(ctx, w.name, metricsResult, latency.Seconds())

			if statusCode != 0 {
				w.metrics.RecordRequestTotal(ctx, strconv.Itoa(statusCode))
				w.metrics.RecordRequestLatency(ctx, strconv.Itoa(statusCode), latency.Seconds())
				return sarErr
			}

			if sarErr != nil {
				w.metrics.RecordRequestTotal(ctx, "<error>")
				w.metrics.RecordRequestLatency(ctx, "<error>", latency.Seconds())
			}

			return sarErr
		}, webhook.DefaultShouldRetry); err != nil {
			klog.Errorf("Failed to make webhook authorizer request: %v", err)

			// we're returning NoOpinion, and the parent context has not timed out or been canceled
			if w.decisionOnError == authorizer.DecisionNoOpinion && ctx.Err() == nil {
				w.metrics.RecordWebhookFailOpen(ctx, w.name, metricsResult)
			}

			return authorizationv1.SubjectAccessReviewStatus{}, err
		}

		// TODO: There is a discrepancy between the if and the else branch, else writes into the r pointer, if does not.
		r.Status = result.Status
		if shouldCache(attr) {
			if r.Status.Allowed {
				w.responseCache.Add(string(key), r.Status, w.authorizedTTL)
			} else {
				w.responseCache.Add(string(key), r.Status, w.unauthorizedTTL)
			}
		}
		return r.Status, nil
	}
}

func (w *WebhookAuthorizer) EvaluateConditions(ctx context.Context, decision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if decision.IsUnconditional() {
		return decision.UnconditionalParts()
	}

	// Fail closed when evaluation is not supported
	if w.authorizationConditionsReviewer == nil {
		return decision.FailClosedDecision(), "failed closed", fmt.Errorf("no authorization conditions review client configured for the webhook authorizer, cannot evaluate conditions")
	}

	// TODO(luxas): Use builtin evaluators to resolve as much as possible of the ConditionsMap or Union

	r := &authorizationv1alpha1.AuthorizationConditionsReview{
		Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
			Decision: serializeConditionsAwareDecision(decision),
		},
	}

	if ac := data.AdmissionControl; ac != nil {
		var serializedUserInfo authenticationv1.UserInfo
		if userInfo := ac.GetUserInfo(); userInfo != nil {
			serializedUserInfo = authenticationv1.UserInfo{
				Username: userInfo.GetName(),
				Groups:   userInfo.GetGroups(),
				UID:      userInfo.GetUID(),
				Extra:    convertToAuthenticationExtra(userInfo.GetExtra()),
			}
		}
		r.Request.AdmissionControlData = &authorizationv1alpha1.AuthorizationConditionsTargetAdmissionControl{
			RequestKind:        ptr.To(metav1.GroupVersionKind(ac.GetKind())),
			RequestResource:    ptr.To(metav1.GroupVersionResource(ac.GetResource())),
			RequestSubResource: ac.GetSubresource(),
			Name:               ac.GetName(),
			Namespace:          ac.GetNamespace(),
			Operation:          v1.Operation(ac.GetOperation()),
			UserInfo:           serializedUserInfo,
			Object: runtime.RawExtension{
				Object: ac.GetObject(),
			},
			OldObject: runtime.RawExtension{
				Object: ac.GetOldObject(),
			},
			DryRun: ptr.To(ac.IsDryRun()),
			Options: runtime.RawExtension{
				Object: ac.GetOperationOptions(),
			},
		}
	}

	var result *authorizationv1alpha1.AuthorizationConditionsReview
	var metricsResult string
	// WithExponentialBackoff will return SAR create error (sarErr) if any.
	if err := webhook.WithExponentialBackoff(ctx, w.retryBackoff, func() error {
		var acrErr error
		result, _, acrErr = w.authorizationConditionsReviewer.Create(ctx, r, metav1.CreateOptions{})
		// TODO(luxas): add metrics

		return acrErr
	}, webhook.DefaultShouldRetry); err != nil {
		klog.Errorf("Failed to make webhook authorizer evaluation request: %v", err)

		// we're returning NoOpinion, and the parent context has not timed out or been canceled
		if w.decisionOnError == authorizer.DecisionNoOpinion && ctx.Err() == nil {
			w.metrics.RecordWebhookFailOpen(ctx, w.name, metricsResult)
		}

		return decision.FailClosedDecision(), "failed closed", err
	}

	if result.Response == nil {
		return authorizer.DecisionNoOpinion, "", nil
	}
	evaluationError := deserializeEvaluationError(result.Response.Decision.EvaluationError)

	switch result.Response.Decision.Type {
	case authorizationv1alpha1.ConditionsAwareDecisionTypeAllow:
		return authorizer.DecisionAllow, result.Response.Decision.Reason, evaluationError
	case authorizationv1alpha1.ConditionsAwareDecisionTypeDeny:
		return authorizer.DecisionDeny, result.Response.Decision.Reason, evaluationError
	case authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion:
		return authorizer.DecisionNoOpinion, result.Response.Decision.Reason, evaluationError
	default:
		return decision.FailClosedDecision(), "failed closed", fmt.Errorf("unrecognized decision type %q", result.Response.Decision.Type)
	}
}

func deserializeEvaluationError(evalErr string) error {
	if len(evalErr) == 0 {
		return nil
	}
	return errors.New(evalErr)
}

func deserializeDecision(serializedDecision authorizationv1.ConditionsAwareDecision, failClosed func(error) authorizer.ConditionsAwareDecision) authorizer.ConditionsAwareDecision {
	evaluationError := deserializeEvaluationError(serializedDecision.EvaluationError)

	switch serializedDecision.Type {
	case authorizationv1.ConditionsAwareDecisionTypeAllow:
		return authorizer.ConditionsAwareDecisionAllow(serializedDecision.Reason, evaluationError)
	case authorizationv1.ConditionsAwareDecisionTypeDeny:
		return authorizer.ConditionsAwareDecisionDeny(serializedDecision.Reason, evaluationError)
	case authorizationv1.ConditionsAwareDecisionTypeNoOpinion:
		return authorizer.ConditionsAwareDecisionNoOpinion(serializedDecision.Reason, evaluationError)
	case authorizationv1.ConditionsAwareDecisionTypeConditionsMap:
		if serializedDecision.ConditionsMap != nil {
			deserializedConditions := make([]authorizer.Condition, len(serializedDecision.ConditionsMap.Conditions))
			for i, serialized := range serializedDecision.ConditionsMap.Conditions {
				deserializedConditions[i] = authorizer.GenericCondition{
					ID:          serialized.ID,
					Effect:      authorizer.ConditionEffect(serialized.Effect),
					Condition:   serialized.Condition,
					Type:        serialized.Type,
					Description: serialized.Description,
				}
			}

			return authorizer.ConditionsAwareDecisionConditionsMap(deserializedConditions...)
		}
		return authorizer.ConditionsAwareDecisionNoOpinion(serializedDecision.Reason, evaluationError)
	case authorizationv1.ConditionsAwareDecisionTypeUnion:
		if len(serializedDecision.Union) != 0 {
			subDecisions := make([]authorizer.ConditionsAwareDecision, 0, len(serializedDecision.Union))
			for _, serializedSubDecision := range serializedDecision.Union {
				subDecisions = append(subDecisions, deserializeDecision(serializedSubDecision, failClosed))
			}
			return authorizer.ConditionsAwareDecisionUnion(subDecisions...)
		}
		return authorizer.ConditionsAwareDecisionNoOpinion(serializedDecision.Reason, evaluationError)
	default:
		return failClosed(fmt.Errorf("unrecognized ConditionsAwareDecision.type=%q", serializedDecision.Type))
	}
}

// TODO(luxas): Deduplicate this code with authorizationutil (helpers.go)
func serializeConditionsAwareDecision(decision authorizer.ConditionsAwareDecision) authorizationv1alpha1.ConditionsAwareDecision {
	var errString string
	if decision.Error() != nil {
		errString = decision.Error().Error()
	}
	switch {
	case decision.IsAllowed():
		return authorizationv1alpha1.ConditionsAwareDecision{
			Type:            authorizationv1alpha1.ConditionsAwareDecisionTypeAllow,
			Reason:          decision.Reason(),
			EvaluationError: errString,
		}
	case decision.IsNoOpinion():
		return authorizationv1alpha1.ConditionsAwareDecision{
			Type:            authorizationv1alpha1.ConditionsAwareDecisionTypeNoOpinion,
			Reason:          decision.Reason(),
			EvaluationError: errString,
		}
	case decision.IsConditionsMap():
		conds := make([]authorizationv1alpha1.Condition, decision.ConditionsMap().Length())
		i := 0
		for condition := range decision.ConditionsMap().Conditions() {
			conds[i] = authorizationv1alpha1.Condition{
				ID:          condition.GetID(),
				Effect:      authorizationv1alpha1.ConditionEffect(condition.GetEffect()),
				Condition:   condition.GetCondition(),
				Type:        condition.GetType(),
				Description: condition.GetDescription(),
			}
			i += 1
		}

		return authorizationv1alpha1.ConditionsAwareDecision{
			Type:            authorizationv1alpha1.ConditionsAwareDecisionTypeConditionsMap,
			Reason:          decision.Reason(),
			EvaluationError: errString,
			ConditionsMap: &authorizationv1alpha1.ConditionsMap{
				Conditions: conds,
			},
		}
	case decision.IsUnion():
		subDecisions := []authorizationv1alpha1.ConditionsAwareDecision{}
		for _, subDecision := range decision.UnionedDecisions() {
			subDecisions = append(subDecisions, serializeConditionsAwareDecision(subDecision))
		}
		return authorizationv1alpha1.ConditionsAwareDecision{
			Type: authorizationv1alpha1.ConditionsAwareDecisionTypeUnion,
			// Reason and EvaluationError are not serialized in Unions, as that information is anyways
			// available when reading the leaves.
			Union: subDecisions,
		}
	default:
		// If none of the other cases matched, it's a Deny
		return authorizationv1alpha1.ConditionsAwareDecision{
			Type:            authorizationv1alpha1.ConditionsAwareDecisionTypeDeny,
			Reason:          decision.Reason(),
			EvaluationError: errString,
		}
	}
}

func resourceAttributesFrom(attr authorizer.Attributes) *authorizationv1.ResourceAttributes {
	ret := &authorizationv1.ResourceAttributes{
		Namespace:   attr.GetNamespace(),
		Verb:        attr.GetVerb(),
		Group:       attr.GetAPIGroup(),
		Version:     attr.GetAPIVersion(),
		Resource:    attr.GetResource(),
		Subresource: attr.GetSubresource(),
		Name:        attr.GetName(),
	}

	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		// If we are able to get any requirements while parsing selectors, use them, even if there's an error.
		// This is because selectors only narrow, so if a subset of selector requirements are available, the request can be allowed.
		if selectorRequirements, _ := fieldSelectorToAuthorizationAPI(attr); len(selectorRequirements) > 0 {
			ret.FieldSelector = &authorizationv1.FieldSelectorAttributes{
				Requirements: selectorRequirements,
			}
		}

		if selectorRequirements, _ := labelSelectorToAuthorizationAPI(attr); len(selectorRequirements) > 0 {
			ret.LabelSelector = &authorizationv1.LabelSelectorAttributes{
				Requirements: selectorRequirements,
			}
		}
	}

	return ret
}

func fieldSelectorToAuthorizationAPI(attr authorizer.Attributes) ([]metav1.FieldSelectorRequirement, error) {
	requirements, getFieldSelectorErr := attr.GetFieldSelector()
	if len(requirements) == 0 {
		return nil, getFieldSelectorErr
	}

	retRequirements := []metav1.FieldSelectorRequirement{}
	for _, requirement := range requirements {
		retRequirement := metav1.FieldSelectorRequirement{}
		switch {
		case requirement.Operator == selection.Equals || requirement.Operator == selection.DoubleEquals || requirement.Operator == selection.In:
			retRequirement.Operator = metav1.FieldSelectorOpIn
			retRequirement.Key = requirement.Field
			retRequirement.Values = []string{requirement.Value}
		case requirement.Operator == selection.NotEquals || requirement.Operator == selection.NotIn:
			retRequirement.Operator = metav1.FieldSelectorOpNotIn
			retRequirement.Key = requirement.Field
			retRequirement.Values = []string{requirement.Value}
		default:
			// ignore this particular requirement. since requirements are AND'd, it is safe to ignore unknown requirements
			// for authorization since the resulting check will only be as broad or broader than the intended.
			continue
		}
		retRequirements = append(retRequirements, retRequirement)
	}

	if len(retRequirements) == 0 {
		// this means that all requirements were dropped (likely due to unknown operators), so we are checking the broader
		// unrestricted action.
		return nil, getFieldSelectorErr
	}
	return retRequirements, getFieldSelectorErr
}

func labelSelectorToAuthorizationAPI(attr authorizer.Attributes) ([]metav1.LabelSelectorRequirement, error) {
	requirements, getLabelSelectorErr := attr.GetLabelSelector()
	if len(requirements) == 0 {
		return nil, getLabelSelectorErr
	}

	retRequirements := []metav1.LabelSelectorRequirement{}
	for _, requirement := range requirements {
		retRequirement := metav1.LabelSelectorRequirement{
			Key: requirement.Key(),
		}
		if values := requirement.ValuesUnsorted(); len(values) > 0 {
			retRequirement.Values = values
		}
		switch requirement.Operator() {
		case selection.Equals, selection.DoubleEquals, selection.In:
			retRequirement.Operator = metav1.LabelSelectorOpIn
		case selection.NotEquals, selection.NotIn:
			retRequirement.Operator = metav1.LabelSelectorOpNotIn
		case selection.Exists:
			retRequirement.Operator = metav1.LabelSelectorOpExists
		case selection.DoesNotExist:
			retRequirement.Operator = metav1.LabelSelectorOpDoesNotExist
		default:
			// ignore this particular requirement. since requirements are AND'd, it is safe to ignore unknown requirements
			// for authorization since the resulting check will only be as broad or broader than the intended.
			continue
		}
		retRequirements = append(retRequirements, retRequirement)
	}

	if len(retRequirements) == 0 {
		// this means that all requirements were dropped (likely due to unknown operators), so we are checking the broader
		// unrestricted action.
		return nil, getLabelSelectorErr
	}
	return retRequirements, getLabelSelectorErr
}

// TODO: need to finish the method to get the rules when using webhook mode
func (w *WebhookAuthorizer) RulesFor(ctx context.Context, user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	var (
		resourceRules    []authorizer.ResourceRuleInfo
		nonResourceRules []authorizer.NonResourceRuleInfo
	)
	incomplete := true
	return resourceRules, nonResourceRules, incomplete, fmt.Errorf("webhook authorizer does not support user rule resolution")
}

// Match is used to evaluate the SubjectAccessReviewSpec against
// the authorizer's matchConditions in the form of cel expressions
// to return match or no match found, which then is used to
// determine if the webhook should be skipped.
func (w *WebhookAuthorizer) match(ctx context.Context, r *authorizationv1.SubjectAccessReview) (bool, error) {
	// A nil celMatcher or zero saved CompilationResults matches all requests.
	if w.celMatcher == nil || w.celMatcher.CompilationResults == nil {
		return true, nil
	}
	return w.celMatcher.Eval(ctx, r)
}

func convertToSARExtra(extra map[string][]string) map[string]authorizationv1.ExtraValue {
	if extra == nil {
		return nil
	}
	ret := map[string]authorizationv1.ExtraValue{}
	for k, v := range extra {
		ret[k] = authorizationv1.ExtraValue(v)
	}

	return ret
}

func convertToAuthenticationExtra(extra map[string][]string) map[string]authenticationv1.ExtraValue {
	if extra == nil {
		return nil
	}
	ret := map[string]authenticationv1.ExtraValue{}
	for k, v := range extra {
		ret[k] = authenticationv1.ExtraValue(v)
	}

	return ret
}

func authorizationConditionsReviewInterfaceFromConfig(config *rest.Config, version string, retryBackoff wait.Backoff) (*authorizationConditionsClientGW, error) {
	localScheme := runtime.NewScheme()
	if err := authorizationv1alpha1.AddToScheme(localScheme); err != nil {
		return nil, err
	}
	switch version {
	case authorizationv1alpha1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authorizationv1alpha1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, config, groupVersions, retryBackoff)
		if err != nil {
			return nil, err
		}
		return &authorizationConditionsClientGW{gw.RestClient}, nil
	default:
		return nil, fmt.Errorf(
			"unsupported webhook conditions review version %q, supported versions are %v",
			version,
			[]string{authorizationv1alpha1.SchemeGroupVersion.Version},
		)
	}
}

// subjectAccessReviewInterfaceFromConfig builds a client from the specified kubeconfig file,
// and returns a SubjectAccessReviewInterface that uses that client. Note that the client submits SubjectAccessReview
// requests to the exact path specified in the kubeconfig file, so arbitrary non-API servers can be targeted.
func subjectAccessReviewInterfaceFromConfig(config *rest.Config, version string, retryBackoff wait.Backoff) (subjectAccessReviewer, error) {
	localScheme := runtime.NewScheme()
	if err := scheme.AddToScheme(localScheme); err != nil {
		return nil, err
	}

	switch version {
	case authorizationv1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authorizationv1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, config, groupVersions, retryBackoff)
		if err != nil {
			return nil, err
		}
		return &subjectAccessReviewV1ClientGW{gw.RestClient}, nil

	case authorizationv1beta1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authorizationv1beta1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, config, groupVersions, retryBackoff)
		if err != nil {
			return nil, err
		}
		return &subjectAccessReviewV1beta1ClientGW{gw.RestClient}, nil

	default:
		return nil, fmt.Errorf(
			"unsupported webhook authorizer version %q, supported versions are %v",
			version,
			[]string{
				authorizationv1.SchemeGroupVersion.Version,
				authorizationv1beta1.SchemeGroupVersion.Version,
			},
		)
	}
}

type subjectAccessReviewV1Client struct {
	client rest.Interface
}

func (t *subjectAccessReviewV1Client) Create(ctx context.Context, subjectAccessReview *authorizationv1.SubjectAccessReview, opts metav1.CreateOptions) (result *authorizationv1.SubjectAccessReview, statusCode int, err error) {
	result = &authorizationv1.SubjectAccessReview{}

	restResult := t.client.Post().
		Resource("subjectaccessreviews").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(subjectAccessReview).
		Do(ctx)

	restResult.StatusCode(&statusCode)
	err = restResult.Into(result)
	return
}

type authorizationConditionsReviewV1Alpha1Client struct {
	client rest.Interface
}

func (t *authorizationConditionsReviewV1Alpha1Client) Create(ctx context.Context, authorizationConditionsReview *authorizationv1alpha1.AuthorizationConditionsReview, opts metav1.CreateOptions) (result *authorizationv1alpha1.AuthorizationConditionsReview, statusCode int, err error) {
	result = &authorizationv1alpha1.AuthorizationConditionsReview{}

	restResult := t.client.Post().
		Resource("authorizationconditionsreviews").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(authorizationConditionsReview).
		Do(ctx)

	restResult.StatusCode(&statusCode)
	err = restResult.Into(result)
	return
}

// authorizationConditionsClientGW used by the generic webhook, doesn't specify GVR.
type authorizationConditionsClientGW struct {
	client rest.Interface
}

func (t *authorizationConditionsClientGW) Create(ctx context.Context, authorizationConditionsReview *authorizationv1alpha1.AuthorizationConditionsReview, _ metav1.CreateOptions) (*authorizationv1alpha1.AuthorizationConditionsReview, int, error) {
	var statusCode int
	result := &authorizationv1alpha1.AuthorizationConditionsReview{}

	restResult := t.client.Post().Body(authorizationConditionsReview).Do(ctx)

	restResult.StatusCode(&statusCode)
	err := restResult.Into(result)

	return result, statusCode, err
}

// subjectAccessReviewV1ClientGW used by the generic webhook, doesn't specify GVR.
type subjectAccessReviewV1ClientGW struct {
	client rest.Interface
}

func (t *subjectAccessReviewV1ClientGW) Create(ctx context.Context, subjectAccessReview *authorizationv1.SubjectAccessReview, _ metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, int, error) {
	var statusCode int
	result := &authorizationv1.SubjectAccessReview{}

	restResult := t.client.Post().Body(subjectAccessReview).Do(ctx)

	restResult.StatusCode(&statusCode)
	err := restResult.Into(result)

	return result, statusCode, err
}

// subjectAccessReviewV1beta1ClientGW used by the generic webhook, doesn't specify GVR.
type subjectAccessReviewV1beta1ClientGW struct {
	client rest.Interface
}

func (t *subjectAccessReviewV1beta1ClientGW) Create(ctx context.Context, subjectAccessReview *authorizationv1.SubjectAccessReview, _ metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, int, error) {
	var statusCode int
	v1beta1Review := &authorizationv1beta1.SubjectAccessReview{Spec: v1SpecToV1beta1Spec(&subjectAccessReview.Spec)}
	v1beta1Result := &authorizationv1beta1.SubjectAccessReview{}

	restResult := t.client.Post().Body(v1beta1Review).Do(ctx)

	restResult.StatusCode(&statusCode)
	err := restResult.Into(v1beta1Result)
	if err == nil {
		subjectAccessReview.Status = v1beta1StatusToV1Status(&v1beta1Result.Status)
	}
	return subjectAccessReview, statusCode, err
}

// shouldCache determines whether it is safe to cache the given request attributes. If the
// requester-controlled attributes are too large, this may be a DoS attempt, so we skip the cache.
func shouldCache(attr authorizer.Attributes) bool {
	controlledAttrSize := int64(len(attr.GetNamespace())) +
		int64(len(attr.GetVerb())) +
		int64(len(attr.GetAPIGroup())) +
		int64(len(attr.GetAPIVersion())) +
		int64(len(attr.GetResource())) +
		int64(len(attr.GetSubresource())) +
		int64(len(attr.GetName())) +
		int64(len(attr.GetPath()))
	return controlledAttrSize < maxControlledAttrCacheSize
}

func v1beta1StatusToV1Status(in *authorizationv1beta1.SubjectAccessReviewStatus) authorizationv1.SubjectAccessReviewStatus {
	return authorizationv1.SubjectAccessReviewStatus{
		Allowed:             in.Allowed,
		Denied:              in.Denied,
		Reason:              in.Reason,
		EvaluationError:     in.EvaluationError,
		ConditionalDecision: in.ConditionalDecision,
	}
}

func v1SpecToV1beta1Spec(in *authorizationv1.SubjectAccessReviewSpec) authorizationv1beta1.SubjectAccessReviewSpec {
	return authorizationv1beta1.SubjectAccessReviewSpec{
		ResourceAttributes:       v1ResourceAttributesToV1beta1ResourceAttributes(in.ResourceAttributes),
		NonResourceAttributes:    v1NonResourceAttributesToV1beta1NonResourceAttributes(in.NonResourceAttributes),
		User:                     in.User,
		Groups:                   in.Groups,
		Extra:                    v1ExtraToV1beta1Extra(in.Extra),
		UID:                      in.UID,
		ConditionalAuthorization: in.ConditionalAuthorization,
	}
}

func v1ResourceAttributesToV1beta1ResourceAttributes(in *authorizationv1.ResourceAttributes) *authorizationv1beta1.ResourceAttributes {
	if in == nil {
		return nil
	}
	return &authorizationv1beta1.ResourceAttributes{
		Namespace:     in.Namespace,
		Verb:          in.Verb,
		Group:         in.Group,
		Version:       in.Version,
		Resource:      in.Resource,
		Subresource:   in.Subresource,
		Name:          in.Name,
		FieldSelector: in.FieldSelector,
		LabelSelector: in.LabelSelector,
	}
}

func v1NonResourceAttributesToV1beta1NonResourceAttributes(in *authorizationv1.NonResourceAttributes) *authorizationv1beta1.NonResourceAttributes {
	if in == nil {
		return nil
	}
	return &authorizationv1beta1.NonResourceAttributes{
		Path: in.Path,
		Verb: in.Verb,
	}
}

func v1ExtraToV1beta1Extra(in map[string]authorizationv1.ExtraValue) map[string]authorizationv1beta1.ExtraValue {
	if in == nil {
		return nil
	}
	ret := make(map[string]authorizationv1beta1.ExtraValue, len(in))
	for k, v := range in {
		ret[k] = authorizationv1beta1.ExtraValue(v)
	}
	return ret
}
