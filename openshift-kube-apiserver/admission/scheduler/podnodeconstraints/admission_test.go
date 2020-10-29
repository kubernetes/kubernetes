package podnodeconstraints

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/apis/podnodeconstraints"
)

func TestPodNodeConstraints(t *testing.T) {
	ns := metav1.NamespaceDefault
	tests := []struct {
		config           *podnodeconstraints.PodNodeConstraintsConfig
		resource         runtime.Object
		kind             schema.GroupKind
		groupresource    schema.GroupResource
		userinfo         user.Info
		reviewResponse   *authorizationv1.SubjectAccessReviewResponse
		expectedResource string
		expectedErrorMsg string
	}{
		// 0: expect unspecified defaults to not error
		{
			config:           emptyConfig(),
			resource:         defaultPod(),
			userinfo:         serviceaccount.UserInfo("", "", ""),
			reviewResponse:   reviewResponse(false, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "",
		},
		// 1: expect nodeSelector to error with user which lacks "pods/binding" access
		{
			config:           testConfig(),
			resource:         nodeSelectorPod(),
			userinfo:         serviceaccount.UserInfo("", "", ""),
			reviewResponse:   reviewResponse(false, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "node selection by label(s) [bogus] is prohibited by policy for your role",
		},
		// 2: expect nodeName to fail with user that lacks "pods/binding" access
		{
			config:           testConfig(),
			resource:         nodeNamePod(),
			userinfo:         serviceaccount.UserInfo("herpy", "derpy", ""),
			reviewResponse:   reviewResponse(false, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "node selection by nodeName is prohibited by policy for your role",
		},
		// 3: expect nodeName and nodeSelector to fail with user that lacks "pods/binding" access
		{
			config:           testConfig(),
			resource:         nodeNameNodeSelectorPod(),
			userinfo:         serviceaccount.UserInfo("herpy", "derpy", ""),
			reviewResponse:   reviewResponse(false, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "node selection by nodeName and label(s) [bogus] is prohibited by policy for your role",
		},
		// 4: expect nodeSelector to succeed with user that has "pods/binding" access
		{
			config:           testConfig(),
			resource:         nodeSelectorPod(),
			userinfo:         serviceaccount.UserInfo("openshift-infra", "daemonset-controller", ""),
			reviewResponse:   reviewResponse(true, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "",
		},
		// 5: expect nodeName to succeed with user that has "pods/binding" access
		{
			config:           testConfig(),
			resource:         nodeNamePod(),
			userinfo:         serviceaccount.UserInfo("openshift-infra", "daemonset-controller", ""),
			reviewResponse:   reviewResponse(true, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "",
		},
		// 6: expect nil config to bypass admission
		{
			config:           nil,
			resource:         defaultPod(),
			userinfo:         serviceaccount.UserInfo("", "", ""),
			reviewResponse:   reviewResponse(false, ""),
			expectedResource: "pods/binding",
			expectedErrorMsg: "",
		},
		// 7: expect nodeName to succeed with node user self targeting mirror pod
		{
			config:           testConfig(),
			resource:         nodeNameMirrorPod(),
			userinfo:         &user.DefaultInfo{Name: "system:node:frank", Groups: []string{user.NodesGroup}},
			expectedErrorMsg: "",
		},
		// 8: expect nodeName to fail with node user self targeting non-mirror pod
		{
			config:           testConfig(),
			resource:         nodeNamePod(),
			userinfo:         &user.DefaultInfo{Name: "system:node:frank", Groups: []string{user.NodesGroup}},
			expectedErrorMsg: "node selection by nodeName is prohibited by policy for your role",
		},
		// 9: expect nodeName to fail with node user non-self targeting mirror pod
		{
			config:           testConfig(),
			resource:         nodeNameMirrorPod(),
			userinfo:         &user.DefaultInfo{Name: "system:node:bob", Groups: []string{user.NodesGroup}},
			expectedErrorMsg: "node selection by nodeName is prohibited by policy for your role",
		},
		// 10: expect nodeName to fail with node user non-self targeting non-mirror pod
		{
			config:           testConfig(),
			resource:         nodeNamePod(),
			userinfo:         &user.DefaultInfo{Name: "system:node:bob", Groups: []string{user.NodesGroup}},
			expectedErrorMsg: "node selection by nodeName is prohibited by policy for your role",
		},
	}
	for i, tc := range tests {
		var expectedError error
		errPrefix := fmt.Sprintf("%d", i)
		prc := NewPodNodeConstraints(tc.config, nodeidentifier.NewDefaultNodeIdentifier())
		prc.(initializer.WantsAuthorizer).SetAuthorizer(fakeAuthorizer(t))
		err := prc.(admission.InitializationValidator).ValidateInitialization()
		if err != nil {
			checkAdmitError(t, err, expectedError, errPrefix)
			continue
		}
		attrs := admission.NewAttributesRecord(tc.resource, nil, kapi.Kind("Pod").WithVersion("version"), ns, "test", kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, tc.userinfo)
		if tc.expectedErrorMsg != "" {
			expectedError = admission.NewForbidden(attrs, fmt.Errorf(tc.expectedErrorMsg))
		}
		err = prc.(admission.ValidationInterface).Validate(context.TODO(), attrs, nil)
		checkAdmitError(t, err, expectedError, errPrefix)
	}
}

func TestPodNodeConstraintsPodUpdate(t *testing.T) {
	ns := metav1.NamespaceDefault
	var expectedError error
	errPrefix := "PodUpdate"
	prc := NewPodNodeConstraints(testConfig(), nodeidentifier.NewDefaultNodeIdentifier())
	prc.(initializer.WantsAuthorizer).SetAuthorizer(fakeAuthorizer(t))
	err := prc.(admission.InitializationValidator).ValidateInitialization()
	if err != nil {
		checkAdmitError(t, err, expectedError, errPrefix)
		return
	}
	attrs := admission.NewAttributesRecord(nodeNamePod(), nodeNamePod(), kapi.Kind("Pod").WithVersion("version"), ns, "test", kapi.Resource("pods").WithVersion("version"), "", admission.Update, nil, false, serviceaccount.UserInfo("", "", ""))
	err = prc.(admission.ValidationInterface).Validate(context.TODO(), attrs, nil)
	checkAdmitError(t, err, expectedError, errPrefix)
}

func TestPodNodeConstraintsNonHandledResources(t *testing.T) {
	ns := metav1.NamespaceDefault
	errPrefix := "ResourceQuotaTest"
	var expectedError error
	prc := NewPodNodeConstraints(testConfig(), nodeidentifier.NewDefaultNodeIdentifier())
	prc.(initializer.WantsAuthorizer).SetAuthorizer(fakeAuthorizer(t))
	err := prc.(admission.InitializationValidator).ValidateInitialization()
	if err != nil {
		checkAdmitError(t, err, expectedError, errPrefix)
		return
	}
	attrs := admission.NewAttributesRecord(resourceQuota(), nil, kapi.Kind("ResourceQuota").WithVersion("version"), ns, "test", kapi.Resource("resourcequotas").WithVersion("version"), "", admission.Create, nil, false, serviceaccount.UserInfo("", "", ""))
	err = prc.(admission.ValidationInterface).Validate(context.TODO(), attrs, nil)
	checkAdmitError(t, err, expectedError, errPrefix)
}

func emptyConfig() *podnodeconstraints.PodNodeConstraintsConfig {
	return &podnodeconstraints.PodNodeConstraintsConfig{}
}

func testConfig() *podnodeconstraints.PodNodeConstraintsConfig {
	return &podnodeconstraints.PodNodeConstraintsConfig{
		NodeSelectorLabelBlacklist: []string{"bogus"},
	}
}

func defaultPod() *kapi.Pod {
	pod := &kapi.Pod{}
	return pod
}

func nodeNameNodeSelectorPod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.NodeName = "frank"
	pod.Spec.NodeSelector = map[string]string{"bogus": "frank"}
	return pod
}

func nodeNamePod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.NodeName = "frank"
	return pod
}

func nodeNameMirrorPod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Annotations = map[string]string{kapi.MirrorPodAnnotationKey: "true"}
	pod.Spec.NodeName = "frank"
	return pod
}

func nodeSelectorPod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.NodeSelector = map[string]string{"bogus": "frank"}
	return pod
}

func resourceQuota() runtime.Object {
	rq := &kapi.ResourceQuota{}
	return rq
}

func checkAdmitError(t *testing.T, err error, expectedError error, prefix string) {
	switch {
	case expectedError == nil && err == nil:
		// continue
	case expectedError != nil && err != nil && err.Error() != expectedError.Error():
		t.Errorf("%s: expected error %q, got: %q", prefix, expectedError.Error(), err.Error())
	case expectedError == nil && err != nil:
		t.Errorf("%s: expected no error, got: %q", prefix, err.Error())
	case expectedError != nil && err == nil:
		t.Errorf("%s: expected error %q, no error received", prefix, expectedError.Error())
	}
}

type fakeTestAuthorizer struct {
	t *testing.T
}

func fakeAuthorizer(t *testing.T) authorizer.Authorizer {
	return &fakeTestAuthorizer{
		t: t,
	}
}

func (a *fakeTestAuthorizer) Authorize(_ context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
	ui := attributes.GetUser()
	if ui == nil {
		return authorizer.DecisionNoOpinion, "", fmt.Errorf("No valid UserInfo for Context")
	}
	// User with pods/bindings. permission:
	if ui.GetName() == "system:serviceaccount:openshift-infra:daemonset-controller" {
		return authorizer.DecisionAllow, "", nil
	}
	// User without pods/bindings. permission:
	return authorizer.DecisionNoOpinion, "", nil
}

func reviewResponse(allowed bool, msg string) *authorizationv1.SubjectAccessReviewResponse {
	return &authorizationv1.SubjectAccessReviewResponse{
		Allowed: allowed,
		Reason:  msg,
	}
}

func TestReadConfig(t *testing.T) {
	configStr := `apiVersion: scheduling.openshift.io/v1
kind: PodNodeConstraintsConfig
nodeSelectorLabelBlacklist:
  - bogus
  - foo
`
	buf := bytes.NewBufferString(configStr)
	config, err := readConfig(buf)
	if err != nil {
		t.Fatalf("unexpected error reading config: %v", err)
	}
	if len(config.NodeSelectorLabelBlacklist) == 0 {
		t.Fatalf("NodeSelectorLabelBlacklist didn't take specified value")
	}
}
