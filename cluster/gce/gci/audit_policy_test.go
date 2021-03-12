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

package gci

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"testing"

	"k8s.io/apiserver/pkg/apis/audit"
	auditinstall "k8s.io/apiserver/pkg/apis/audit/install"
	auditpkg "k8s.io/apiserver/pkg/audit"
	auditpolicy "k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	// Register audit scheme to parse audit config.
	auditinstall.Install(auditpkg.Scheme)
}

func TestCreateMasterAuditPolicy(t *testing.T) {
	baseDir, err := ioutil.TempDir("", "configure-helper-test") // cleaned up by c.tearDown()
	require.NoError(t, err, "Failed to create temp directory")

	policyFile := filepath.Join(baseDir, "audit_policy.yaml")
	c := ManifestTestCase{
		t:                t,
		kubeHome:         baseDir,
		manifestFuncName: fmt.Sprintf("create-master-audit-policy %s", policyFile),
	}
	defer c.tearDown()

	// Initialize required environment variables.
	c.mustInvokeFunc(
		kubeAPIServerEnv{KubeHome: c.kubeHome},
		[]string{"configure-helper.sh"},
		"base.template",
		"testdata/kube-apiserver/base.template",
	)

	policy, err := auditpolicy.LoadPolicyFromFile(policyFile)
	require.NoError(t, err, "Failed to load generated policy.")

	// Users for test cases
	var (
		anonymous           = newUserInfo(user.Anonymous, user.AllUnauthenticated)
		kubeproxy           = newUserInfo(user.KubeProxy, user.AllAuthenticated)
		ingress             = newUserInfo("system:unsecured", user.AllAuthenticated, user.SystemPrivilegedGroup)
		kubelet             = newUserInfo("kubelet", user.AllAuthenticated, user.NodesGroup)
		node                = newUserInfo("system:node:node-123", user.AllAuthenticated, user.NodesGroup)
		controller          = newUserInfo(user.KubeControllerManager, user.AllAuthenticated)
		scheduler           = newUserInfo(user.KubeScheduler, user.AllAuthenticated)
		apiserver           = newUserInfo(user.APIServerUser, user.SystemPrivilegedGroup)
		autoscaler          = newUserInfo("cluster-autoscaler", user.AllAuthenticated)
		npd                 = newUserInfo("system:node-problem-detector", user.AllAuthenticated)
		npdSA               = serviceaccount.UserInfo("kube-system", "node-problem-detector", "")
		namespaceController = serviceaccount.UserInfo("kube-system", "namespace-controller", "")
		endpointController  = serviceaccount.UserInfo("kube-system", "endpoint-controller", "")
		defaultSA           = serviceaccount.UserInfo("default", "default", "")

		allUsers = []user.Info{anonymous, kubeproxy, ingress, kubelet, node, controller, scheduler, apiserver, autoscaler, npd, npdSA, namespaceController, endpointController, defaultSA}
	)

	// Resources for test cases
	var (
		nodes           = resource("nodes")
		nodeStatus      = resource("nodes", "", "", "status")
		endpoints       = resource("endpoints", "default")
		sysEndpoints    = resource("endpoints", "kube-system")
		services        = resource("services", "default")
		serviceStatus   = resource("services", "default", "", "status")
		configmaps      = resource("configmaps", "default")
		sysConfigmaps   = resource("configmaps", "kube-system")
		namespaces      = resource("namespaces")
		namespaceStatus = resource("namespaces", "", "", "status")
		namespaceFinal  = resource("namespaces", "", "", "finalize")
		podMetrics      = resource("podmetrics", "default", "metrics.k8s.io")
		nodeMetrics     = resource("nodemetrics", "", "metrics.k8s.io")
		pods            = resource("pods", "default")
		podStatus       = resource("pods", "default", "", "status")
		secrets         = resource("secrets", "default")
		tokenReviews    = resource("tokenreviews", "", "authentication.k8s.io")
		deployments     = resource("deployments", "default", "apps")
		clusterRoles    = resource("clusterroles", "", "rbac.authorization.k8s.io")
		events          = resource("events", "default")
		foobars         = resource("foos", "default", "example.com")
		foobarbaz       = resource("foos", "default", "example.com", "baz")
	)

	// Aliases
	const (
		none     = audit.LevelNone
		metadata = audit.LevelMetadata
		request  = audit.LevelRequest
		response = audit.LevelRequestResponse
	)

	at := auditTester{
		T:       t,
		checker: auditpolicy.NewChecker(policy),
	}

	at.testResources(none, kubeproxy, "watch", endpoints, sysEndpoints, services, serviceStatus)
	at.testResources(request, kubeproxy, "watch", nodes, pods)

	at.testResources(none, ingress, "get", sysConfigmaps)
	at.testResources(metadata, ingress, "get", configmaps)

	at.testResources(none, kubelet, node, "get", nodes, nodeStatus)
	at.testResources(metadata, kubelet, node, "get", sysConfigmaps, secrets)
	at.testResources(response, kubelet, node, "create", deployments, pods)

	at.testResources(none, controller, scheduler, endpointController, "get", "update", sysEndpoints)
	at.testResources(request, controller, scheduler, endpointController, "get", endpoints)
	at.testResources(response, controller, scheduler, endpointController, "update", endpoints)

	at.testResources(none, apiserver, "get", namespaces, namespaceStatus, namespaceFinal)
	at.testResources(metadata, apiserver, "get", "create", "update", sysConfigmaps, secrets)

	at.testResources(none, autoscaler, "get", "update", sysConfigmaps, sysEndpoints)
	at.testResources(metadata, autoscaler, "get", "update", configmaps)
	at.testResources(response, autoscaler, "update", endpoints)

	at.testResources(none, controller, "get", "list", podMetrics, nodeMetrics)

	at.testNonResources(none, allUsers, "/healthz", "/healthz/etcd", "/swagger-2.0.0.json", "/swagger-2.0.0.pb-v1.gz", "/version")
	at.testNonResources(metadata, allUsers, "/logs", "/openapi/v2", "/apis/policy", "/metrics", "/api")

	at.testResources(none, node, apiserver, defaultSA, anonymous, "get", "list", "create", "patch", "update", "delete", events)

	at.testResources(request, kubelet, node, npd, npdSA, "update", "patch", nodeStatus, podStatus)

	at.testResources(request, namespaceController, "deletecollection", pods, namespaces)

	at.testResources(metadata, defaultSA, anonymous, npd, namespaceController, "get", "create", "update", secrets, configmaps, sysConfigmaps, tokenReviews)
	at.testResources(request, defaultSA, anonymous, npd, namespaceController, "get", "list", "watch", sysEndpoints, podMetrics, pods, clusterRoles, deployments)
	at.testResources(response, defaultSA, anonymous, npd, namespaceController, "create", "update", "patch", "delete", sysEndpoints, podMetrics, pods, clusterRoles, deployments)

	at.testResources(metadata, defaultSA, anonymous, npd, namespaceController, "get", "list", "watch", "create", "update", "patch", "delete", foobars, foobarbaz)
}

type auditTester struct {
	*testing.T
	checker auditpolicy.Checker
}

func (t *auditTester) testResources(level audit.Level, usrVerbRes ...interface{}) {
	verbs := []string{}
	users := []user.Info{}
	resources := []Resource{}
	for _, arg := range usrVerbRes {
		switch v := arg.(type) {
		case string:
			verbs = append(verbs, v)
		case user.Info:
			users = append(users, v)
		case Resource:
			resources = append(resources, v)
		default:
			t.Fatalf("Invalid test argument: %+v", arg)
		}
	}
	require.NotEmpty(t, verbs, "testcases must have a verb")
	require.NotEmpty(t, users, "testcases must have a user")
	require.NotEmpty(t, resources, "resource testcases must have a resource")

	for _, usr := range users {
		for _, verb := range verbs {
			for _, res := range resources {
				attrs := &authorizer.AttributesRecord{
					User:            usr,
					Verb:            verb,
					Namespace:       res.Namespace,
					APIGroup:        res.Group,
					APIVersion:      "v1",
					Resource:        res.Resource,
					Subresource:     res.Subresource,
					ResourceRequest: true,
				}
				t.expectLevel(level, attrs)
			}
		}
	}
}

func (t *auditTester) testNonResources(level audit.Level, users []user.Info, paths ...string) {
	for _, usr := range users {
		for _, verb := range []string{"get", "post"} {
			for _, path := range paths {
				attrs := &authorizer.AttributesRecord{
					User:            usr,
					Verb:            verb,
					ResourceRequest: false,
					Path:            path,
				}
				t.expectLevel(level, attrs)
			}
		}
	}
}

func (t *auditTester) expectLevel(expected audit.Level, attrs authorizer.Attributes) {
	obj := attrs.GetPath()
	if attrs.IsResourceRequest() {
		obj = attrs.GetResource()
		if attrs.GetNamespace() != "" {
			obj = obj + ":" + attrs.GetNamespace()
		}
	}
	name := fmt.Sprintf("%s.%s.%s", attrs.GetUser().GetName(), attrs.GetVerb(), obj)
	checker := t.checker
	t.Run(name, func(t *testing.T) {
		level, stages := checker.LevelAndStages(attrs)
		assert.Equal(t, expected, level)
		if level != audit.LevelNone {
			assert.ElementsMatch(t, stages, []audit.Stage{audit.StageRequestReceived})
		}
	})
}

func newUserInfo(name string, groups ...string) user.Info {
	return &user.DefaultInfo{
		Name:   name,
		Groups: groups,
	}
}

type Resource struct {
	Group, Resource, Subresource, Namespace string
}

func resource(kind string, nsGroupSub ...string) Resource {
	res := Resource{Resource: kind}
	if len(nsGroupSub) > 0 {
		res.Namespace = nsGroupSub[0]
	}
	if len(nsGroupSub) > 1 {
		res.Group = nsGroupSub[1]
	}
	if len(nsGroupSub) > 2 {
		res.Subresource = nsGroupSub[2]
	}
	return res
}
