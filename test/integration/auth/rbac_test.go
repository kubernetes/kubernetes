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

package auth

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	gopath "path"
	"reflect"
	"strings"
	"testing"

	rbacapi "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/authentication/group"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	unionauthn "k8s.io/apiserver/pkg/authentication/request/union"
	"k8s.io/apiserver/pkg/authentication/token/tokenfile"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	unionauthz "k8s.io/apiserver/pkg/authorization/union"
	"k8s.io/apiserver/pkg/registry/generic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
	clusterrolestore "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding"
	clusterrolebindingstore "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/role"
	rolestore "k8s.io/kubernetes/pkg/registry/rbac/role/storage"
	"k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
	rolebindingstore "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/storage"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func clientForToken(user string, rt http.RoundTripper) *http.Client {
	return &http.Client{
		Transport: transport.NewBearerAuthRoundTripper(
			user,
			transport.DebugWrappers(rt),
		),
	}
}

func clientsetForToken(user string, config *restclient.Config) (clientset.Interface, clientset.Interface) {
	configCopy := *config
	configCopy.BearerToken = user
	return clientset.NewForConfigOrDie(&configCopy), clientset.NewForConfigOrDie(&configCopy)
}

type testRESTOptionsGetter struct {
	config *controlplane.Config
}

func (getter *testRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource, example runtime.Object) (generic.RESTOptions, error) {
	storageConfig, err := getter.config.ControlPlane.Extra.StorageFactory.NewConfig(resource, example)
	if err != nil {
		return generic.RESTOptions{}, fmt.Errorf("failed to get storage: %v", err)
	}
	return generic.RESTOptions{StorageConfig: storageConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: resource.Resource}, nil
}

func newRBACAuthorizer(t *testing.T, config *controlplane.Config) (authorizer.Authorizer, func()) {
	optsGetter := &testRESTOptionsGetter{config}
	roleRest, err := rolestore.NewREST(optsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	roleRegistry := role.AuthorizerAdapter{Registry: role.NewRegistry(roleRest)}
	rolebindingRest, err := rolebindingstore.NewREST(optsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	roleBindingRegistry := rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(rolebindingRest)}
	clusterroleRest, err := clusterrolestore.NewREST(optsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	clusterRoleRegistry := clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterroleRest)}
	clusterrolebindingRest, err := clusterrolebindingstore.NewREST(optsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	clusterRoleBindingRegistry := clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterrolebindingRest)}

	tearDownFn := func() {
		roleRest.Destroy()
		rolebindingRest.Destroy()
		clusterroleRest.Destroy()
		clusterrolebindingRest.Destroy()
	}
	return rbac.New(roleRegistry, roleBindingRegistry, clusterRoleRegistry, clusterRoleBindingRegistry), tearDownFn
}

// bootstrapRoles are a set of RBAC roles which will be populated before the test.
type bootstrapRoles struct {
	roles               []rbacapi.Role
	roleBindings        []rbacapi.RoleBinding
	clusterRoles        []rbacapi.ClusterRole
	clusterRoleBindings []rbacapi.ClusterRoleBinding
}

// bootstrap uses the provided client to create the bootstrap roles and role bindings.
//
// client should be authenticated as the RBAC super user.
func (b bootstrapRoles) bootstrap(client clientset.Interface) error {
	for _, r := range b.clusterRoles {
		_, err := client.RbacV1().ClusterRoles().Create(context.TODO(), &r, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.roles {
		_, err := client.RbacV1().Roles(r.Namespace).Create(context.TODO(), &r, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.clusterRoleBindings {
		_, err := client.RbacV1().ClusterRoleBindings().Create(context.TODO(), &r, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.roleBindings {
		_, err := client.RbacV1().RoleBindings(r.Namespace).Create(context.TODO(), &r, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}

	return nil
}

// request is a test case which can.
type request struct {
	// The bearer token sent as part of the request
	token string

	// Resource metadata
	verb      string
	apiGroup  string
	resource  string
	namespace string
	name      string

	// The actual resource.
	body string

	// The expected return status of this request.
	expectedStatus int
}

func (r request) String() string {
	return fmt.Sprintf("%s %s %s", r.token, r.verb, r.resource)
}

type statusCode int

func (s statusCode) String() string {
	return fmt.Sprintf("%d %s", int(s), http.StatusText(int(s)))
}

// Declare a set of raw objects to use.
var (
	writeJobsRoleBinding = `
{
  "apiVersion": "rbac.authorization.k8s.io/v1",
  "kind": "RoleBinding",
  "metadata": {
    "name": "pi"%s
  },
  "roleRef": {
    "apiGroup": "rbac.authorization.k8s.io",
    "kind": "ClusterRole",
    "name": "write-jobs"
  },
  "subjects": [{
    "apiGroup": "rbac.authorization.k8s.io",
    "kind": "User",
    "name": "admin"
  }]
}`

	aJob = `
{
  "apiVersion": "batch/v1",
  "kind": "Job",
  "metadata": {
    "name": "pi"%s
  },
  "spec": {
    "template": {
      "metadata": {
        "name": "a",
        "labels": {
          "name": "pijob"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "pi",
            "image": "perl",
            "command": [
              "perl",
              "-Mbignum=bpi",
              "-wle",
              "print bpi(2000)"
            ]
          }
        ],
        "restartPolicy": "Never"
      }
    }
  }
}
`
	aLimitRange = `
{
  "apiVersion": "v1",
  "kind": "LimitRange",
  "metadata": {
    "name": "a"%s
  }
}
`
	podNamespace = `
{
  "apiVersion": "v1",
  "kind": "Namespace",
  "metadata": {
	"name": "pod-namespace"%s
  }
}
`
	jobNamespace = `
{
  "apiVersion": "v1",
  "kind": "Namespace",
  "metadata": {
	"name": "job-namespace"%s
  }
}
`
	forbiddenNamespace = `
{
  "apiVersion": "v1",
  "kind": "Namespace",
  "metadata": {
	"name": "forbidden-namespace"%s
  }
}
`
	limitRangeNamespace = `
{
  "apiVersion": "v1",
  "kind": "Namespace",
  "metadata": {
	"name": "limitrange-namespace"%s
  }
}
`
)

// Declare some PolicyRules beforehand.
var (
	ruleAllowAll  = rbachelper.NewRule("*").Groups("*").Resources("*").RuleOrDie()
	ruleReadPods  = rbachelper.NewRule("list", "get", "watch").Groups("").Resources("pods").RuleOrDie()
	ruleWriteJobs = rbachelper.NewRule("*").Groups("batch").Resources("*").RuleOrDie()
)

func TestRBAC(t *testing.T) {
	superUser := "admin/system:masters"

	tests := []struct {
		bootstrapRoles bootstrapRoles

		requests []request
	}{
		{
			bootstrapRoles: bootstrapRoles{
				clusterRoles: []rbacapi.ClusterRole{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "allow-all"},
						Rules:      []rbacapi.PolicyRule{ruleAllowAll},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "read-pods"},
						Rules:      []rbacapi.PolicyRule{ruleReadPods},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "read-pods"},
						Subjects: []rbacapi.Subject{
							{Kind: "User", Name: "pod-reader"},
						},
						RoleRef: rbacapi.RoleRef{Kind: "ClusterRole", Name: "read-pods"},
					},
				},
			},
			requests: []request{
				// Create the namespace used later in the test
				{superUser, "POST", "", "namespaces", "", "", podNamespace, http.StatusCreated},

				{superUser, "GET", "", "pods", "", "", "", http.StatusOK},
				{superUser, "GET", "", "pods", "pod-namespace", "a", "", http.StatusNotFound},
				{superUser, "POST", "", "pods", "pod-namespace", "", aPod, http.StatusCreated},
				{superUser, "GET", "", "pods", "pod-namespace", "a", "", http.StatusOK},

				{"bob", "GET", "", "pods", "", "", "", http.StatusForbidden},
				{"bob", "GET", "", "pods", "pod-namespace", "a", "", http.StatusForbidden},

				{"pod-reader", "GET", "", "pods", "", "", "", http.StatusOK},
				{"pod-reader", "POST", "", "pods", "pod-namespace", "", aPod, http.StatusForbidden},
			},
		},
		{
			bootstrapRoles: bootstrapRoles{
				clusterRoles: []rbacapi.ClusterRole{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "write-jobs"},
						Rules:      []rbacapi.PolicyRule{ruleWriteJobs},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "create-rolebindings"},
						Rules: []rbacapi.PolicyRule{
							rbachelper.NewRule("create").Groups("rbac.authorization.k8s.io").Resources("rolebindings").RuleOrDie(),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "bind-any-clusterrole"},
						Rules: []rbacapi.PolicyRule{
							rbachelper.NewRule("bind").Groups("rbac.authorization.k8s.io").Resources("clusterroles").RuleOrDie(),
						},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "write-jobs"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "job-writer"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "write-jobs"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "create-rolebindings"},
						Subjects: []rbacapi.Subject{
							{Kind: "User", Name: "job-writer"},
							{Kind: "User", Name: "nonescalating-rolebinding-writer"},
							{Kind: "User", Name: "any-rolebinding-writer"},
						},
						RoleRef: rbacapi.RoleRef{Kind: "ClusterRole", Name: "create-rolebindings"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "bind-any-clusterrole"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "any-rolebinding-writer"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "bind-any-clusterrole"},
					},
				},
				roleBindings: []rbacapi.RoleBinding{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "write-jobs", Namespace: "job-namespace"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "job-writer-namespace"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "write-jobs"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "create-rolebindings", Namespace: "job-namespace"},
						Subjects: []rbacapi.Subject{
							{Kind: "User", Name: "job-writer-namespace"},
							{Kind: "User", Name: "any-rolebinding-writer-namespace"},
						},
						RoleRef: rbacapi.RoleRef{Kind: "ClusterRole", Name: "create-rolebindings"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "bind-any-clusterrole", Namespace: "job-namespace"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "any-rolebinding-writer-namespace"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "bind-any-clusterrole"},
					},
				},
			},
			requests: []request{
				// Create the namespace used later in the test
				{superUser, "POST", "", "namespaces", "", "", jobNamespace, http.StatusCreated},
				{superUser, "POST", "", "namespaces", "", "", forbiddenNamespace, http.StatusCreated},

				{"user-with-no-permissions", "POST", "batch", "jobs", "job-namespace", "", aJob, http.StatusForbidden},
				{"user-with-no-permissions", "GET", "batch", "jobs", "job-namespace", "pi", "", http.StatusForbidden},

				// job-writer-namespace cannot write to the "forbidden-namespace"
				{"job-writer-namespace", "GET", "batch", "jobs", "forbidden-namespace", "", "", http.StatusForbidden},
				{"job-writer-namespace", "GET", "batch", "jobs", "forbidden-namespace", "pi", "", http.StatusForbidden},
				{"job-writer-namespace", "POST", "batch", "jobs", "forbidden-namespace", "", aJob, http.StatusForbidden},
				{"job-writer-namespace", "GET", "batch", "jobs", "forbidden-namespace", "pi", "", http.StatusForbidden},

				// job-writer can write to any namespace
				{"job-writer", "GET", "batch", "jobs", "forbidden-namespace", "", "", http.StatusOK},
				{"job-writer", "GET", "batch", "jobs", "forbidden-namespace", "pi", "", http.StatusNotFound},
				{"job-writer", "POST", "batch", "jobs", "forbidden-namespace", "", aJob, http.StatusCreated},
				{"job-writer", "GET", "batch", "jobs", "forbidden-namespace", "pi", "", http.StatusOK},

				{"job-writer-namespace", "GET", "batch", "jobs", "job-namespace", "", "", http.StatusOK},
				{"job-writer-namespace", "GET", "batch", "jobs", "job-namespace", "pi", "", http.StatusNotFound},
				{"job-writer-namespace", "POST", "batch", "jobs", "job-namespace", "", aJob, http.StatusCreated},
				{"job-writer-namespace", "GET", "batch", "jobs", "job-namespace", "pi", "", http.StatusOK},

				// cannot bind role anywhere
				{"user-with-no-permissions", "POST", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "", writeJobsRoleBinding, http.StatusForbidden},
				// can only bind role in namespace where they have explicit bind permission
				{"any-rolebinding-writer-namespace", "POST", "rbac.authorization.k8s.io", "rolebindings", "forbidden-namespace", "", writeJobsRoleBinding, http.StatusForbidden},
				// can only bind role in namespace where they have covering permissions
				{"job-writer-namespace", "POST", "rbac.authorization.k8s.io", "rolebindings", "forbidden-namespace", "", writeJobsRoleBinding, http.StatusForbidden},
				{"job-writer-namespace", "POST", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "", writeJobsRoleBinding, http.StatusCreated},
				{superUser, "DELETE", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "pi", "", http.StatusOK},
				// can bind role in any namespace where they have covering permissions
				{"job-writer", "POST", "rbac.authorization.k8s.io", "rolebindings", "forbidden-namespace", "", writeJobsRoleBinding, http.StatusCreated},
				{superUser, "DELETE", "rbac.authorization.k8s.io", "rolebindings", "forbidden-namespace", "pi", "", http.StatusOK},
				// cannot bind role because they don't have covering permissions
				{"nonescalating-rolebinding-writer", "POST", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "", writeJobsRoleBinding, http.StatusForbidden},
				// can bind role because they have explicit bind permission
				{"any-rolebinding-writer", "POST", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "", writeJobsRoleBinding, http.StatusCreated},
				{superUser, "DELETE", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "pi", "", http.StatusOK},
				{"any-rolebinding-writer-namespace", "POST", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "", writeJobsRoleBinding, http.StatusCreated},
				{superUser, "DELETE", "rbac.authorization.k8s.io", "rolebindings", "job-namespace", "pi", "", http.StatusOK},
			},
		},
		{
			bootstrapRoles: bootstrapRoles{
				clusterRoles: []rbacapi.ClusterRole{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "allow-all"},
						Rules:      []rbacapi.PolicyRule{ruleAllowAll},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "update-limitranges"},
						Rules: []rbacapi.PolicyRule{
							rbachelper.NewRule("update").Groups("").Resources("limitranges").RuleOrDie(),
						},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "update-limitranges"},
						Subjects: []rbacapi.Subject{
							{Kind: "User", Name: "limitrange-updater"},
						},
						RoleRef: rbacapi.RoleRef{Kind: "ClusterRole", Name: "update-limitranges"},
					},
				},
			},
			requests: []request{
				// Create the namespace used later in the test
				{superUser, "POST", "", "namespaces", "", "", limitRangeNamespace, http.StatusCreated},

				{"limitrange-updater", "PUT", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusForbidden},
				{superUser, "PUT", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusCreated},
				{superUser, "PUT", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusOK},
				{"limitrange-updater", "PUT", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusOK},
			},
		},
		{
			bootstrapRoles: bootstrapRoles{
				clusterRoles: []rbacapi.ClusterRole{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "allow-all"},
						Rules:      []rbacapi.PolicyRule{ruleAllowAll},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "patch-limitranges"},
						Rules: []rbacapi.PolicyRule{
							rbachelper.NewRule("patch").Groups("").Resources("limitranges").RuleOrDie(),
						},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "patch-limitranges"},
						Subjects: []rbacapi.Subject{
							{Kind: "User", Name: "limitrange-patcher"},
						},
						RoleRef: rbacapi.RoleRef{Kind: "ClusterRole", Name: "patch-limitranges"},
					},
				},
			},
			requests: []request{
				// Create the namespace used later in the test
				{superUser, "POST", "", "namespaces", "", "", limitRangeNamespace, http.StatusCreated},

				{"limitrange-patcher", "PATCH", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusForbidden},
				{superUser, "PATCH", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusCreated},
				{superUser, "PATCH", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusOK},
				{"limitrange-patcher", "PATCH", "", "limitranges", "limitrange-namespace", "a", aLimitRange, http.StatusOK},
			},
		},
	}

	for i, tc := range tests {
		t.Run(fmt.Sprintf("case-%d", i), func(t *testing.T) {
			authenticator := group.NewAuthenticatedGroupAdder(bearertoken.New(tokenfile.New(map[string]*user.DefaultInfo{
				superUser:                          {Name: "admin", Groups: []string{"system:masters"}},
				"any-rolebinding-writer":           {Name: "any-rolebinding-writer"},
				"any-rolebinding-writer-namespace": {Name: "any-rolebinding-writer-namespace"},
				"bob":                              {Name: "bob"},
				"job-writer":                       {Name: "job-writer"},
				"job-writer-namespace":             {Name: "job-writer-namespace"},
				"nonescalating-rolebinding-writer": {Name: "nonescalating-rolebinding-writer"},
				"pod-reader":                       {Name: "pod-reader"},
				"limitrange-updater":               {Name: "limitrange-updater"},
				"limitrange-patcher":               {Name: "limitrange-patcher"},
				"user-with-no-permissions":         {Name: "user-with-no-permissions"},
			})))

			tCtx := ktesting.Init(t)
			var tearDownAuthorizerFn func()
			defer func() {
				if tearDownAuthorizerFn != nil {
					tearDownAuthorizerFn()
				}
			}()

			_, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
					// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
					// Also disable namespace lifecycle to workaroung the test limitation that first creates
					// roles/rolebindings and only then creates corresponding namespaces.
					opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "NamespaceLifecycle"}
					// Disable built-in authorizers
					opts.Authorization.Modes = []string{"AlwaysDeny"}
				},
				ModifyServerConfig: func(config *controlplane.Config) {
					// Append our custom test authenticator
					config.ControlPlane.Generic.Authentication.Authenticator = unionauthn.New(config.ControlPlane.Generic.Authentication.Authenticator, authenticator)
					// Append our custom test authorizer
					var rbacAuthz authorizer.Authorizer
					rbacAuthz, tearDownAuthorizerFn = newRBACAuthorizer(t, config)
					config.ControlPlane.Generic.Authorization.Authorizer = unionauthz.New(config.ControlPlane.Generic.Authorization.Authorizer, rbacAuthz)
				},
			})
			defer tearDownFn()

			transport, err := restclient.TransportFor(kubeConfig)
			if err != nil {
				t.Fatal(err)
			}

			// Bootstrap the API Server with the test case's initial roles.
			superuserClient, _ := clientsetForToken(superUser, kubeConfig)
			if err := tc.bootstrapRoles.bootstrap(superuserClient); err != nil {
				t.Errorf("case %d: failed to apply initial roles: %v", i, err)
				return
			}
			previousResourceVersion := make(map[string]float64)

			for j, r := range tc.requests {
				// This is a URL-path, not a local path, so we use the "path"
				// package (aliased as "gopath") instead of "path/filepath".
				urlPath := "/"
				if r.apiGroup == "" {
					urlPath = gopath.Join(urlPath, "api/v1")
				} else {
					urlPath = gopath.Join(urlPath, "apis", r.apiGroup, "v1")
				}
				if r.namespace != "" {
					urlPath = gopath.Join(urlPath, "namespaces", r.namespace)
				}
				if r.resource != "" {
					urlPath = gopath.Join(urlPath, r.resource)
				}
				if r.name != "" {
					urlPath = gopath.Join(urlPath, r.name)
				}

				var body io.Reader
				if r.body != "" {
					sub := ""
					if r.verb == "PUT" {
						// For update operations, insert previous resource version
						if resVersion := previousResourceVersion[getPreviousResourceVersionKey(urlPath, "")]; resVersion != 0 {
							sub += fmt.Sprintf(",\"resourceVersion\": \"%v\"", resVersion)
						}
					}
					body = strings.NewReader(fmt.Sprintf(r.body, sub))
				}

				req, err := http.NewRequest(r.verb, kubeConfig.Host+urlPath, body)
				if r.verb == "PATCH" {
					// For patch operations, use the apply content type
					req.Header.Add("Content-Type", string(types.ApplyPatchType))
					q := req.URL.Query()
					q.Add("fieldManager", "rbac_test")
					req.URL.RawQuery = q.Encode()
				}

				if err != nil {
					t.Fatalf("failed to create request: %v", err)
				}

				func() {
					reqDump, err := httputil.DumpRequest(req, true)
					if err != nil {
						t.Fatalf("failed to dump request: %v", err)
						return
					}

					resp, err := clientForToken(r.token, transport).Do(req)
					if err != nil {
						t.Errorf("case %d, req %d: failed to make request: %v", i, j, err)
						return
					}
					defer resp.Body.Close()

					respDump, err := httputil.DumpResponse(resp, true)
					if err != nil {
						t.Fatalf("failed to dump response: %v", err)
						return
					}

					if resp.StatusCode != r.expectedStatus {
						// When debugging is on, dump the entire request and response. Very helpful for
						// debugging malformed test cases.
						//
						// To turn on debugging, use the '-args' flag.
						//
						//    go test -v -tags integration -run RBAC -args -v 10
						//
						klog.V(8).Infof("case %d, req %d: %s\n%s\n", i, j, reqDump, respDump)
						t.Errorf("case %d, req %d: %s expected %q got %q", i, j, r, statusCode(r.expectedStatus), statusCode(resp.StatusCode))
					}

					b, _ := io.ReadAll(resp.Body)

					if r.verb == "POST" && (resp.StatusCode/100) == 2 {
						// For successful create operations, extract resourceVersion
						id, currentResourceVersion, err := parseResourceVersion(b)
						if err == nil {
							key := getPreviousResourceVersionKey(urlPath, id)
							previousResourceVersion[key] = currentResourceVersion
						} else {
							t.Logf("error in trying to extract resource version: %s", err)
						}
					}
				}()
			}
		})
	}
}

func TestBootstrapping(t *testing.T) {
	tCtx := ktesting.Init(t)
	clientset, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Authorization.Modes = []string{"RBAC"}
		},
	})
	defer tearDownFn()

	watcher, err := clientset.RbacV1().ClusterRoles().Watch(tCtx, metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = watchtools.UntilWithoutRetry(tCtx, watcher, func(event watch.Event) (bool, error) {
		if event.Type != watch.Added {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	clusterRoles, err := clientset.RbacV1().ClusterRoles().List(tCtx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(clusterRoles.Items) == 0 {
		t.Fatalf("missing cluster roles")
	}

	for _, clusterRole := range clusterRoles.Items {
		if clusterRole.Name == "cluster-admin" {
			return
		}
	}

	t.Errorf("missing cluster-admin: %v", clusterRoles)

	healthBytes, err := clientset.Discovery().RESTClient().Get().AbsPath("/healthz/poststarthook/rbac/bootstrap-roles").DoRaw(tCtx)
	if err != nil {
		t.Error(err)
	}
	t.Errorf("error bootstrapping roles: %s", string(healthBytes))
}

// TestDiscoveryUpgradeBootstrapping is primarily meant to test the behavior of
// primePublicInfoClusterRoleBinding in storage_rbac.go during cluster upgrades.
func TestDiscoveryUpgradeBootstrapping(t *testing.T) {
	var tearDownFn func()
	defer func() {
		if tearDownFn != nil {
			tearDownFn()
		}
	}()

	etcdConfig := framework.SharedEtcd()

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Ensure we're using the same etcd across apiserver restarts.
			opts.Etcd.StorageConfig = *etcdConfig
			opts.Authorization.Modes = []string{"RBAC"}
		},
	})

	// Modify the default RBAC discovery ClusterRoleBidnings to look more like the defaults that
	// existed prior to v1.14, but with user modifications.
	t.Logf("Modifying default `system:discovery` ClusterRoleBinding")
	discRoleBinding, err := client.RbacV1().ClusterRoleBindings().Get(tCtx, "system:discovery", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get `system:discovery` ClusterRoleBinding: %v", err)
	}
	discRoleBinding.Annotations["rbac.authorization.kubernetes.io/autoupdate"] = "false"
	discRoleBinding.Annotations["rbac-discovery-upgrade-test"] = "pass"
	discRoleBinding.Subjects = []rbacapi.Subject{
		{
			Name:     "system:authenticated",
			Kind:     "Group",
			APIGroup: "rbac.authorization.k8s.io",
		},
	}
	if discRoleBinding, err = client.RbacV1().ClusterRoleBindings().Update(tCtx, discRoleBinding, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update `system:discovery` ClusterRoleBinding: %v", err)
	}
	t.Logf("Modifying default `system:basic-user` ClusterRoleBinding")
	basicUserRoleBinding, err := client.RbacV1().ClusterRoleBindings().Get(tCtx, "system:basic-user", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get `system:basic-user` ClusterRoleBinding: %v", err)
	}
	basicUserRoleBinding.Annotations["rbac.authorization.kubernetes.io/autoupdate"] = "false"
	basicUserRoleBinding.Annotations["rbac-discovery-upgrade-test"] = "pass"
	if basicUserRoleBinding, err = client.RbacV1().ClusterRoleBindings().Update(tCtx, basicUserRoleBinding, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update `system:basic-user` ClusterRoleBinding: %v", err)
	}
	t.Logf("Deleting default `system:public-info-viewer` ClusterRoleBinding")
	if err = client.RbacV1().ClusterRoleBindings().Delete(tCtx, "system:public-info-viewer", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Failed to delete `system:public-info-viewer` ClusterRoleBinding: %v", err)
	}

	// Stop the first API server.
	tearDownFn()
	tearDownFn = nil

	// Check that upgraded API servers inherit `system:public-info-viewer` settings from
	// `system:discovery`, and respect auto-reconciliation annotations.
	client, _, tearDownFn = framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Ensure we're using the same etcd across apiserver restarts.
			opts.Etcd.StorageConfig = *etcdConfig
			opts.Authorization.Modes = []string{"RBAC"}
		},
	})

	newDiscRoleBinding, err := client.RbacV1().ClusterRoleBindings().Get(tCtx, "system:discovery", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get `system:discovery` ClusterRoleBinding: %v", err)
	}
	if !reflect.DeepEqual(newDiscRoleBinding, discRoleBinding) {
		t.Errorf("`system:discovery` should have been unmodified. Wanted: %v, got %v", discRoleBinding, newDiscRoleBinding)
	}
	newBasicUserRoleBinding, err := client.RbacV1().ClusterRoleBindings().Get(tCtx, "system:basic-user", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get `system:basic-user` ClusterRoleBinding: %v", err)
	}
	if !reflect.DeepEqual(newBasicUserRoleBinding, basicUserRoleBinding) {
		t.Errorf("`system:basic-user` should have been unmodified. Wanted: %v, got %v", basicUserRoleBinding, newBasicUserRoleBinding)
	}
	publicInfoViewerRoleBinding, err := client.RbacV1().ClusterRoleBindings().Get(tCtx, "system:public-info-viewer", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get `system:public-info-viewer` ClusterRoleBinding: %v", err)
	}
	if publicInfoViewerRoleBinding.Annotations["rbac.authorization.kubernetes.io/autoupdate"] != "false" {
		t.Errorf("publicInfoViewerRoleBinding.Annotations[\"rbac.authorization.kubernetes.io/autoupdate\"] should be %v, got %v", publicInfoViewerRoleBinding.Annotations["rbac.authorization.kubernetes.io/autoupdate"], "false")
	}
	if publicInfoViewerRoleBinding.Annotations["rbac-discovery-upgrade-test"] != "pass" {
		t.Errorf("publicInfoViewerRoleBinding.Annotations[\"rbac-discovery-upgrade-test\"] should be %v, got %v", publicInfoViewerRoleBinding.Annotations["rbac-discovery-upgrade-test"], "pass")
	}
	if !reflect.DeepEqual(publicInfoViewerRoleBinding.Subjects, newDiscRoleBinding.Subjects) {
		t.Errorf("`system:public-info-viewer` should have inherited Subjects from `system:discovery` Wanted: %v, got %v", newDiscRoleBinding.Subjects, publicInfoViewerRoleBinding.Subjects)
	}
}
