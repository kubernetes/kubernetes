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
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"strings"
	"testing"
	"time"

	"k8s.io/klog"

	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	"k8s.io/apiserver/pkg/authentication/token/tokenfile"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	externalclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/transport"
	csiclientset "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	api "k8s.io/kubernetes/pkg/apis/core"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/master"
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
)

func clientForToken(user string) *http.Client {
	return &http.Client{
		Transport: transport.NewBearerAuthRoundTripper(
			user,
			transport.DebugWrappers(http.DefaultTransport),
		),
	}
}

func clientsetForToken(user string, config *restclient.Config) (clientset.Interface, externalclientset.Interface) {
	configCopy := *config
	configCopy.BearerToken = user
	return clientset.NewForConfigOrDie(&configCopy), externalclientset.NewForConfigOrDie(&configCopy)
}

func crdClientsetForToken(user string, config *restclient.Config) apiextensionsclient.Interface {
	configCopy := *config
	configCopy.BearerToken = user
	return apiextensionsclient.NewForConfigOrDie(&configCopy)
}

func csiClientsetForToken(user string, config *restclient.Config) csiclientset.Interface {
	configCopy := *config
	configCopy.BearerToken = user
	configCopy.ContentType = "application/json" // // csi client works with CRDs that support json only
	return csiclientset.NewForConfigOrDie(&configCopy)
}

type testRESTOptionsGetter struct {
	config *master.Config
}

func (getter *testRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	storageConfig, err := getter.config.ExtraConfig.StorageFactory.NewConfig(resource)
	if err != nil {
		return generic.RESTOptions{}, fmt.Errorf("failed to get storage: %v", err)
	}
	return generic.RESTOptions{StorageConfig: storageConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: resource.Resource}, nil
}

func newRBACAuthorizer(config *master.Config) authorizer.Authorizer {
	optsGetter := &testRESTOptionsGetter{config}
	roleRegistry := role.AuthorizerAdapter{Registry: role.NewRegistry(rolestore.NewREST(optsGetter))}
	roleBindingRegistry := rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(rolebindingstore.NewREST(optsGetter))}
	clusterRoleRegistry := clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterrolestore.NewREST(optsGetter))}
	clusterRoleBindingRegistry := clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterrolebindingstore.NewREST(optsGetter))}
	return rbac.New(roleRegistry, roleBindingRegistry, clusterRoleRegistry, clusterRoleBindingRegistry)
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
		_, err := client.Rbac().ClusterRoles().Create(&r)
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.roles {
		_, err := client.Rbac().Roles(r.Namespace).Create(&r)
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.clusterRoleBindings {
		_, err := client.Rbac().ClusterRoleBindings().Create(&r)
		if err != nil {
			return fmt.Errorf("failed to make request: %v", err)
		}
	}
	for _, r := range b.roleBindings {
		_, err := client.Rbac().RoleBindings(r.Namespace).Create(&r)
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
	// Make a role binding with the version enabled in testapi.Rbac
	// This assumes testapi is using rbac.authorization.k8s.io/v1beta1 or rbac.authorization.k8s.io/v1, which are identical in structure.
	// TODO: rework or remove testapi usage to allow writing integration tests that don't depend on envvars
	writeJobsRoleBinding = `
{
  "apiVersion": "` + testapi.Rbac.GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "pod-namespace"%s
  }
}
`
	jobNamespace = `
{
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "job-namespace"%s
  }
}
`
	forbiddenNamespace = `
{
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "forbidden-namespace"%s
  }
}
`
	limitRangeNamespace = `
{
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "limitrange-namespace"%s
  }
}
`
)

// Declare some PolicyRules beforehand.
var (
	ruleAllowAll  = rbacapi.NewRule("*").Groups("*").Resources("*").RuleOrDie()
	ruleReadPods  = rbacapi.NewRule("list", "get", "watch").Groups("").Resources("pods").RuleOrDie()
	ruleWriteJobs = rbacapi.NewRule("*").Groups("batch").Resources("*").RuleOrDie()
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
							rbacapi.NewRule("create").Groups("rbac.authorization.k8s.io").Resources("rolebindings").RuleOrDie(),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "bind-any-clusterrole"},
						Rules: []rbacapi.PolicyRule{
							rbacapi.NewRule("bind").Groups("rbac.authorization.k8s.io").Resources("clusterroles").RuleOrDie(),
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
							rbacapi.NewRule("update").Groups("").Resources("limitranges").RuleOrDie(),
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
	}

	for i, tc := range tests {
		// Create an API Server.
		masterConfig := framework.NewIntegrationTestMasterConfig()
		masterConfig.GenericConfig.Authorization.Authorizer = newRBACAuthorizer(masterConfig)
		masterConfig.GenericConfig.Authentication.Authenticator = bearertoken.New(tokenfile.New(map[string]*user.DefaultInfo{
			superUser:                          {Name: "admin", Groups: []string{"system:masters"}},
			"any-rolebinding-writer":           {Name: "any-rolebinding-writer"},
			"any-rolebinding-writer-namespace": {Name: "any-rolebinding-writer-namespace"},
			"bob":                              {Name: "bob"},
			"job-writer":                       {Name: "job-writer"},
			"job-writer-namespace":             {Name: "job-writer-namespace"},
			"nonescalating-rolebinding-writer": {Name: "nonescalating-rolebinding-writer"},
			"pod-reader":                       {Name: "pod-reader"},
			"limitrange-updater":               {Name: "limitrange-updater"},
			"user-with-no-permissions":         {Name: "user-with-no-permissions"},
		}))
		_, s, closeFn := framework.RunAMaster(masterConfig)
		defer closeFn()

		clientConfig := &restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}

		// Bootstrap the API Server with the test case's initial roles.
		superuserClient, _ := clientsetForToken(superUser, clientConfig)
		if err := tc.bootstrapRoles.bootstrap(superuserClient); err != nil {
			t.Errorf("case %d: failed to apply initial roles: %v", i, err)
			continue
		}
		previousResourceVersion := make(map[string]float64)

		for j, r := range tc.requests {
			testGroup, ok := testapi.Groups[r.apiGroup]
			if !ok {
				t.Errorf("case %d %d: unknown api group %q, %s", i, j, r.apiGroup, r)
				continue
			}
			path := testGroup.ResourcePath(r.resource, r.namespace, r.name)

			var body io.Reader
			if r.body != "" {
				sub := ""
				if r.verb == "PUT" {
					// For update operations, insert previous resource version
					if resVersion := previousResourceVersion[getPreviousResourceVersionKey(path, "")]; resVersion != 0 {
						sub += fmt.Sprintf(",\"resourceVersion\": \"%v\"", resVersion)
					}
				}
				body = strings.NewReader(fmt.Sprintf(r.body, sub))
			}

			req, err := http.NewRequest(r.verb, s.URL+path, body)
			if err != nil {
				t.Fatalf("failed to create request: %v", err)
			}

			func() {
				reqDump, err := httputil.DumpRequest(req, true)
				if err != nil {
					t.Fatalf("failed to dump request: %v", err)
					return
				}

				resp, err := clientForToken(r.token).Do(req)
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

				b, _ := ioutil.ReadAll(resp.Body)

				if r.verb == "POST" && (resp.StatusCode/100) == 2 {
					// For successful create operations, extract resourceVersion
					id, currentResourceVersion, err := parseResourceVersion(b)
					if err == nil {
						key := getPreviousResourceVersionKey(path, id)
						previousResourceVersion[key] = currentResourceVersion
					} else {
						t.Logf("error in trying to extract resource version: %s", err)
					}
				}
			}()
		}
	}
}

func TestBootstrapping(t *testing.T) {
	superUser := "admin/system:masters"

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authorization.Authorizer = newRBACAuthorizer(masterConfig)
	masterConfig.GenericConfig.Authentication.Authenticator = bearertoken.New(tokenfile.New(map[string]*user.DefaultInfo{
		superUser: {Name: "admin", Groups: []string{"system:masters"}},
	}))
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	clientset := clientset.NewForConfigOrDie(&restclient.Config{BearerToken: superUser, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	watcher, err := clientset.Rbac().ClusterRoles().Watch(metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, watcher, func(event watch.Event) (bool, error) {
		if event.Type != watch.Added {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	clusterRoles, err := clientset.Rbac().ClusterRoles().List(metav1.ListOptions{})
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

	healthBytes, err := clientset.Discovery().RESTClient().Get().AbsPath("/healthz/poststarthook/rbac/bootstrap-roles").DoRaw()
	if err != nil {
		t.Error(err)
	}
	t.Errorf("error bootstrapping roles: %s", string(healthBytes))
}
