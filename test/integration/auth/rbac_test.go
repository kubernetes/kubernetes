// +build integration,!no-etcd

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
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/transport"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
	clusterroleetcd "k8s.io/kubernetes/pkg/registry/rbac/clusterrole/etcd"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding"
	clusterrolebindingetcd "k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding/etcd"
	"k8s.io/kubernetes/pkg/registry/rbac/role"
	roleetcd "k8s.io/kubernetes/pkg/registry/rbac/role/etcd"
	"k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
	rolebindingetcd "k8s.io/kubernetes/pkg/registry/rbac/rolebinding/etcd"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/anytoken"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/test/integration/framework"
)

func newFakeAuthenticator() authenticator.Request {
	return bearertoken.New(anytoken.AnyTokenAuthenticator{})
}

func clientForUser(user string) *http.Client {
	return &http.Client{
		Transport: transport.NewBearerAuthRoundTripper(
			user,
			transport.DebugWrappers(http.DefaultTransport),
		),
	}
}

func clientsetForUser(user string, config *restclient.Config) clientset.Interface {
	configCopy := *config
	configCopy.BearerToken = user
	return clientset.NewForConfigOrDie(&configCopy)
}

func newRBACAuthorizer(t *testing.T, config *master.Config) authorizer.Authorizer {
	newRESTOptions := func(resource string) generic.RESTOptions {
		storageConfig, err := config.StorageFactory.NewConfig(rbacapi.Resource(resource))
		if err != nil {
			t.Fatalf("failed to get storage: %v", err)
		}
		return generic.RESTOptions{StorageConfig: storageConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: resource}
	}

	roleRegistry := role.AuthorizerAdapter{Registry: role.NewRegistry(roleetcd.NewREST(newRESTOptions("roles")))}
	roleBindingRegistry := rolebinding.AuthorizerAdapter{Registry: rolebinding.NewRegistry(rolebindingetcd.NewREST(newRESTOptions("rolebindings")))}
	clusterRoleRegistry := clusterrole.AuthorizerAdapter{Registry: clusterrole.NewRegistry(clusterroleetcd.NewREST(newRESTOptions("clusterroles")))}
	clusterRoleBindingRegistry := clusterrolebinding.AuthorizerAdapter{Registry: clusterrolebinding.NewRegistry(clusterrolebindingetcd.NewREST(newRESTOptions("clusterrolebindings")))}
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
	// The username attempting to send the request.
	user string

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
	return fmt.Sprintf("%s %s %s", r.user, r.verb, r.resource)
}

type statusCode int

func (s statusCode) String() string {
	return fmt.Sprintf("%d %s", int(s), http.StatusText(int(s)))
}

// Declare a set of raw objects to use.
var (
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
	podNamespace = `
{
  "apiVersion": "` + registered.GroupOrDie(api.GroupName).GroupVersion.String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "pod-namespace"%s
  }
}
`
	jobNamespace = `
{
  "apiVersion": "` + registered.GroupOrDie(api.GroupName).GroupVersion.String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "job-namespace"%s
  }
}
`
	forbiddenNamespace = `
{
  "apiVersion": "` + registered.GroupOrDie(api.GroupName).GroupVersion.String() + `",
  "kind": "Namespace",
  "metadata": {
	"name": "forbidden-namespace"%s
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
						ObjectMeta: api.ObjectMeta{Name: "allow-all"},
						Rules:      []rbacapi.PolicyRule{ruleAllowAll},
					},
					{
						ObjectMeta: api.ObjectMeta{Name: "read-pods"},
						Rules:      []rbacapi.PolicyRule{ruleReadPods},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: api.ObjectMeta{Name: "read-pods"},
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
						ObjectMeta: api.ObjectMeta{Name: "write-jobs"},
						Rules:      []rbacapi.PolicyRule{ruleWriteJobs},
					},
				},
				clusterRoleBindings: []rbacapi.ClusterRoleBinding{
					{
						ObjectMeta: api.ObjectMeta{Name: "write-jobs"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "job-writer"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "write-jobs"},
					},
				},
				roleBindings: []rbacapi.RoleBinding{
					{
						ObjectMeta: api.ObjectMeta{Name: "write-jobs", Namespace: "job-namespace"},
						Subjects:   []rbacapi.Subject{{Kind: "User", Name: "job-writer-namespace"}},
						RoleRef:    rbacapi.RoleRef{Kind: "ClusterRole", Name: "write-jobs"},
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
			},
		},
	}

	for i, tc := range tests {
		// Create an API Server.
		masterConfig := framework.NewIntegrationTestMasterConfig()
		masterConfig.GenericConfig.Authorizer = newRBACAuthorizer(t, masterConfig)
		masterConfig.GenericConfig.Authenticator = newFakeAuthenticator()
		_, s := framework.RunAMaster(masterConfig)
		defer s.Close()

		clientConfig := &restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}

		// Bootstrap the API Server with the test case's initial roles.
		if err := tc.bootstrapRoles.bootstrap(clientsetForUser(superUser, clientConfig)); err != nil {
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
				// For any creation requests, add the namespace to the object meta.
				if r.verb == "POST" || r.verb == "PUT" {
					if r.namespace != "" {
						sub += fmt.Sprintf(",\"namespace\": %q", r.namespace)
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

				resp, err := clientForUser(r.user).Do(req)
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
					glog.V(8).Infof("case %d, req %d: %s\n%s\n", i, j, reqDump, respDump)
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
	masterConfig.GenericConfig.Authorizer = newRBACAuthorizer(t, masterConfig)
	masterConfig.GenericConfig.Authenticator = newFakeAuthenticator()
	_, s := framework.RunAMaster(masterConfig)
	defer s.Close()

	clientset := clientset.NewForConfigOrDie(&restclient.Config{BearerToken: superUser, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})

	watcher, err := clientset.Rbac().ClusterRoles().Watch(api.ListOptions{ResourceVersion: "0"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = watch.Until(30*time.Second, watcher, func(event watch.Event) (bool, error) {
		if event.Type != watch.Added {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	clusterRoles, err := clientset.Rbac().ClusterRoles().List(api.ListOptions{})
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

	healthBytes, err := clientset.Discovery().RESTClient().Get().AbsPath("/healthz/poststarthooks/rbac/bootstrap-roles").DoRaw()
	if err != nil {
		t.Error(err)
	}
	t.Errorf("expected %v, got %v", "asdf", string(healthBytes))
}
