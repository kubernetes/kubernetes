/*
Copyright 2024 The Kubernetes Authors.

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

// Package apiclient contains wrapping logic for Kubernetes API clients.
package apiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lithammer/dedent"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	fakerest "k8s.io/client-go/rest/fake"
	"k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// DryRun is responsible for performing verbose dry-run operations with a set of different
// API clients. Any REST action that reaches the FakeClient() of a DryRun will be processed
// as follows by the fake client reactor chain:
//   - Log the action.
//   - If the action is not GET or LIST just use the fake client store to write it, unless
//     a user reactor was added with PrependReactor() or AppendReactor().
//   - Attempt to GET or LIST using the real dynamic client.
//   - If the above fails try to GET or LIST the object from the fake client store, unless
//     a user reactor was added with PrependReactor() or AppendReactor().
type DryRun struct {
	fake               *testing.Fake
	fakeClient         clientset.Interface
	client             clientset.Interface
	dynamicClient      dynamic.Interface
	serverVersion      *version.Info
	listKinds          map[schema.GroupVersionResource]schema.GroupVersionKind
	requestInfoFactory *request.RequestInfoFactory

	writer      io.Writer
	marshalFunc func(runtime.Object, schema.GroupVersion) ([]byte, error)
}

// NewDryRun creates a new DryRun object that only has a fake client.
func NewDryRun() *DryRun {
	d := &DryRun{
		fake:          &testing.Fake{},
		serverVersion: constants.CurrentKubernetesVersion.Info(),
		listKinds:     map[schema.GroupVersionResource]schema.GroupVersionKind{},
		requestInfoFactory: &request.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
	}
	d.addListKinds()
	tracker := testing.NewObjectTracker(clientsetscheme.Scheme, clientsetscheme.Codecs.UniversalDecoder())
	d.fake.AddReactor("*", "*", testing.ObjectReaction(tracker))
	d.addReactors()
	d.fakeClient = clientset.NewForConfigOrDie(&rest.Config{
		Transport:          fakerest.CreateHTTPClient(d.roundTrip).Transport,
		ContentType:        runtime.ContentTypeJSON,
		AcceptContentTypes: runtime.ContentTypeJSON,
		QPS:                -1,
	})
	return d
}

// WithKubeConfigFile takes a file path and creates real clientset and dynamic clients.
func (d *DryRun) WithKubeConfigFile(file string) error {
	config, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return errors.Wrap(err, "failed to load kubeconfig")
	}
	return d.WithKubeConfig(config)
}

// WithKubeConfig takes a Config (kubeconfig) and creates real clientset and dynamic client.
func (d *DryRun) WithKubeConfig(config *clientcmdapi.Config) error {
	restConfig, err := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return errors.Wrap(err, "failed to create API client configuration from kubeconfig")
	}
	return d.WithRestConfig(restConfig)
}

// WithRestConfig takes a rest Config and creates real clientset and dynamic clients.
func (d *DryRun) WithRestConfig(config *rest.Config) error {
	var err error

	d.client, err = clientset.NewForConfig(config)
	if err != nil {
		return err
	}

	d.dynamicClient, err = dynamic.NewForConfig(config)
	if err != nil {
		return err
	}

	return nil
}

// WithWriter sets the io.Writer used for printing by the DryRun.
func (d *DryRun) WithWriter(w io.Writer) *DryRun {
	d.writer = w
	return d
}

// WithMarshalFunction sets the DryRun marshal function.
func (d *DryRun) WithMarshalFunction(f func(runtime.Object, schema.GroupVersion) ([]byte, error)) *DryRun {
	d.marshalFunc = f
	return d
}

// WithDefaultMarshalFunction sets the DryRun marshal function to the default one.
func (d *DryRun) WithDefaultMarshalFunction() *DryRun {
	d.WithMarshalFunction(kubeadmutil.MarshalToYaml)
	return d
}

// WithServerVersion sets the version that the fake client serves at /version.
func (d *DryRun) WithServerVersion(v *version.Info) *DryRun {
	d.serverVersion = v
	return d
}

// PrependReactor prepends a new reactor in the fake client ReactorChain at position 1.
// Keeps position 0 for the log reactor:
// [ log, r, ... rest of the chain, default fake client reactor ]
func (d *DryRun) PrependReactor(r *testing.SimpleReactor) *DryRun {
	log := d.fake.ReactionChain[0]
	chain := make([]testing.Reactor, len(d.fake.ReactionChain)+1)
	chain[0] = log
	chain[1] = r
	copy(chain[2:], d.fake.ReactionChain[1:])
	d.fake.ReactionChain = chain
	return d
}

// AppendReactor appends a new reactor in the fake client ReactorChain at position len-2.
// Keeps position len-1 for the default fake client reactor.
// [ log, rest of the chain... , r, default fake client reactor ]
func (d *DryRun) AppendReactor(r *testing.SimpleReactor) *DryRun {
	sz := len(d.fake.ReactionChain)
	def := d.fake.ReactionChain[sz-1]
	d.fake.ReactionChain[sz-1] = r
	d.fake.ReactionChain = append(d.fake.ReactionChain, def)
	return d
}

// Client returns the clientset for this DryRun.
func (d *DryRun) Client() clientset.Interface {
	return d.client
}

// DynamicClient returns the dynamic client for this DryRun.
func (d *DryRun) DynamicClient() dynamic.Interface {
	return d.dynamicClient
}

// FakeClient returns the fake client for this DryRun.
func (d *DryRun) FakeClient() clientset.Interface {
	return d.fakeClient
}

// addListKinds maps each resource to the kind the object tracker needs to build LIST results.
func (d *DryRun) addListKinds() {
	for gvk := range clientsetscheme.Scheme.AllKnownTypes() {
		kind, ok := strings.CutSuffix(gvk.Kind, "List")
		if !ok || kind == "" {
			continue
		}
		gvk.Kind = kind
		plural, _ := meta.UnsafeGuessKindToResource(gvk)
		d.listKinds[plural] = gvk
	}
}

// addRectors is by default called by NewDryRun after creating the fake client.
// It prepends a set of reactors before the default fake client reactor.
func (d *DryRun) addReactors() {
	reactors := []testing.Reactor{
		// Add a reactor for logging all requests that reach the fake client.
		&testing.SimpleReactor{
			Verb:     "*",
			Resource: "*",
			Reaction: func(action testing.Action) (bool, runtime.Object, error) {
				d.LogAction(action)
				return false, nil, nil
			},
		},
		// Add a reactor for all GET requests that reach the fake client.
		// This reactor calls the real dynamic client, but if it cannot process the object
		// the reactor chain is continued.
		&testing.SimpleReactor{
			Verb:     "get",
			Resource: "*",
			Reaction: func(action testing.Action) (bool, runtime.Object, error) {
				getAction, ok := action.(testing.GetAction)
				if !ok {
					return true, nil, errors.New("cannot cast reactor action to GetAction")
				}

				handled, obj, err := d.handleGetAction(getAction)
				if err != nil {
					fmt.Fprintln(d.writer, "[dryrun] Real object does not exist. "+
						"Attempting to GET from followup reactors or from the fake client tracker")
					return false, nil, nil
				}

				d.LogObject(obj, action.GetResource().GroupVersion())
				return handled, obj, err
			},
		},
		// Add a reactor for all LIST requests that reach the fake client.
		// This reactor calls the real dynamic client, but if it cannot process the object
		// the reactor chain is continued.
		&testing.SimpleReactor{
			Verb:     "list",
			Resource: "*",
			Reaction: func(action testing.Action) (bool, runtime.Object, error) {
				listAction, ok := action.(testing.ListAction)
				if !ok {
					return true, nil, errors.New("cannot cast reactor action to ListAction")
				}

				handled, obj, err := d.handleListAction(listAction)
				if err != nil {
					fmt.Fprintln(d.writer, "[dryrun] Real object does not exist. "+
						"Attempting to LIST from followup reactors or from the fake client tracker")
					return false, nil, nil
				}

				d.LogObject(obj, action.GetResource().GroupVersion())
				return handled, obj, err
			},
		},
	}
	d.fake.ReactionChain = append(reactors, d.fake.ReactionChain...)
}

// handleGetAction tries to handle all GET actions with the dynamic client.
func (d *DryRun) handleGetAction(action testing.GetAction) (bool, runtime.Object, error) {
	if d.dynamicClient == nil {
		return false, nil, errors.New("dynamicClient is nil")
	}

	unstructuredObj, err := d.dynamicClient.
		Resource(action.GetResource()).
		Namespace(action.GetNamespace()).
		Get(context.Background(), action.GetName(), metav1.GetOptions{})
	if err != nil {
		return true, nil, err
	}

	newObj, err := d.decodeUnstructuredIntoAPIObject(action, unstructuredObj)
	if err != nil {
		return true, nil, err
	}

	return true, newObj, err
}

// handleListAction tries to handle all LIST actions with the dynamic client.
func (d *DryRun) handleListAction(action testing.ListAction) (bool, runtime.Object, error) {
	if d.dynamicClient == nil {
		return false, nil, errors.New("dynamicClient is nil")
	}

	listOpts := metav1.ListOptions{
		LabelSelector: action.GetListRestrictions().Labels.String(),
		FieldSelector: action.GetListRestrictions().Fields.String(),
	}

	unstructuredObj, err := d.dynamicClient.
		Resource(action.GetResource()).
		Namespace(action.GetNamespace()).
		List(context.Background(), listOpts)
	if err != nil {
		return true, nil, err
	}

	newObj, err := d.decodeUnstructuredIntoAPIObject(action, unstructuredObj)
	if err != nil {
		return true, nil, err
	}

	return true, newObj, err
}

// decodeUnstructuredIntoAPIObject decodes an unstructured object into an API object.
func (d *DryRun) decodeUnstructuredIntoAPIObject(action testing.Action, obj runtime.Unstructured) (runtime.Object, error) {
	objBytes, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	newObj, err := runtime.Decode(clientsetscheme.Codecs.UniversalDecoder(action.GetResource().GroupVersion()), objBytes)
	if err != nil {
		return nil, err
	}
	return newObj, nil
}

// roundTrip answers clientset requests in-process by turning each one into the
// client-go testing Action the typed fake clients produce and running the reactor chain.
func (d *DryRun) roundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Path == "/version" {
		data, err := json.Marshal(d.serverVersion)
		if err != nil {
			return errorResponse(req, err), nil
		}
		return jsonResponse(req, http.StatusOK, data), nil
	}
	action, err := actionFor(req, d.requestInfoFactory, d.listKinds)
	if err != nil {
		return errorResponse(req, err), nil
	}
	obj, err := d.fake.Invokes(action, nil)
	if err != nil {
		return errorResponse(req, err), nil
	}
	if list, ok := action.(testing.ListAction); ok && obj != nil {
		if err := filterByLabels(obj, list.GetListRestrictions().Labels); err != nil {
			return errorResponse(req, err), nil
		}
	}
	if obj == nil {
		obj = &metav1.Status{Status: metav1.StatusSuccess, Code: http.StatusOK}
	}
	data, err := runtime.Encode(clientsetscheme.Codecs.LegacyCodec(action.GetResource().GroupVersion()), obj)
	if err != nil {
		return errorResponse(req, err), nil
	}
	return jsonResponse(req, http.StatusOK, data), nil
}

// LogAction logs details about an action, such as name, object and resource.
func (d *DryRun) LogAction(action testing.Action) {
	// actionWithName is the generic interface for an action that has a name associated with it.
	type actionWithNameAndNamespace interface {
		testing.Action
		GetName() string
		GetNamespace() string
	}

	// actionWithObject is the generic interface for an action that has an object associated with it.
	type actionWithObject interface {
		testing.Action
		GetObject() runtime.Object
	}

	group := action.GetResource().Group
	if len(group) == 0 {
		group = "core"
	}
	fmt.Fprintf(d.writer, "[dryrun] Would perform action %s on resource %q in API group \"%s/%s\"\n",
		strings.ToUpper(action.GetVerb()), action.GetResource().Resource, group, action.GetResource().Version)

	namedAction, ok := action.(actionWithNameAndNamespace)
	if ok {
		fmt.Fprintf(d.writer, "[dryrun] Resource name %q, namespace %q\n",
			namedAction.GetName(), namedAction.GetNamespace())
	}

	objAction, ok := action.(actionWithObject)
	if ok && objAction.GetObject() != nil {
		d.LogObject(objAction.GetObject(), objAction.GetResource().GroupVersion())
	}
}

// LogObject marshals the object and then prints it to the io.Writer of this DryRun.
func (d *DryRun) LogObject(obj runtime.Object, gv schema.GroupVersion) {
	objBytes, err := d.marshalFunc(obj, gv)
	if err == nil {
		fmt.Fprintln(d.writer, "[dryrun] Attached object:")
		fmt.Fprintln(d.writer, string(objBytes))
	}
}

// HealthCheckJobReactor returns a reactor that handles the GET action for the Job
// object used for the "CreateJob" upgrade preflight check.
func (d *DryRun) HealthCheckJobReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "jobs",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if !strings.HasPrefix(a.GetName(), "upgrade-health-check") || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getJob(a.GetName(), a.GetNamespace())
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// PatchNodeReactor returns a reactor that handles the generic PATCH action on Node objects.
func (d *DryRun) PatchNodeReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "patch",
		Resource: "nodes",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.PatchAction)
			obj := getNode(a.GetName())
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetNodeReactor returns a reactor that handles the generic GET action of Node objects.
func (d *DryRun) GetNodeReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "nodes",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			obj := getNode(a.GetName())
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetClusterInfoReactor returns a reactor that handles the GET action of the "cluster-info"
// ConfigMap used during node bootstrap.
func (d *DryRun) GetClusterInfoReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "configmaps",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != bootstrapapi.ConfigMapClusterInfo || a.GetNamespace() != metav1.NamespacePublic {
				return false, nil, nil
			}
			obj := getClusterInfoConfigMap()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetKubeadmConfigReactor returns a reactor that handles the GET action of the "kubeadm-config"
// ConfigMap.
func (d *DryRun) GetKubeadmConfigReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "configmaps",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != constants.KubeadmConfigConfigMap || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}

			obj := getKubeadmConfigMap()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetKubeadmCertsReactor returns a reactor that handles the GET action of the "kubeadm-certs" Secret.
func (d *DryRun) GetKubeadmCertsReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "secrets",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != constants.KubeadmCertsSecret || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getKubeadmCertsSecret()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetKubeletConfigReactor returns a reactor that handles the GET action of the "kubelet-config"
// ConfigMap.
func (d *DryRun) GetKubeletConfigReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "configmaps",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != constants.KubeletBaseConfigurationConfigMap || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getKubeletConfigMap()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetKubeProxyConfigReactor returns a reactor that handles the GET action of the "kube-proxy"
// ConfigMap.
func (d *DryRun) GetKubeProxyConfigReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "configmaps",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != constants.KubeProxyConfigMap || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getKubeProxyConfigMap()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// GetCoreDNSConfigReactor returns a reactor that handles the GET action of the "coredns"
// ConfigMap.
func (d *DryRun) GetCoreDNSConfigReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "get",
		Resource: "configmaps",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.GetAction)
			if a.GetName() != constants.CoreDNSConfigMap || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getCoreDNSConfigMap()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// DeleteBootstrapTokenReactor returns a reactor that handles the DELETE action
// of bootstrap token Secret.
func (d *DryRun) DeleteBootstrapTokenReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "delete",
		Resource: "secrets",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.DeleteAction)
			if !strings.HasPrefix(a.GetName(), bootstrapapi.BootstrapTokenSecretPrefix) || a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      a.GetName(),
					Namespace: a.GetNamespace(),
				},
			}
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// ListPodsReactor returns a reactor that handles the LIST action on pods.
func (d *DryRun) ListPodsReactor(nodeName string) *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "list",
		Resource: "pods",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.ListAction)
			if a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getPodList(nodeName)
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// ListDeploymentsReactor returns a reactor that handles the LIST action on deployments.
func (d *DryRun) ListDeploymentsReactor() *testing.SimpleReactor {
	return &testing.SimpleReactor{
		Verb:     "list",
		Resource: "deployments",
		Reaction: func(action testing.Action) (bool, runtime.Object, error) {
			a := action.(testing.ListAction)
			if a.GetNamespace() != metav1.NamespaceSystem {
				return false, nil, nil
			}
			obj := getDeploymentList()
			d.LogObject(obj, action.GetResource().GroupVersion())
			return true, obj, nil
		},
	}
}

// getJob returns a fake Job object.
func getJob(namespace, name string) *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Status: batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{
				{Type: batchv1.JobComplete},
			},
		},
	}
}

// getNode returns a fake Node object.
func getNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"kubernetes.io/hostname": name,
			},
		},
	}
}

// getConfigMap returns a fake ConfigMap object.
func getConfigMap(namespace, name string, data map[string]string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: data,
	}
}

// getSecret returns a fake Secret object.
func getSecret(namespace, name string, data map[string][]byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: data,
	}
}

// getClusterInfoConfigMap returns a fake "cluster-info" ConfigMap.
func getClusterInfoConfigMap() *corev1.ConfigMap {
	kubeconfig := dedent.Dedent(`apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURCVENDQWUyZ0F3SUJBZ0lJTkpmdGFCK09Xd0F3RFFZSktvWklodmNOQVFFTEJRQXdGVEVUTUJFR0ExVUUKQXhNS2EzVmlaWEp1WlhSbGN6QWVGdzB5TkRBM01EZ3hNalF3TURSYUZ3MHpOREEzTURZeE1qUTFNRFJhTUJVeApFekFSQmdOVkJBTVRDbXQxWW1WeWJtVjBaWE13Z2dFaU1BMEdDU3FHU0liM0RRRUJBUVVBQTRJQkR3QXdnZ0VLCkFvSUJBUURhUURkaWdPeVdpbTZXLzQ4bjNQSG9WZVZCU2lkNldjbmRFV3VVcTVnQldYZGx0OTk2aCtWbkl0bHQKOHpDaGwvb1I1V2ZSYVJDODA1WitvTW4vWThJR1ZRM3QxaG55SW1ZbjR3M3Z6UlhvdUdlQmVpdTJTU1ZqZ0J3agpYanliYk1DbXJBZEljYkllWm1INjZldjV6KzVZS21aUlVZYzNoRGFIcFhkMEVFblp5SlY1d2FaczBYTVFVSE03CmVxT1pBWko5L21PM05VQnBsdnJQbnBPTUs3a1NFUFBnNzVjVTdXSG9KSEZrZVlXNTkzZ3NnQ3MyQnRVdTY0Y3EKYlZYOWJpZ3JZZGRwWmtvRUtLeFU4SEl3SHNJNVY4Um9uM21LRkdsckUxN2IybC84Q3FtQXVPdnl6TEllaVFHWAplZ0lhUi9uUkhISUQ5QVRpNnRYOEVhRERwZXYvQWdNQkFBR2pXVEJYTUE0R0ExVWREd0VCL3dRRUF3SUNwREFQCkJnTlZIUk1CQWY4RUJUQURBUUgvTUIwR0ExVWREZ1FXQkJRWWFDRGU2eXVWcVNhV1Y3M3pMaldtR2hpYVR6QVYKQmdOVkhSRUVEakFNZ2dwcmRXSmxjbTVsZEdWek1BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQ2NlTW5uaVBhcQpHUkVqR3A2WFJTN1FFWW1RcmJFZCtUVjVKUTE4byttVzZvR3BaOENBM0pBSFFETk5QRW1QYzB5dVhEeE85QkZYCmJnMlhSSCtLTkozVzJKZlExVEgzVFhKRWF4Zkh1WDRtQjU5UkNsQzExNGtsV2RIeDFqN0FtRWt1eTZ0ZGpuYWkKZmh0U0dueEEwM2JwN2I4Z28zSWpXNE5wV1JOMVdHNTl2YTBKOEJIRmg3Q0RpZUxuK0RNdUk2M0Jna1kveTJzMApML2RtOVBmcWdVSzFBMy8wZGhDVjZiRUNqekEzSkJld21kSC8rVUJPeVkybUMwNVlQMzNkMHA5eXlrYmtkWE5xCkRONXlBc3ZNUC9PV0NuQjFlQlFUb2pNODJMU3F3dHZtbU1SNHRXYXVoOXVkVktHY2s0eEJaV3Vkcm5LRFVVWEkKUURNUFJnSkMvTng0Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
    server: https://192.168.0.101:6443
  name: ""
contexts: null
current-context: ""
kind: Config
users: null
`)
	data := map[string]string{
		bootstrapapi.JWSSignatureKeyPrefix + "abcdef": "eyJhbGciOiJIUzI1NiIsImtpZCI6ImFiY2RlZiJ9..rh3cVKiU2mdt3CqHzC81sNE-4WQLRHMtXduHWfSbrIM",
		bootstrapapi.KubeConfigKey:                    kubeconfig,
	}
	return getConfigMap(metav1.NamespacePublic, bootstrapapi.ConfigMapClusterInfo, data)
}

// getKubeadmConfigMap returns a fake "kubeadm-config" ConfigMap.
func getKubeadmConfigMap() *corev1.ConfigMap {
	ccData := fmt.Sprintf(`apiServer: {}
apiVersion: kubeadm.k8s.io/v1beta4
caCertificateValidityPeriod: 87600h0m0s
certificateValidityPeriod: 100h0m0s
certificatesDir: /etc/kubernetes/pki
clusterName: kubernetes
controllerManager:
  extraArgs:
  - name: cluster-signing-duration
    value: 24h
controlPlaneEndpoint: 192.168.0.101:6443
dns: {}
encryptionAlgorithm: RSA-2048
etcd:
  local:
    dataDir: /var/lib/etcd
imageRepository: registry.k8s.io
kind: ClusterConfiguration
kubernetesVersion: %s
networking:
  dnsDomain: cluster.local
  podSubnet: 192.168.0.0/16
  serviceSubnet: 10.96.0.0/12
proxy: {}
scheduler: {}
`, constants.MinimumControlPlaneVersion)

	data := map[string]string{
		constants.ClusterConfigurationKind: ccData,
	}
	return getConfigMap(metav1.NamespaceSystem, constants.KubeadmConfigConfigMap, data)
}

// getKubeadmCertsSecret returns a fake "kubeadm-certs" Secret.
func getKubeadmCertsSecret() *corev1.Secret {
	// The cert data is empty because the actual content is not relevant for the dryrun test.
	data := map[string][]byte{
		constants.CACertName:                                   {},
		constants.CAKeyName:                                    {},
		constants.FrontProxyCACertName:                         {},
		constants.FrontProxyCAKeyName:                          {},
		constants.ServiceAccountPrivateKeyName:                 {},
		constants.ServiceAccountPublicKeyName:                  {},
		strings.ReplaceAll(constants.EtcdCACertName, "/", "-"): {},
		strings.ReplaceAll(constants.EtcdCAKeyName, "/", "-"):  {},
	}

	return getSecret(metav1.NamespaceSystem, constants.KubeadmCertsSecret, data)
}

// getKubeletConfigMap returns a fake "kubelet-config" ConfigMap.
func getKubeletConfigMap() *corev1.ConfigMap {
	configData := `apiVersion: kubelet.config.k8s.io/v1beta1
authentication:
  anonymous:
    enabled: false
  webhook:
    enabled: true
  x509:
    clientCAFile: /etc/kubernetes/pki/ca.crt
authorization:
  mode: Webhook
cgroupDriver: systemd
clusterDNS:
- 10.96.0.10
clusterDomain: cluster.local
healthzBindAddress: 127.0.0.1
healthzPort: 10248
kind: KubeletConfiguration
memorySwap: {}
resolvConf: /run/systemd/resolve/resolv.conf
rotateCertificates: true
staticPodPath: /etc/kubernetes/manifests
`
	data := map[string]string{
		constants.KubeletBaseConfigurationConfigMapKey: configData,
	}
	return getConfigMap(metav1.NamespaceSystem, constants.KubeletBaseConfigurationConfigMap, data)
}

// getKubeProxyConfigMap returns a fake "kube-proxy" ConfigMap.
func getKubeProxyConfigMap() *corev1.ConfigMap {
	configData := `apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
bindAddressHardFail: false
clientConnection:
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
clusterCIDR: 192.168.0.0/16
kind: KubeProxyConfiguration
`
	kubeconfigData := `apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    server: https://192.168.0.101:6443
  name: default
contexts:
- context:
    cluster: default
    namespace: default
    user: default
  name: default
current-context: default
users:
- name: default
  user:
    tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
`
	data := map[string]string{
		constants.KubeProxyConfigMapKey: configData,
		"kubeconfig.conf":               kubeconfigData,
	}
	return getConfigMap(metav1.NamespaceSystem, constants.KubeProxyConfigMap, data)
}

// getCoreDNSConfigMap returns a fake "coredns" ConfigMap.
func getCoreDNSConfigMap() *corev1.ConfigMap {
	data := map[string]string{
		"Corefile": "",
	}
	return getConfigMap(metav1.NamespaceSystem, constants.CoreDNSConfigMap, data)
}

// getPod returns a fake Pod.
func getPod(name, nodeName string) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name + "-" + nodeName,
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"component": name,
				"tier":      constants.ControlPlaneTier,
			},
			Annotations: map[string]string{
				constants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "0.0.0.0:6443",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name:  name,
					Image: "registry.k8s.io/" + name + ":v1.1.1",
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}
}

// getPodList returns a list of fake pods.
func getPodList(nodeName string) *corev1.PodList {
	return &corev1.PodList{
		Items: []corev1.Pod{
			getPod(constants.KubeAPIServer, nodeName),
			getPod(constants.KubeControllerManager, nodeName),
			getPod(constants.KubeScheduler, nodeName),
			getPod(constants.Etcd, nodeName),
		},
	}
}

// getDeploymentList returns a fake list of deployments.
func getDeploymentList() *appsv1.DeploymentList {
	return &appsv1.DeploymentList{
		Items: []appsv1.Deployment{
			{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"k8s-app": "kube-dns",
					},
				},
				Spec: appsv1.DeploymentSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image: "registry.k8s.io/coredns/coredns:" + constants.CoreDNSVersion,
								},
							},
						},
					},
				},
			},
		},
	}
}

// actionFor maps a request to the testing Action the typed fake clients would have produced.
func actionFor(req *http.Request, infoFactory *request.RequestInfoFactory, listKinds map[schema.GroupVersionResource]schema.GroupVersionKind) (testing.Action, error) {
	info, err := infoFactory.NewRequestInfo(req)
	if err != nil {
		return nil, apierrors.NewBadRequest(err.Error())
	}
	if !info.IsResourceRequest {
		return nil, apierrors.NewNotFound(schema.GroupResource{}, req.URL.Path)
	}
	gvr := schema.GroupVersionResource{Group: info.APIGroup, Version: info.APIVersion, Resource: info.Resource}

	switch info.Verb {
	case "get":
		return testing.NewGetSubresourceAction(gvr, info.Namespace, info.Subresource, info.Name), nil
	case "list":
		// NewListActionWithOptions panics on malformed selectors; parse them first.
		query := req.URL.Query()
		opts := metav1.ListOptions{
			LabelSelector: query.Get("labelSelector"),
			FieldSelector: query.Get("fieldSelector"),
		}
		if _, err := labels.Parse(opts.LabelSelector); err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		if _, err := fields.ParseSelector(opts.FieldSelector); err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		return testing.NewListActionWithOptions(gvr, listKinds[gvr], info.Namespace, opts), nil
	case "create", "update", "patch":
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		if info.Verb == "patch" {
			patchType := types.PatchType(req.Header.Get("Content-Type"))
			return testing.NewPatchSubresourceAction(gvr, info.Namespace, info.Name, patchType, body, info.Subresource), nil
		}
		obj, _, err := clientsetscheme.Codecs.UniversalDeserializer().Decode(body, nil, nil)
		if err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		if info.Verb == "create" {
			return testing.NewCreateSubresourceAction(gvr, info.Name, info.Subresource, info.Namespace, obj), nil
		}
		return testing.NewUpdateSubresourceAction(gvr, info.Subresource, info.Namespace, obj), nil
	case "delete":
		return testing.NewDeleteSubresourceAction(gvr, info.Subresource, info.Namespace, info.Name), nil
	}
	// kubeadm never watches or deletes collections.
	return nil, apierrors.NewMethodNotSupported(gvr.GroupResource(), info.Verb)
}

// filterByLabels drops the list items sel does not match, as the typed fake clients did.
func filterByLabels(obj runtime.Object, sel labels.Selector) error {
	if sel == nil || sel.Empty() {
		return nil
	}
	items, err := meta.ExtractList(obj)
	if err != nil {
		return err
	}
	kept := items[:0]
	for _, item := range items {
		if m, err := meta.Accessor(item); err == nil && sel.Matches(labels.Set(m.GetLabels())) {
			kept = append(kept, item)
		}
	}
	return meta.SetList(obj, kept)
}

// errorResponse encodes err as a metav1.Status like the API server does, so that helpers
// such as apierrors.IsNotFound keep working for callers of the dry-run client.
func errorResponse(req *http.Request, err error) *http.Response {
	var apiStatus apierrors.APIStatus
	if !errors.As(err, &apiStatus) {
		apiStatus = apierrors.NewInternalError(err)
	}
	status := apiStatus.Status()
	if status.Code == 0 {
		status.Code = http.StatusInternalServerError
	}
	data, err := runtime.Encode(clientsetscheme.Codecs.LegacyCodec(metav1.SchemeGroupVersion), &status)
	if err != nil {
		return jsonResponse(req, http.StatusInternalServerError, []byte(err.Error()))
	}
	return jsonResponse(req, int(status.Code), data)
}

func jsonResponse(req *http.Request, code int, data []byte) *http.Response {
	return &http.Response{
		StatusCode: code,
		Header:     http.Header{"Content-Type": []string{runtime.ContentTypeJSON}},
		Body:       io.NopCloser(bytes.NewReader(data)),
		Request:    req,
	}
}
