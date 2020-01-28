/*
Copyright 2018 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	discoveryv1alpha1 "k8s.io/api/discovery/v1alpha1"
	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	schedulerapi "k8s.io/api/scheduling/v1"
	settingsv1alpha1 "k8s.io/api/settings/v1alpha1"
	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	diskcached "k8s.io/client-go/discovery/cached/disk"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	"k8s.io/kubernetes/test/integration/framework"
)

var kindWhiteList = sets.NewString(
	// k8s.io/api/core
	"APIGroup",
	"APIVersions",
	"Binding",
	"DeleteOptions",
	"EphemeralContainers",
	"ExportOptions",
	"GetOptions",
	"ListOptions",
	"CreateOptions",
	"UpdateOptions",
	"PatchOptions",
	"NodeProxyOptions",
	"PodAttachOptions",
	"PodExecOptions",
	"PodPortForwardOptions",
	"PodLogOptions",
	"PodProxyOptions",
	"PodStatusResult",
	"RangeAllocation",
	"ServiceProxyOptions",
	"SerializedReference",
	// --

	// k8s.io/api/admission
	"AdmissionReview",
	// --

	// k8s.io/api/authentication
	"TokenRequest",
	"TokenReview",
	// --

	// k8s.io/api/authorization
	"LocalSubjectAccessReview",
	"SelfSubjectAccessReview",
	"SelfSubjectRulesReview",
	"SubjectAccessReview",
	// --

	// k8s.io/api/autoscaling
	"Scale",
	// --

	// k8s.io/api/apps
	"DeploymentRollback",
	// --

	// k8s.io/api/batch
	"JobTemplate",
	// --

	// k8s.io/api/imagepolicy
	"ImageReview",
	// --

	// k8s.io/api/policy
	"Eviction",
	// --

	// k8s.io/apimachinery/pkg/apis/meta
	"WatchEvent",
	"Status",
	// --
)

// TODO (soltysh): this list has to go down to 0!
var missingHanlders = sets.NewString(
	"ClusterRole",
	"LimitRange",
	"ResourceQuota",
	"Role",
	"PriorityClass",
	"PodPreset",
	"AuditSink",
	"FlowSchema",                 // TODO(yue9944882): remove this comment by merging print-handler for flow-control API
	"PriorityLevelConfiguration", // TODO(yue9944882): remove this comment by merging print-handler for flow-control API
)

// known types that are no longer served we should tolerate restmapper errors for
var unservedTypes = map[schema.GroupVersionKind]bool{
	{Group: "extensions", Version: "v1beta1", Kind: "ControllerRevision"}: true,
	{Group: "extensions", Version: "v1beta1", Kind: "DaemonSet"}:          true,
	{Group: "extensions", Version: "v1beta1", Kind: "Deployment"}:         true,
	{Group: "extensions", Version: "v1beta1", Kind: "NetworkPolicy"}:      true,
	{Group: "extensions", Version: "v1beta1", Kind: "PodSecurityPolicy"}:  true,
	{Group: "extensions", Version: "v1beta1", Kind: "ReplicaSet"}:         true,

	{Group: "apps", Version: "v1beta1", Kind: "ControllerRevision"}: true,
	{Group: "apps", Version: "v1beta1", Kind: "DaemonSet"}:          true,
	{Group: "apps", Version: "v1beta1", Kind: "Deployment"}:         true,
	{Group: "apps", Version: "v1beta1", Kind: "ReplicaSet"}:         true,
	{Group: "apps", Version: "v1beta1", Kind: "StatefulSet"}:        true,

	{Group: "apps", Version: "v1beta2", Kind: "ControllerRevision"}: true,
	{Group: "apps", Version: "v1beta2", Kind: "DaemonSet"}:          true,
	{Group: "apps", Version: "v1beta2", Kind: "Deployment"}:         true,
	{Group: "apps", Version: "v1beta2", Kind: "ReplicaSet"}:         true,
	{Group: "apps", Version: "v1beta2", Kind: "StatefulSet"}:        true,
}

func TestServerSidePrint(t *testing.T) {
	s, _, closeFn := setupWithResources(t,
		// additional groupversions needed for the test to run
		[]schema.GroupVersion{
			auditregv1alpha1.SchemeGroupVersion,
			batchv2alpha1.SchemeGroupVersion,
			discoveryv1alpha1.SchemeGroupVersion,
			discoveryv1beta1.SchemeGroupVersion,
			rbacv1alpha1.SchemeGroupVersion,
			settingsv1alpha1.SchemeGroupVersion,
			schedulerapi.SchemeGroupVersion,
			storagev1alpha1.SchemeGroupVersion,
			extensionsv1beta1.SchemeGroupVersion,
			nodev1alpha1.SchemeGroupVersion,
			flowcontrolv1alpha1.SchemeGroupVersion,
		},
		[]schema.GroupVersionResource{},
	)
	defer closeFn()

	ns := framework.CreateTestingNamespace("server-print", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	tableParam := fmt.Sprintf("application/json;as=Table;g=%s;v=%s, application/json", metav1beta1.GroupName, metav1beta1.SchemeGroupVersion.Version)
	printer := newFakePrinter(printersinternal.AddHandlers)

	configFlags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(clientcmd.NewDefaultClientConfig(*createKubeConfig(s.URL), &clientcmd.ConfigOverrides{}))

	restConfig, err := configFlags.ToRESTConfig()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	cacheDir, err := ioutil.TempDir(os.TempDir(), "test-integration-apiserver-print")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer func() {
		os.Remove(cacheDir)
	}()

	cachedClient, err := diskcached.NewCachedDiscoveryClientForConfig(restConfig, cacheDir, "", time.Duration(10*time.Minute))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	configFlags.WithDiscoveryClient(cachedClient)

	factory := util.NewFactory(configFlags)
	mapper, err := factory.ToRESTMapper()
	if err != nil {
		t.Errorf("unexpected error getting mapper: %v", err)
		return
	}
	for gvk, apiType := range legacyscheme.Scheme.AllKnownTypes() {
		// we do not care about internal objects or lists // TODO make sure this is always true
		if gvk.Version == runtime.APIVersionInternal || strings.HasSuffix(apiType.Name(), "List") {
			continue
		}
		if kindWhiteList.Has(gvk.Kind) || missingHanlders.Has(gvk.Kind) {
			continue
		}

		t.Logf("Checking %s", gvk)
		// read table definition as returned by the server
		mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			if unservedTypes[gvk] {
				continue
			}
			t.Errorf("unexpected error getting mapping for GVK %s: %v", gvk, err)
			continue
		}
		client, err := factory.ClientForMapping(mapping)
		if err != nil {
			t.Errorf("unexpected error getting client for GVK %s: %v", gvk, err)
			continue
		}
		req := client.Get()
		if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
			req = req.Namespace(ns.Name)
		}
		body, err := req.Resource(mapping.Resource.Resource).SetHeader("Accept", tableParam).Do(context.TODO()).Raw()
		if err != nil {
			t.Errorf("unexpected error getting %s: %v", gvk, err)
			continue
		}
		actual, err := decodeIntoTable(body)
		if err != nil {
			t.Errorf("unexpected error decoding %s: %v", gvk, err)
			continue
		}

		// get table definition used in printers
		obj, err := legacyscheme.Scheme.New(gvk)
		if err != nil {
			t.Errorf("unexpected error creating %s: %v", gvk, err)
			continue
		}
		intGV := gvk.GroupKind().WithVersion(runtime.APIVersionInternal).GroupVersion()
		intObj, err := legacyscheme.Scheme.ConvertToVersion(obj, intGV)
		if err != nil {
			t.Errorf("unexpected error converting %s to internal: %v", gvk, err)
			continue
		}
		expectedColumnDefinitions, ok := printer.handlers[reflect.TypeOf(intObj)]
		if !ok {
			t.Errorf("missing handler for type %v", gvk)
			continue
		}

		for _, e := range expectedColumnDefinitions {
			for _, a := range actual.ColumnDefinitions {
				if a.Name == e.Name && !reflect.DeepEqual(a, e) {
					t.Errorf("unexpected difference in column definition %s for %s:\nexpected:\n%#v\nactual:\n%#v\n", e.Name, gvk, e, a)
				}
			}
		}
	}
}

type fakePrinter struct {
	handlers map[reflect.Type][]metav1beta1.TableColumnDefinition
}

var _ printers.PrintHandler = &fakePrinter{}

func (f *fakePrinter) Handler(columns, columnsWithWide []string, printFunc interface{}) error {
	return nil
}

func (f *fakePrinter) TableHandler(columns []metav1beta1.TableColumnDefinition, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	objType := printFuncValue.Type().In(0)
	f.handlers[objType] = columns
	return nil
}

func (f *fakePrinter) DefaultTableHandler(columns []metav1beta1.TableColumnDefinition, printFunc interface{}) error {
	return nil
}

func newFakePrinter(fns ...func(printers.PrintHandler)) *fakePrinter {
	handlers := make(map[reflect.Type][]metav1beta1.TableColumnDefinition, len(fns))
	p := &fakePrinter{handlers: handlers}
	for _, fn := range fns {
		fn(p)
	}
	return p
}

func decodeIntoTable(body []byte) (*metav1beta1.Table, error) {
	table := &metav1beta1.Table{}
	err := json.Unmarshal(body, table)
	if err != nil {
		return nil, err
	}
	return table, nil
}

func createKubeConfig(url string) *clientcmdapi.Config {
	clusterNick := "cluster"
	userNick := "user"
	contextNick := "context"

	config := clientcmdapi.NewConfig()

	cluster := clientcmdapi.NewCluster()
	cluster.Server = url
	cluster.InsecureSkipTLSVerify = true
	config.Clusters[clusterNick] = cluster

	context := clientcmdapi.NewContext()
	context.Cluster = clusterNick
	context.AuthInfo = userNick
	config.Contexts[contextNick] = context
	config.CurrentContext = contextNick

	return config
}
