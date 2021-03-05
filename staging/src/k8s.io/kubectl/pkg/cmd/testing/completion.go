/*
Copyright 2021 The Kubernetes Authors.

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

package testing

import (
	"io/ioutil"
	"net/http"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/spf13/cobra"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

// PreparePodsForCompletion prepares the test factory and streams for pods
func PreparePodsForCompletion(t *testing.T) (cmdutil.Factory, genericclioptions.IOStreams) {
	pods, _, _ := TestData()

	tf := NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: DefaultHeader(), Body: ObjBody(codec, pods)},
	}

	streams, _, _, _ := genericclioptions.NewTestIOStreams()
	return tf, streams
}

// PrepareNodesForCompletion prepares the test factory and streams for pods
func PrepareNodesForCompletion(t *testing.T) (cmdutil.Factory, genericclioptions.IOStreams) {
	// TODO create more than one node
	// nodes := &corev1.NodeList{
	// 	ListMeta: metav1.ListMeta{
	// 		ResourceVersion: "1",
	// 	},
	// 	Items: []corev1.Node{
	// 		{
	// 			ObjectMeta: metav1.ObjectMeta{
	// 				Name:              "firstnode",
	// 				CreationTimestamp: metav1.Time{Time: time.Now()},
	// 			},
	// 			Status: corev1.NodeStatus{},
	// 		},
	// 		{
	// 			ObjectMeta: metav1.ObjectMeta{
	// 				Name:              "secondnode",
	// 				CreationTimestamp: metav1.Time{Time: time.Now()},
	// 			},
	// 			Status: corev1.NodeStatus{},
	// 		},
	// 	},
	// }

	nodes := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "firstnode",
			CreationTimestamp: metav1.Time{Time: time.Now()},
		},
		Status: corev1.NodeStatus{},
	}

	tf := NewTestFactory()
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: DefaultHeader(), Body: ObjBody(codec, nodes)},
	}

	streams, _, _, _ := genericclioptions.NewTestIOStreams()
	return tf, streams
}

// PrepareConfigForCompletion prepares some contexts for completion testing
func PrepareConfigForCompletion(t *testing.T) (*clientcmd.PathOptions, genericclioptions.IOStreams, cmdutil.Factory) {
	conf := clientcmdapi.Config{
		Kind:       "Config",
		APIVersion: "v1",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube-cluster": {Server: "https://192.168.99.100:8443"},
			"my-cluster":       {Server: "https://192.168.0.1:3434"},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"minikube-context": {AuthInfo: "minikube", Cluster: "minikube"},
			"my-context":       {AuthInfo: "my-context", Cluster: "my-context"},
		},
		CurrentContext: "minikube-context",
	}

	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(conf, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""

	streams, _, _, _ := genericclioptions.NewTestIOStreams()
	factory := cmdutil.NewFactory(genericclioptions.NewTestConfigFlags())

	return pathOptions, streams, factory
}

// CheckCompletion checks that the directive is correct and that each completion is present
func CheckCompletion(t *testing.T, comps, expectedComps []string, directive, expectedDirective cobra.ShellCompDirective) {
	if e, d := expectedDirective, directive; e != d {
		t.Errorf("expected directive\n%v\nbut got\n%v", e, d)
	}

	sort.Strings(comps)
	sort.Strings(expectedComps)

	if len(expectedComps) != len(comps) {
		t.Fatalf("expected completions\n%v\nbut got\n%v", expectedComps, comps)
	}

	for i := range comps {
		if expectedComps[i] != comps[i] {
			t.Errorf("expected completions\n%v\nbut got\n%v", expectedComps, comps)
			break
		}
	}
}
