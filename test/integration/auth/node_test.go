/*
Copyright 2017 The Kubernetes Authors.

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
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer"
	"k8s.io/kubernetes/plugin/pkg/admission/noderestriction"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestNodeAuthorizer(t *testing.T) {
	// Start the server so we know the address
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	// Build client config, clientset, and informers
	clientConfig := &restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	superuserClient := clientsetForUser("admin/system:masters", clientConfig)
	informerFactory := informers.NewSharedInformerFactory(superuserClient, time.Minute)

	// Set up Node+RBAC authorizer
	authorizerConfig := &authorizer.AuthorizationConfig{
		AuthorizationModes: []string{"Node", "RBAC"},
		InformerFactory:    informerFactory,
	}
	nodeRBACAuthorizer, err := authorizerConfig.New()
	if err != nil {
		t.Fatal(err)
	}
	defer bootstrappolicy.ClearClusterRoleBindingFilters()

	// Set up NodeRestriction admission
	nodeRestrictionAdmission := noderestriction.NewPlugin(nodeidentifier.NewDefaultNodeIdentifier())
	nodeRestrictionAdmission.SetInternalKubeClientSet(superuserClient)
	if err := nodeRestrictionAdmission.Validate(); err != nil {
		t.Fatal(err)
	}

	// Start the server
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authenticator = newFakeAuthenticator()
	masterConfig.GenericConfig.Authorizer = nodeRBACAuthorizer
	masterConfig.GenericConfig.AdmissionControl = nodeRestrictionAdmission
	_, _, closeFn := framework.RunAMasterUsingServer(masterConfig, apiServer, h)
	defer closeFn()

	// Start the informers
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	// Wait for a healthy server
	for {
		result := superuserClient.Core().RESTClient().Get().AbsPath("/healthz").Do()
		_, err := result.Raw()
		if err == nil {
			break
		}
		t.Log(err)
		time.Sleep(time.Second)
	}

	// Create objects
	if _, err := superuserClient.Core().Secrets("ns").Create(&api.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mysecret"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().Secrets("ns").Create(&api.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mypvsecret"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().ConfigMaps("ns").Create(&api.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "myconfigmap"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().PersistentVolumeClaims("ns").Create(&api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "mypvc"},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceStorage: resource.MustParse("1")}},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().PersistentVolumes().Create(&api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "mypv"},
		Spec: api.PersistentVolumeSpec{
			AccessModes:            []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
			Capacity:               api.ResourceList{api.ResourceStorage: resource.MustParse("1")},
			ClaimRef:               &api.ObjectReference{Namespace: "ns", Name: "mypvc"},
			PersistentVolumeSource: api.PersistentVolumeSource{AzureFile: &api.AzureFileVolumeSource{ShareName: "default", SecretName: "mypvsecret"}},
		},
	}); err != nil {
		t.Fatal(err)
	}

	getSecret := func(client clientset.Interface) error {
		_, err := client.Core().Secrets("ns").Get("mysecret", metav1.GetOptions{})
		return err
	}
	getPVSecret := func(client clientset.Interface) error {
		_, err := client.Core().Secrets("ns").Get("mypvsecret", metav1.GetOptions{})
		return err
	}
	getConfigMap := func(client clientset.Interface) error {
		_, err := client.Core().ConfigMaps("ns").Get("myconfigmap", metav1.GetOptions{})
		return err
	}
	getPVC := func(client clientset.Interface) error {
		_, err := client.Core().PersistentVolumeClaims("ns").Get("mypvc", metav1.GetOptions{})
		return err
	}
	getPV := func(client clientset.Interface) error {
		_, err := client.Core().PersistentVolumes().Get("mypv", metav1.GetOptions{})
		return err
	}

	createNode2NormalPod := func(client clientset.Interface) error {
		_, err := client.Core().Pods("ns").Create(&api.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "node2normalpod"},
			Spec: api.PodSpec{
				NodeName:   "node2",
				Containers: []api.Container{{Name: "image", Image: "busybox"}},
				Volumes: []api.Volume{
					{Name: "secret", VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "mysecret"}}},
					{Name: "cm", VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{LocalObjectReference: api.LocalObjectReference{Name: "myconfigmap"}}}},
					{Name: "pvc", VolumeSource: api.VolumeSource{PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc"}}},
				},
			},
		})
		return err
	}
	updateNode2NormalPodStatus := func(client clientset.Interface) error {
		startTime := metav1.NewTime(time.Now())
		_, err := client.Core().Pods("ns").UpdateStatus(&api.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "node2normalpod"},
			Status:     api.PodStatus{StartTime: &startTime},
		})
		return err
	}
	deleteNode2NormalPod := func(client clientset.Interface) error {
		zero := int64(0)
		return client.Core().Pods("ns").Delete("node2normalpod", &metav1.DeleteOptions{GracePeriodSeconds: &zero})
	}

	createNode2MirrorPod := func(client clientset.Interface) error {
		_, err := client.Core().Pods("ns").Create(&api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "node2mirrorpod",
				Annotations: map[string]string{api.MirrorPodAnnotationKey: "true"},
			},
			Spec: api.PodSpec{
				NodeName:   "node2",
				Containers: []api.Container{{Name: "image", Image: "busybox"}},
			},
		})
		return err
	}
	deleteNode2MirrorPod := func(client clientset.Interface) error {
		zero := int64(0)
		return client.Core().Pods("ns").Delete("node2mirrorpod", &metav1.DeleteOptions{GracePeriodSeconds: &zero})
	}

	createNode2 := func(client clientset.Interface) error {
		_, err := client.Core().Nodes().Create(&api.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2"}})
		return err
	}
	updateNode2Status := func(client clientset.Interface) error {
		_, err := client.Core().Nodes().UpdateStatus(&api.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "node2"},
			Status:     api.NodeStatus{},
		})
		return err
	}
	deleteNode2 := func(client clientset.Interface) error {
		return client.Core().Nodes().Delete("node2", nil)
	}
	createNode2NormalPodEviction := func(client clientset.Interface) error {
		return client.Policy().Evictions("ns").Evict(&policy.Eviction{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "policy/v1beta1",
				Kind:       "Eviction",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "node2normalpod",
				Namespace: "ns",
			},
		})
	}
	createNode2MirrorPodEviction := func(client clientset.Interface) error {
		return client.Policy().Evictions("ns").Evict(&policy.Eviction{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "policy/v1beta1",
				Kind:       "Eviction",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "node2mirrorpod",
				Namespace: "ns",
			},
		})
	}

	nodeanonClient := clientsetForUser("unknown/system:nodes", clientConfig)
	node1Client := clientsetForUser("system:node:node1/system:nodes", clientConfig)
	node2Client := clientsetForUser("system:node:node2/system:nodes", clientConfig)

	// all node requests from node1 and unknown node fail
	expectForbidden(t, getSecret(nodeanonClient))
	expectForbidden(t, getPVSecret(nodeanonClient))
	expectForbidden(t, getConfigMap(nodeanonClient))
	expectForbidden(t, getPVC(nodeanonClient))
	expectForbidden(t, getPV(nodeanonClient))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPod(nodeanonClient))
	expectForbidden(t, deleteNode2NormalPod(nodeanonClient))
	expectForbidden(t, deleteNode2MirrorPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPodEviction(nodeanonClient))
	expectForbidden(t, createNode2(nodeanonClient))
	expectForbidden(t, updateNode2Status(nodeanonClient))
	expectForbidden(t, deleteNode2(nodeanonClient))

	expectForbidden(t, getSecret(node1Client))
	expectForbidden(t, getPVSecret(node1Client))
	expectForbidden(t, getConfigMap(node1Client))
	expectForbidden(t, getPVC(node1Client))
	expectForbidden(t, getPV(node1Client))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPod(node1Client))
	expectNotFound(t, deleteNode2MirrorPod(node1Client))
	expectNotFound(t, createNode2MirrorPodEviction(node1Client))
	expectForbidden(t, createNode2(node1Client))
	expectForbidden(t, updateNode2Status(node1Client))
	expectForbidden(t, deleteNode2(node1Client))

	// related object requests from node2 fail
	expectForbidden(t, getSecret(node2Client))
	expectForbidden(t, getPVSecret(node2Client))
	expectForbidden(t, getConfigMap(node2Client))
	expectForbidden(t, getPVC(node2Client))
	expectForbidden(t, getPV(node2Client))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	// mirror pod and self node lifecycle is allowed
	expectAllowed(t, createNode2MirrorPod(node2Client))
	expectAllowed(t, deleteNode2MirrorPod(node2Client))
	expectAllowed(t, createNode2MirrorPod(node2Client))
	expectAllowed(t, createNode2MirrorPodEviction(node2Client))
	expectAllowed(t, createNode2(node2Client))
	expectAllowed(t, updateNode2Status(node2Client))
	expectAllowed(t, deleteNode2(node2Client))

	// create a pod as an admin to add object references
	expectAllowed(t, createNode2NormalPod(superuserClient))

	// unidentifiable node and node1 are still forbidden
	expectForbidden(t, getSecret(nodeanonClient))
	expectForbidden(t, getPVSecret(nodeanonClient))
	expectForbidden(t, getConfigMap(nodeanonClient))
	expectForbidden(t, getPVC(nodeanonClient))
	expectForbidden(t, getPV(nodeanonClient))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	expectForbidden(t, updateNode2NormalPodStatus(nodeanonClient))
	expectForbidden(t, deleteNode2NormalPod(nodeanonClient))
	expectForbidden(t, createNode2NormalPodEviction(nodeanonClient))
	expectForbidden(t, createNode2MirrorPod(nodeanonClient))
	expectForbidden(t, deleteNode2MirrorPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPodEviction(nodeanonClient))

	expectForbidden(t, getSecret(node1Client))
	expectForbidden(t, getPVSecret(node1Client))
	expectForbidden(t, getConfigMap(node1Client))
	expectForbidden(t, getPVC(node1Client))
	expectForbidden(t, getPV(node1Client))
	expectForbidden(t, createNode2NormalPod(node1Client))
	expectForbidden(t, updateNode2NormalPodStatus(node1Client))
	expectForbidden(t, deleteNode2NormalPod(node1Client))
	expectForbidden(t, createNode2NormalPodEviction(node1Client))
	expectForbidden(t, createNode2MirrorPod(node1Client))
	expectNotFound(t, deleteNode2MirrorPod(node1Client))
	expectNotFound(t, createNode2MirrorPodEviction(node1Client))

	// node2 can get referenced objects now
	expectAllowed(t, getSecret(node2Client))
	expectAllowed(t, getPVSecret(node2Client))
	expectAllowed(t, getConfigMap(node2Client))
	expectAllowed(t, getPVC(node2Client))
	expectAllowed(t, getPV(node2Client))
	expectForbidden(t, createNode2NormalPod(node2Client))
	expectAllowed(t, updateNode2NormalPodStatus(node2Client))
	expectAllowed(t, deleteNode2NormalPod(node2Client))
	expectAllowed(t, createNode2MirrorPod(node2Client))
	expectAllowed(t, deleteNode2MirrorPod(node2Client))
	// recreate as an admin to test eviction
	expectAllowed(t, createNode2NormalPod(superuserClient))
	expectAllowed(t, createNode2MirrorPod(superuserClient))
	expectAllowed(t, createNode2NormalPodEviction(node2Client))
	expectAllowed(t, createNode2MirrorPodEviction(node2Client))
}

func expectForbidden(t *testing.T, err error) {
	if !errors.IsForbidden(err) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("%s:%d: Expected forbidden error, got %v", filepath.Base(file), line, err)
	}
}

func expectNotFound(t *testing.T, err error) {
	if !errors.IsNotFound(err) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("%s:%d: Expected notfound error, got %v", filepath.Base(file), line, err)
	}
}

func expectAllowed(t *testing.T, err error) {
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("%s:%d: Expected no error, got %v", filepath.Base(file), line, err)
	}
}
