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
	"fmt"
	"io/ioutil"
	"strings"
	"testing"
	"time"

	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	externalclientset "k8s.io/client-go/kubernetes"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csiclientset "k8s.io/csi-api/pkg/client/clientset/versioned"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

func TestNodeAuthorizer(t *testing.T) {
	const (
		// Define credentials
		tokenMaster      = "master-token"
		tokenNodeUnknown = "unknown-token"
		tokenNode1       = "node1-token"
		tokenNode2       = "node2-token"
	)

	// Enabled CSIPersistentVolume feature at startup so volumeattachments get watched
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIPersistentVolume, true)()

	// Enable DynamicKubeletConfig feature so that Node.Spec.ConfigSource can be set
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicKubeletConfig, true)()

	// Enable NodeLease feature so that nodes can create leases
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeLease, true)()

	// Enable CSINodeInfo feature so that nodes can create CSINodeInfo objects.
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeInfo, true)()

	tokenFile, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	tokenFile.WriteString(strings.Join([]string{
		fmt.Sprintf(`%s,admin,uid1,"system:masters"`, tokenMaster),
		fmt.Sprintf(`%s,unknown,uid2,"system:nodes"`, tokenNodeUnknown),
		fmt.Sprintf(`%s,system:node:node1,uid3,"system:nodes"`, tokenNode1),
		fmt.Sprintf(`%s,system:node:node2,uid4,"system:nodes"`, tokenNode2),
	}, "\n"))
	tokenFile.Close()

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--authorization-mode", "Node,RBAC",
		"--token-auth-file", tokenFile.Name(),
		"--enable-admission-plugins", "NodeRestriction",
		// The "default" SA is not installed, causing the ServiceAccount plugin to retry for ~1s per
		// API request.
		"--disable-admission-plugins", "ServiceAccount,TaintNodesByCondition",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// Build client config and superuser clientset
	clientConfig := server.ClientConfig
	superuserClient, superuserClientExternal := clientsetForToken(tokenMaster, clientConfig)
	superuserCRDClient := crdClientsetForToken(tokenMaster, clientConfig)

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
	if _, err := superuserClient.Core().Namespaces().Create(&core.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns"}}); err != nil {
		t.Fatal(err)
	}

	if _, err := superuserClient.Core().Secrets("ns").Create(&api.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mysecret"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().Secrets("ns").Create(&api.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mypvsecret"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().ConfigMaps("ns").Create(&api.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "myconfigmap"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.Core().ConfigMaps("ns").Create(&api.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "myconfigmapconfigsource"}}); err != nil {
		t.Fatal(err)
	}
	pvName := "mypv"
	if _, err := superuserClientExternal.StorageV1beta1().VolumeAttachments().Create(&storagev1beta1.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{Name: "myattachment"},
		Spec: storagev1beta1.VolumeAttachmentSpec{
			Attacher: "foo",
			Source:   storagev1beta1.VolumeAttachmentSource{PersistentVolumeName: &pvName},
			NodeName: "node2",
		},
	}); err != nil {
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
			PersistentVolumeSource: api.PersistentVolumeSource{AzureFile: &api.AzureFilePersistentVolumeSource{ShareName: "default", SecretName: "mypvsecret"}},
		},
	}); err != nil {
		t.Fatal(err)
	}

	csiNodeInfoCRD, err := crdFromManifest("../../../staging/src/k8s.io/csi-api/pkg/crd/manifests/csinodeinfo.yaml")
	if err != nil {
		t.Fatal(err)
	}

	etcd.CreateTestCRDs(t, superuserCRDClient, false, csiNodeInfoCRD)

	getSecret := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().Secrets("ns").Get("mysecret", metav1.GetOptions{})
			return err
		}
	}
	getPVSecret := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().Secrets("ns").Get("mypvsecret", metav1.GetOptions{})
			return err
		}
	}
	getConfigMap := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().ConfigMaps("ns").Get("myconfigmap", metav1.GetOptions{})
			return err
		}
	}
	getConfigMapConfigSource := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().ConfigMaps("ns").Get("myconfigmapconfigsource", metav1.GetOptions{})
			return err
		}
	}
	getPVC := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().PersistentVolumeClaims("ns").Get("mypvc", metav1.GetOptions{})
			return err
		}
	}
	getPV := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().PersistentVolumes().Get("mypv", metav1.GetOptions{})
			return err
		}
	}
	getVolumeAttachment := func(client externalclientset.Interface) func() error {
		return func() error {
			_, err := client.StorageV1beta1().VolumeAttachments().Get("myattachment", metav1.GetOptions{})
			return err
		}
	}

	createNode2NormalPod := func(client clientset.Interface) func() error {
		return func() error {
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
	}
	updateNode2NormalPodStatus := func(client clientset.Interface) func() error {
		return func() error {
			startTime := metav1.NewTime(time.Now())
			_, err := client.Core().Pods("ns").UpdateStatus(&api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "node2normalpod"},
				Status:     api.PodStatus{StartTime: &startTime},
			})
			return err
		}
	}
	deleteNode2NormalPod := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.Core().Pods("ns").Delete("node2normalpod", &metav1.DeleteOptions{GracePeriodSeconds: &zero})
		}
	}

	createNode2MirrorPod := func(client clientset.Interface) func() error {
		return func() error {
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
	}
	deleteNode2MirrorPod := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.Core().Pods("ns").Delete("node2mirrorpod", &metav1.DeleteOptions{GracePeriodSeconds: &zero})
		}
	}

	createNode2 := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().Nodes().Create(&api.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2"}})
			return err
		}
	}
	setNode2ConfigSource := func(client clientset.Interface) func() error {
		return func() error {
			node2, err := client.Core().Nodes().Get("node2", metav1.GetOptions{})
			if err != nil {
				return err
			}
			node2.Spec.ConfigSource = &api.NodeConfigSource{
				ConfigMap: &api.ConfigMapNodeConfigSource{
					Namespace:        "ns",
					Name:             "myconfigmapconfigsource",
					KubeletConfigKey: "kubelet",
				},
			}
			_, err = client.Core().Nodes().Update(node2)
			return err
		}
	}
	unsetNode2ConfigSource := func(client clientset.Interface) func() error {
		return func() error {
			node2, err := client.Core().Nodes().Get("node2", metav1.GetOptions{})
			if err != nil {
				return err
			}
			node2.Spec.ConfigSource = nil
			_, err = client.Core().Nodes().Update(node2)
			return err
		}
	}
	updateNode2Status := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Core().Nodes().UpdateStatus(&api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status:     api.NodeStatus{},
			})
			return err
		}
	}
	deleteNode2 := func(client clientset.Interface) func() error {
		return func() error {
			return client.Core().Nodes().Delete("node2", nil)
		}
	}
	createNode2NormalPodEviction := func(client clientset.Interface) func() error {
		return func() error {
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
	}
	createNode2MirrorPodEviction := func(client clientset.Interface) func() error {
		return func() error {
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
	}

	capacity := 50
	updatePVCCapacity := func(client clientset.Interface) func() error {
		return func() error {
			capacity++
			statusString := fmt.Sprintf("{\"status\": {\"capacity\": {\"storage\": \"%dG\"}}}", capacity)
			patchBytes := []byte(statusString)
			_, err := client.Core().PersistentVolumeClaims("ns").Patch("mypvc", types.StrategicMergePatchType, patchBytes, "status")
			return err
		}
	}

	updatePVCPhase := func(client clientset.Interface) func() error {
		return func() error {
			patchBytes := []byte(`{"status":{"phase": "Bound"}}`)
			_, err := client.Core().PersistentVolumeClaims("ns").Patch("mypvc", types.StrategicMergePatchType, patchBytes, "status")
			return err
		}
	}

	getNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.Coordination().Leases(api.NamespaceNodeLease).Get("node1", metav1.GetOptions{})
			return err
		}
	}
	node1LeaseDurationSeconds := int32(40)
	createNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			lease := &coordination.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
				Spec: coordination.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("node1"),
					LeaseDurationSeconds: pointer.Int32Ptr(node1LeaseDurationSeconds),
					RenewTime:            &metav1.MicroTime{Time: time.Now()},
				},
			}
			_, err := client.Coordination().Leases(api.NamespaceNodeLease).Create(lease)
			return err
		}
	}
	updateNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			lease, err := client.Coordination().Leases(api.NamespaceNodeLease).Get("node1", metav1.GetOptions{})
			if err != nil {
				return err
			}
			lease.Spec.RenewTime = &metav1.MicroTime{Time: time.Now()}
			_, err = client.Coordination().Leases(api.NamespaceNodeLease).Update(lease)
			return err
		}
	}
	patchNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			node1LeaseDurationSeconds++
			bs := []byte(fmt.Sprintf(`{"spec": {"leaseDurationSeconds": %d}}`, node1LeaseDurationSeconds))
			_, err := client.Coordination().Leases(api.NamespaceNodeLease).Patch("node1", types.StrategicMergePatchType, bs)
			return err
		}
	}
	deleteNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			return client.Coordination().Leases(api.NamespaceNodeLease).Delete("node1", &metav1.DeleteOptions{})
		}
	}

	getNode1CSINodeInfo := func(client csiclientset.Interface) func() error {
		return func() error {
			_, err := client.CsiV1alpha1().CSINodeInfos().Get("node1", metav1.GetOptions{})
			return err
		}
	}
	createNode1CSINodeInfo := func(client csiclientset.Interface) func() error {
		return func() error {
			nodeInfo := &csiv1alpha1.CSINodeInfo{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
				Spec: csiv1alpha1.CSINodeInfoSpec{
					Drivers: []csiv1alpha1.CSIDriverInfoSpec{
						{
							Name:         "com.example.csi/driver1",
							NodeID:       "com.example.csi/node1",
							TopologyKeys: []string{"com.example.csi/zone"},
						},
					},
				},
				Status: csiv1alpha1.CSINodeInfoStatus{
					Drivers: []csiv1alpha1.CSIDriverInfoStatus{
						{
							Name:                  "com.example.csi/driver1",
							Available:             true,
							VolumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
						},
					},
				},
			}
			_, err := client.CsiV1alpha1().CSINodeInfos().Create(nodeInfo)
			return err
		}
	}
	updateNode1CSINodeInfo := func(client csiclientset.Interface) func() error {
		return func() error {
			nodeInfo, err := client.CsiV1alpha1().CSINodeInfos().Get("node1", metav1.GetOptions{})
			if err != nil {
				return err
			}
			nodeInfo.Spec.Drivers = []csiv1alpha1.CSIDriverInfoSpec{
				{
					Name:         "com.example.csi/driver1",
					NodeID:       "com.example.csi/node1",
					TopologyKeys: []string{"com.example.csi/rack"},
				},
			}
			nodeInfo.Status.Drivers = []csiv1alpha1.CSIDriverInfoStatus{
				{
					Name:                  "com.example.csi/driver1",
					Available:             true,
					VolumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
				},
			}
			_, err = client.CsiV1alpha1().CSINodeInfos().Update(nodeInfo)
			return err
		}
	}
	patchNode1CSINodeInfo := func(client csiclientset.Interface) func() error {
		return func() error {
			bs := []byte(fmt.Sprintf(`{"csiDrivers": [ { "driver": "net.example.storage/driver2", "nodeID": "net.example.storage/node1", "topologyKeys": [ "net.example.storage/region" ] } ] }`))
			// StrategicMergePatch is unsupported by CRs. Falling back to MergePatch
			_, err := client.CsiV1alpha1().CSINodeInfos().Patch("node1", types.MergePatchType, bs)
			return err
		}
	}
	deleteNode1CSINodeInfo := func(client csiclientset.Interface) func() error {
		return func() error {
			return client.CsiV1alpha1().CSINodeInfos().Delete("node1", &metav1.DeleteOptions{})
		}
	}

	nodeanonClient, _ := clientsetForToken(tokenNodeUnknown, clientConfig)
	node1Client, node1ClientExternal := clientsetForToken(tokenNode1, clientConfig)
	node2Client, node2ClientExternal := clientsetForToken(tokenNode2, clientConfig)
	csiNode1Client := csiClientsetForToken(tokenNode1, clientConfig)
	csiNode2Client := csiClientsetForToken(tokenNode2, clientConfig)

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
	// self deletion is not allowed
	expectForbidden(t, deleteNode2(node2Client))
	// clean up node2
	expectAllowed(t, deleteNode2(superuserClient))

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

	// re-create a pod as an admin to add object references
	expectAllowed(t, createNode2NormalPod(superuserClient))

	// ExpandPersistentVolumes feature disabled
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandPersistentVolumes, false)()
	expectForbidden(t, updatePVCCapacity(node1Client))
	expectForbidden(t, updatePVCCapacity(node2Client))

	// ExpandPersistentVolumes feature enabled
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandPersistentVolumes, true)()
	expectForbidden(t, updatePVCCapacity(node1Client))
	expectAllowed(t, updatePVCCapacity(node2Client))
	expectForbidden(t, updatePVCPhase(node2Client))

	// Disabled CSIPersistentVolume feature
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIPersistentVolume, false)()
	expectForbidden(t, getVolumeAttachment(node1ClientExternal))
	expectForbidden(t, getVolumeAttachment(node2ClientExternal))
	// Enabled CSIPersistentVolume feature
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIPersistentVolume, true)()
	expectForbidden(t, getVolumeAttachment(node1ClientExternal))
	expectAllowed(t, getVolumeAttachment(node2ClientExternal))

	// create node2 again
	expectAllowed(t, createNode2(node2Client))
	// node2 can not set its own config source
	expectForbidden(t, setNode2ConfigSource(node2Client))
	// node2 can not access the configmap config source yet
	expectForbidden(t, getConfigMapConfigSource(node2Client))
	// superuser can access the configmap config source
	expectAllowed(t, getConfigMapConfigSource(superuserClient))
	// superuser can set node2's config source
	expectAllowed(t, setNode2ConfigSource(superuserClient))
	// node2 can now get the configmap assigned as its config source
	expectAllowed(t, getConfigMapConfigSource(node2Client))
	// superuser can unset node2's config source
	expectAllowed(t, unsetNode2ConfigSource(superuserClient))
	// node2 can no longer get the configmap after it is unassigned as its config source
	expectForbidden(t, getConfigMapConfigSource(node2Client))
	// clean up node2
	expectAllowed(t, deleteNode2(superuserClient))

	//TODO(mikedanese): integration test node restriction of TokenRequest

	// node1 allowed to operate on its own lease
	expectAllowed(t, createNode1Lease(node1Client))
	expectAllowed(t, getNode1Lease(node1Client))
	expectAllowed(t, updateNode1Lease(node1Client))
	expectAllowed(t, patchNode1Lease(node1Client))
	expectAllowed(t, deleteNode1Lease(node1Client))
	// node2 not allowed to operate on another node's lease
	expectForbidden(t, createNode1Lease(node2Client))
	expectForbidden(t, getNode1Lease(node2Client))
	expectForbidden(t, updateNode1Lease(node2Client))
	expectForbidden(t, patchNode1Lease(node2Client))
	expectForbidden(t, deleteNode1Lease(node2Client))

	// node1 allowed to operate on its own CSINodeInfo
	expectAllowed(t, createNode1CSINodeInfo(csiNode1Client))
	expectAllowed(t, getNode1CSINodeInfo(csiNode1Client))
	expectAllowed(t, updateNode1CSINodeInfo(csiNode1Client))
	expectAllowed(t, patchNode1CSINodeInfo(csiNode1Client))
	expectAllowed(t, deleteNode1CSINodeInfo(csiNode1Client))
	// node2 not allowed to operate on another node's CSINodeInfo
	expectForbidden(t, createNode1CSINodeInfo(csiNode2Client))
	expectForbidden(t, getNode1CSINodeInfo(csiNode2Client))
	expectForbidden(t, updateNode1CSINodeInfo(csiNode2Client))
	expectForbidden(t, patchNode1CSINodeInfo(csiNode2Client))
	expectForbidden(t, deleteNode1CSINodeInfo(csiNode2Client))
}

// expect executes a function a set number of times until it either returns the
// expected error or executes too many times. It returns if the retries timed
// out and the last error returned by the method.
func expect(t *testing.T, f func() error, wantErr func(error) bool) (timeout bool, lastErr error) {
	t.Helper()
	err := wait.PollImmediate(time.Second, 30*time.Second, func() (bool, error) {
		t.Helper()
		lastErr = f()
		if wantErr(lastErr) {
			return true, nil
		}
		t.Logf("unexpected response, will retry: %v", lastErr)
		return false, nil
	})
	return err == nil, lastErr
}

func expectForbidden(t *testing.T, f func() error) {
	t.Helper()
	if ok, err := expect(t, f, errors.IsForbidden); !ok {
		t.Errorf("Expected forbidden error, got %v", err)
	}
}

func expectNotFound(t *testing.T, f func() error) {
	t.Helper()
	if ok, err := expect(t, f, errors.IsNotFound); !ok {
		t.Errorf("Expected notfound error, got %v", err)
	}
}

func expectAllowed(t *testing.T, f func() error) {
	t.Helper()
	if ok, err := expect(t, f, func(e error) bool { return e == nil }); !ok {
		t.Errorf("Expected no error, got %v", err)
	}
}

// crdFromManifest reads a .json/yaml file and returns the CRD in it.
func crdFromManifest(filename string) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	var crd apiextensionsv1beta1.CustomResourceDefinition
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), data, &crd); err != nil {
		return nil, err
	}
	return &crd, nil
}
