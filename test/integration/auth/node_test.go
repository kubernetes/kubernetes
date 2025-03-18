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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	authenticationv1 "k8s.io/api/authentication/v1"
	coordination "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanagertesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/kubeconfig"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

func TestNodeAuthorizer(t *testing.T) {
	const (
		// Define credentials
		// Fake values for testing.
		tokenMaster      = "master-token"
		tokenNodeUnknown = "unknown-token"
		tokenNode1       = "node1-token"
		tokenNode2       = "node2-token"
	)

	tokenFile, err := os.CreateTemp("", "kubeconfig")
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

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--runtime-config=api/all=true",
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

	// Create objects
	if _, err := superuserClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if _, err := superuserClient.CoreV1().Secrets("ns").Create(context.TODO(), &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mysecret"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.CoreV1().Secrets("ns").Create(context.TODO(), &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "mypvsecret"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.CoreV1().ConfigMaps("ns").Create(context.TODO(), &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "myconfigmap"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1beta1().ResourceClaims("ns").Create(context.TODO(), &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "mynamedresourceclaim"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1beta1().ResourceClaims("ns").Create(context.TODO(), &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "mytemplatizedresourceclaim"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1beta1().ResourceSlices().Create(context.TODO(), &resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "myslice1"}, Spec: resourceapi.ResourceSliceSpec{NodeName: "node1", Driver: "dra.example.com", Pool: resourceapi.ResourcePool{Name: "node1-slice", ResourceSliceCount: 1}}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1beta1().ResourceSlices().Create(context.TODO(), &resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "myslice2"}, Spec: resourceapi.ResourceSliceSpec{NodeName: "node2", Driver: "dra.example.com", Pool: resourceapi.ResourcePool{Name: "node2-slice", ResourceSliceCount: 1}}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	pvName := "mypv"
	if _, err := superuserClientExternal.StorageV1().VolumeAttachments().Create(context.TODO(), &storagev1.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{Name: "myattachment"},
		Spec: storagev1.VolumeAttachmentSpec{
			Attacher: "foo",
			Source:   storagev1.VolumeAttachmentSource{PersistentVolumeName: &pvName},
			NodeName: "node2",
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.CoreV1().PersistentVolumeClaims("ns").Create(context.TODO(), &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "mypvc"},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if _, err := superuserClient.CoreV1().PersistentVolumes().Create(context.TODO(), &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "mypv"},
		Spec: corev1.PersistentVolumeSpec{
			AccessModes:            []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Capacity:               corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")},
			ClaimRef:               &corev1.ObjectReference{Namespace: "ns", Name: "mypvc"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{AzureFile: &corev1.AzureFilePersistentVolumeSource{ShareName: "default", SecretName: "mypvsecret"}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	getSecret := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Secrets("ns").Get(context.TODO(), "mysecret", metav1.GetOptions{})
			return err
		}
	}
	getPVSecret := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Secrets("ns").Get(context.TODO(), "mypvsecret", metav1.GetOptions{})
			return err
		}
	}
	getConfigMap := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().ConfigMaps("ns").Get(context.TODO(), "myconfigmap", metav1.GetOptions{})
			return err
		}
	}
	getPVC := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().PersistentVolumeClaims("ns").Get(context.TODO(), "mypvc", metav1.GetOptions{})
			return err
		}
	}
	getPV := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().PersistentVolumes().Get(context.TODO(), "mypv", metav1.GetOptions{})
			return err
		}
	}
	getVolumeAttachment := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.StorageV1().VolumeAttachments().Get(context.TODO(), "myattachment", metav1.GetOptions{})
			return err
		}
	}
	getResourceClaim := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.ResourceV1beta1().ResourceClaims("ns").Get(context.TODO(), "mynamedresourceclaim", metav1.GetOptions{})
			return err
		}
	}
	getResourceClaimTemplate := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.ResourceV1beta1().ResourceClaims("ns").Get(context.TODO(), "mytemplatizedresourceclaim", metav1.GetOptions{})
			return err
		}
	}
	deleteResourceSliceCollection := func(client clientset.Interface, nodeName *string) func() error {
		return func() error {
			var listOptions metav1.ListOptions
			if nodeName != nil {
				listOptions.FieldSelector = resourceapi.ResourceSliceSelectorNodeName + "=" + *nodeName
			}
			return client.ResourceV1beta1().ResourceSlices().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, listOptions)
		}
	}
	addResourceClaimTemplateReference := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Pods("ns").Patch(context.TODO(), "node2normalpod", types.MergePatchType,
				[]byte(`{"status":{"resourceClaimStatuses":[{"name":"templateclaim","resourceClaimName":"mytemplatizedresourceclaim"}]}}`),
				metav1.PatchOptions{}, "status")
			return err
		}
	}
	removeResourceClaimReference := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Pods("ns").Patch(context.TODO(), "node2normalpod", types.MergePatchType,
				[]byte(`{"status":{"resourceClaimStatuses":null}}`),
				metav1.PatchOptions{}, "status")
			return err
		}
	}

	createNode2NormalPod := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Pods("ns").Create(context.TODO(), &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "node2normalpod"},
				Spec: corev1.PodSpec{
					NodeName:   "node2",
					Containers: []corev1.Container{{Name: "image", Image: "busybox"}},
					Volumes: []corev1.Volume{
						{Name: "secret", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "mysecret"}}},
						{Name: "cm", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "myconfigmap"}}}},
						{Name: "pvc", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc"}}},
					},
					ResourceClaims: []corev1.PodResourceClaim{
						{Name: "namedclaim", ResourceClaimName: pointer.String("mynamedresourceclaim")},
						{Name: "templateclaim", ResourceClaimTemplateName: pointer.String("myresourceclaimtemplate")},
					},
				},
			}, metav1.CreateOptions{})
			return err
		}
	}
	updateNode2NormalPodStatus := func(client clientset.Interface) func() error {
		return func() error {
			startTime := metav1.NewTime(time.Now())
			_, err := client.CoreV1().Pods("ns").UpdateStatus(context.TODO(), &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "node2normalpod"},
				Status:     corev1.PodStatus{StartTime: &startTime},
			}, metav1.UpdateOptions{})
			return err
		}
	}
	deleteNode2NormalPod := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.CoreV1().Pods("ns").Delete(context.TODO(), "node2normalpod", metav1.DeleteOptions{GracePeriodSeconds: &zero})
		}
	}

	createNode2MirrorPod := func(client clientset.Interface) func() error {
		return func() error {
			const nodeName = "node2"
			node, err := client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			controller := true
			_, err = client.CoreV1().Pods("ns").Create(context.TODO(), &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "node2mirrorpod",
					Annotations: map[string]string{corev1.MirrorPodAnnotationKey: "true"},
					OwnerReferences: []metav1.OwnerReference{{
						APIVersion: corev1.SchemeGroupVersion.String(),
						Kind:       "Node",
						Name:       nodeName,
						UID:        node.UID,
						Controller: &controller,
					}},
				},
				Spec: corev1.PodSpec{
					NodeName:   nodeName,
					Containers: []corev1.Container{{Name: "image", Image: "busybox"}},
				},
			}, metav1.CreateOptions{})
			return err
		}
	}
	deleteNode2MirrorPod := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.CoreV1().Pods("ns").Delete(context.TODO(), "node2mirrorpod", metav1.DeleteOptions{GracePeriodSeconds: &zero})
		}
	}

	createNode2 := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Nodes().Create(context.TODO(), &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2"}}, metav1.CreateOptions{})
			return err
		}
	}
	updateNode2Status := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().Nodes().UpdateStatus(context.TODO(), &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status:     corev1.NodeStatus{},
			}, metav1.UpdateOptions{})
			return err
		}
	}
	deleteNode2 := func(client clientset.Interface) func() error {
		return func() error {
			return client.CoreV1().Nodes().Delete(context.TODO(), "node2", metav1.DeleteOptions{})
		}
	}
	createNode2NormalPodEviction := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.PolicyV1().Evictions("ns").Evict(context.TODO(), &policy.Eviction{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "policy/v1",
					Kind:       "Eviction",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "node2normalpod",
					Namespace: "ns",
				},
				DeleteOptions: &metav1.DeleteOptions{GracePeriodSeconds: &zero},
			})
		}
	}
	createNode2MirrorPodEviction := func(client clientset.Interface) func() error {
		return func() error {
			zero := int64(0)
			return client.PolicyV1().Evictions("ns").Evict(context.TODO(), &policy.Eviction{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "policy/v1",
					Kind:       "Eviction",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "node2mirrorpod",
					Namespace: "ns",
				},
				DeleteOptions: &metav1.DeleteOptions{GracePeriodSeconds: &zero},
			})
		}
	}

	capacity := 50
	updatePVCCapacity := func(client clientset.Interface) func() error {
		return func() error {
			capacity++
			statusString := fmt.Sprintf("{\"status\": {\"capacity\": {\"storage\": \"%dG\"}}}", capacity)
			patchBytes := []byte(statusString)
			_, err := client.CoreV1().PersistentVolumeClaims("ns").Patch(context.TODO(), "mypvc", types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
			return err
		}
	}

	updatePVCPhase := func(client clientset.Interface) func() error {
		return func() error {
			patchBytes := []byte(`{"status":{"phase": "Bound"}}`)
			_, err := client.CoreV1().PersistentVolumeClaims("ns").Patch(context.TODO(), "mypvc", types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
			return err
		}
	}

	getNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Get(context.TODO(), "node1", metav1.GetOptions{})
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
					HolderIdentity:       pointer.String("node1"),
					LeaseDurationSeconds: pointer.Int32(node1LeaseDurationSeconds),
					RenewTime:            &metav1.MicroTime{Time: time.Now()},
				},
			}
			_, err := client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Create(context.TODO(), lease, metav1.CreateOptions{})
			return err
		}
	}
	updateNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			lease, err := client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Get(context.TODO(), "node1", metav1.GetOptions{})
			if err != nil {
				return err
			}
			lease.Spec.RenewTime = &metav1.MicroTime{Time: time.Now()}
			_, err = client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Update(context.TODO(), lease, metav1.UpdateOptions{})
			return err
		}
	}
	patchNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			node1LeaseDurationSeconds++
			bs := []byte(fmt.Sprintf(`{"spec": {"leaseDurationSeconds": %d}}`, node1LeaseDurationSeconds))
			_, err := client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Patch(context.TODO(), "node1", types.StrategicMergePatchType, bs, metav1.PatchOptions{})
			return err
		}
	}
	deleteNode1Lease := func(client clientset.Interface) func() error {
		return func() error {
			return client.CoordinationV1().Leases(corev1.NamespaceNodeLease).Delete(context.TODO(), "node1", metav1.DeleteOptions{})
		}
	}

	getNode1CSINode := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.StorageV1().CSINodes().Get(context.TODO(), "node1", metav1.GetOptions{})
			return err
		}
	}
	createNode1CSINode := func(client clientset.Interface) func() error {
		return func() error {
			nodeInfo := &storagev1.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node1",
				},
				Spec: storagev1.CSINodeSpec{
					Drivers: []storagev1.CSINodeDriver{
						{
							Name:         "com.example.csi.driver1",
							NodeID:       "com.example.csi/node1",
							TopologyKeys: []string{"com.example.csi/zone"},
						},
					},
				},
			}
			_, err := client.StorageV1().CSINodes().Create(context.TODO(), nodeInfo, metav1.CreateOptions{})
			return err
		}
	}
	updateNode1CSINode := func(client clientset.Interface) func() error {
		return func() error {
			nodeInfo, err := client.StorageV1().CSINodes().Get(context.TODO(), "node1", metav1.GetOptions{})
			if err != nil {
				return err
			}
			nodeInfo.Spec.Drivers = []storagev1.CSINodeDriver{
				{
					Name:         "com.example.csi.driver2",
					NodeID:       "com.example.csi/node1",
					TopologyKeys: []string{"com.example.csi/rack"},
				},
			}
			_, err = client.StorageV1().CSINodes().Update(context.TODO(), nodeInfo, metav1.UpdateOptions{})
			return err
		}
	}
	patchNode1CSINode := func(client clientset.Interface) func() error {
		return func() error {
			bs := []byte(fmt.Sprintf(`{"csiDrivers": [ { "driver": "net.example.storage.driver2", "nodeID": "net.example.storage/node1", "topologyKeys": [ "net.example.storage/region" ] } ] }`))
			// StrategicMergePatch is unsupported by CRs. Falling back to MergePatch
			_, err := client.StorageV1().CSINodes().Patch(context.TODO(), "node1", types.MergePatchType, bs, metav1.PatchOptions{})
			return err
		}
	}
	deleteNode1CSINode := func(client clientset.Interface) func() error {
		return func() error {
			return client.StorageV1().CSINodes().Delete(context.TODO(), "node1", metav1.DeleteOptions{})
		}
	}

	nodeanonClient, _ := clientsetForToken(tokenNodeUnknown, clientConfig)
	node1Client, node1ClientExternal := clientsetForToken(tokenNode1, clientConfig)
	node2Client, node2ClientExternal := clientsetForToken(tokenNode2, clientConfig)
	_, csiNode1Client := clientsetForToken(tokenNode1, clientConfig)
	_, csiNode2Client := clientsetForToken(tokenNode2, clientConfig)

	// all node requests from node1 and unknown node fail
	expectForbidden(t, getSecret(nodeanonClient))
	expectForbidden(t, getPVSecret(nodeanonClient))
	expectForbidden(t, getConfigMap(nodeanonClient))
	expectForbidden(t, getPVC(nodeanonClient))
	expectForbidden(t, getPV(nodeanonClient))
	expectForbidden(t, getResourceClaim(nodeanonClient))
	expectForbidden(t, getResourceClaimTemplate(nodeanonClient))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	expectForbidden(t, deleteNode2NormalPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPodEviction(nodeanonClient))
	expectForbidden(t, createNode2(nodeanonClient))
	expectForbidden(t, updateNode2Status(nodeanonClient))
	expectForbidden(t, deleteNode2(nodeanonClient))

	expectForbidden(t, getSecret(node1Client))
	expectForbidden(t, getPVSecret(node1Client))
	expectForbidden(t, getConfigMap(node1Client))
	expectForbidden(t, getPVC(node1Client))
	expectForbidden(t, getPV(node1Client))
	expectForbidden(t, getResourceClaim(node1Client))
	expectForbidden(t, getResourceClaimTemplate(node1Client))
	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	expectNotFound(t, createNode2MirrorPodEviction(node1Client))
	expectForbidden(t, createNode2(node1Client))
	expectNotFound(t, updateNode2Status(node1Client))
	expectForbidden(t, deleteNode2(node1Client))

	// related object requests from node2 fail
	expectForbidden(t, getSecret(node2Client))
	expectForbidden(t, getPVSecret(node2Client))
	expectForbidden(t, getConfigMap(node2Client))
	expectForbidden(t, getPVC(node2Client))
	expectForbidden(t, getPV(node2Client))
	expectForbidden(t, getResourceClaim(node2Client))
	expectForbidden(t, getResourceClaimTemplate(node2Client))

	expectForbidden(t, createNode2NormalPod(nodeanonClient))
	// mirror pod and self node lifecycle is allowed
	expectAllowed(t, createNode2(node2Client))
	expectAllowed(t, updateNode2Status(node2Client))
	expectForbidden(t, createNode2MirrorPod(nodeanonClient))
	expectForbidden(t, deleteNode2MirrorPod(nodeanonClient))
	expectForbidden(t, createNode2MirrorPod(node1Client))
	expectNotFound(t, deleteNode2MirrorPod(node1Client))
	// create a pod as an admin to add object references
	expectAllowed(t, createNode2NormalPod(superuserClient))

	expectAllowed(t, createNode2MirrorPod(node2Client))
	expectAllowed(t, deleteNode2MirrorPod(node2Client))
	expectAllowed(t, createNode2MirrorPod(node2Client))
	expectAllowed(t, createNode2MirrorPodEviction(node2Client))
	// self deletion is not allowed
	expectForbidden(t, deleteNode2(node2Client))
	// modification of another node's status is not allowed
	expectForbidden(t, updateNode2Status(node1Client))

	// unidentifiable node and node1 are still forbidden
	expectForbidden(t, getSecret(nodeanonClient))
	expectForbidden(t, getPVSecret(nodeanonClient))
	expectForbidden(t, getConfigMap(nodeanonClient))
	expectForbidden(t, getPVC(nodeanonClient))
	expectForbidden(t, getPV(nodeanonClient))
	expectForbidden(t, getResourceClaim(nodeanonClient))
	expectForbidden(t, getResourceClaimTemplate(nodeanonClient))
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
	expectForbidden(t, getResourceClaim(node1Client))
	expectForbidden(t, getResourceClaimTemplate(node1Client))
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

	// node2 can only get direct claim references
	expectAllowed(t, getResourceClaim(node2Client))
	expectForbidden(t, getResourceClaimTemplate(node2Client))

	// node cannot add a claim reference
	expectForbidden(t, addResourceClaimTemplateReference(node2Client))
	// superuser can add a claim reference
	expectAllowed(t, addResourceClaimTemplateReference(superuserClient))
	// node can get direct and template claim references
	expectAllowed(t, getResourceClaim(node2Client))
	expectAllowed(t, getResourceClaimTemplate(node2Client))

	// node cannot remove a claim reference
	expectForbidden(t, removeResourceClaimReference(node2Client))
	// superuser can remove a claim reference
	expectAllowed(t, removeResourceClaimReference(superuserClient))
	// node2 can only get direct claim references
	expectAllowed(t, getResourceClaim(node2Client))
	expectForbidden(t, getResourceClaimTemplate(node2Client))

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
	// clean up node2
	expectAllowed(t, deleteNode2(superuserClient))

	// re-create a pod as an admin to add object references
	expectAllowed(t, createNode2NormalPod(superuserClient))

	expectForbidden(t, updatePVCCapacity(node1Client))
	expectAllowed(t, updatePVCCapacity(node2Client))
	expectForbidden(t, updatePVCPhase(node2Client))

	// Enabled CSIPersistentVolume feature
	expectForbidden(t, getVolumeAttachment(node1ClientExternal))
	expectAllowed(t, getVolumeAttachment(node2ClientExternal))

	// create node2 again
	expectAllowed(t, createNode2(node2Client))
	// clean up node2
	expectAllowed(t, deleteNode2(superuserClient))

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

	// node1 allowed to operate on its own CSINode
	expectAllowed(t, createNode1CSINode(csiNode1Client))
	expectAllowed(t, getNode1CSINode(csiNode1Client))
	expectAllowed(t, updateNode1CSINode(csiNode1Client))
	expectAllowed(t, patchNode1CSINode(csiNode1Client))
	expectAllowed(t, deleteNode1CSINode(csiNode1Client))
	// node2 not allowed to operate on another node's CSINode
	expectForbidden(t, createNode1CSINode(csiNode2Client))
	expectForbidden(t, getNode1CSINode(csiNode2Client))
	expectForbidden(t, updateNode1CSINode(csiNode2Client))
	expectForbidden(t, patchNode1CSINode(csiNode2Client))
	expectForbidden(t, deleteNode1CSINode(csiNode2Client))

	// Always allowed. Permission to delete specific objects is checked per object.
	// Beware, this is destructive!
	expectAllowed(t, deleteResourceSliceCollection(csiNode1Client, ptr.To("node1")))

	// One slice must have been deleted, the other not.
	slices, err := superuserClient.ResourceV1beta1().ResourceSlices().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(slices.Items) != 1 {
		t.Fatalf("unexpected slices: %v", slices.Items)
	}
	if slices.Items[0].Spec.NodeName != "node2" {
		t.Fatal("wrong slice deleted")
	}

	// Superuser can delete.
	expectAllowed(t, deleteResourceSliceCollection(superuserClient, nil))
	slices, err = superuserClient.ResourceV1beta1().ResourceSlices().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(slices.Items) != 0 {
		t.Fatalf("unexpected slices: %v", slices.Items)
	}
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
	if ok, err := expect(t, f, apierrors.IsForbidden); !ok {
		t.Errorf("Expected forbidden error, got %v", err)
	}
}

func expectNotFound(t *testing.T, f func() error) {
	t.Helper()
	if ok, err := expect(t, f, apierrors.IsNotFound); !ok {
		t.Errorf("Expected notfound error, got %v", err)
	}
}

func expectAllowed(t *testing.T, f func() error) {
	t.Helper()
	if ok, err := expect(t, f, func(e error) bool { return e == nil }); !ok {
		t.Errorf("Expected no error, got %v", err)
	}
}

func checkNilError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func expectedForbiddenMessage(t *testing.T, f func() error, expectedMessage string) {
	t.Helper()
	if ok, err := expect(t, f, func(e error) bool { return apierrors.IsForbidden(e) && strings.Contains(e.Error(), expectedMessage) }); !ok {
		t.Errorf("Expected forbidden error with message %q, got %v", expectedMessage, err)
	}
}

// TestNodeRestrictionServiceAccount is an integration test to verify that
// the NodeRestriction admission plugin
// - forbids kubelet to request a token for a service account token that's not bound to a pod
// - forbids kubelet to request a token for a service account token when pod is not found
// - kubelet successfully requests a token for a service account token when pod is found and uid matches
// This test is run with the ServiceAccountNodeAudienceRestriction feature disabled
// to validate the default behavior of the NodeRestriction admission plugin.
func TestNodeRestrictionServiceAccount(t *testing.T) {
	const (
		// Define credentials
		// Fake values for testing.
		tokenMaster = "master-token"
		tokenNode1  = "node1-token"
		tokenNode2  = "node2-token"
	)

	tokenFile, err := os.CreateTemp("", "kubeconfig")
	checkNilError(t, err)

	_, err = tokenFile.WriteString(strings.Join([]string{
		fmt.Sprintf(`%s,admin,uid1,"system:masters"`, tokenMaster),
		fmt.Sprintf(`%s,system:node:node1,uid3,"system:nodes"`, tokenNode1),
		fmt.Sprintf(`%s,system:node:node2,uid4,"system:nodes"`, tokenNode2),
	}, "\n"))
	checkNilError(t, err)
	checkNilError(t, tokenFile.Close())

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountNodeAudienceRestriction, false)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--runtime-config=api/all=true",
		"--authorization-mode", "Node,RBAC",
		"--token-auth-file", tokenFile.Name(),
		"--enable-admission-plugins", "NodeRestriction",
		"--disable-admission-plugins", "ServiceAccount,TaintNodesByCondition",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// Build client config and superuser clientset
	clientConfig := server.ClientConfig
	superuserClient, _ := clientsetForToken(tokenMaster, clientConfig)

	if _, err := superuserClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// RBAC permissions are required to test service account token requests
	// using the node client because we cannot rely on the node authorizer since we have not
	// configured the references needed for it to work.  This test is focused on exercising
	// the node admission logic, not the node authorizer logic.  We want to know what happens
	// in admission if the request was already authorized.
	configureRBACForServiceAccountToken(t, superuserClient)

	createTokenRequestNodeNotBoundToPod := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.CoreV1().ServiceAccounts("ns").CreateToken(context.TODO(), "default", &authenticationv1.TokenRequest{}, metav1.CreateOptions{})
			return err
		}
	}

	node1Client, _ := clientsetForToken(tokenNode1, clientConfig)
	createNode(t, node1Client, "node1")
	node2Client, _ := clientsetForToken(tokenNode2, clientConfig)

	t.Run("service account token request is forbidden when not bound to a pod", func(t *testing.T) {
		expectedForbiddenMessage(t, createTokenRequestNodeNotBoundToPod(node1Client), "node requested token not bound to a pod")
	})

	t.Run("service account token request is forbidden when pod is not found", func(t *testing.T) {
		expectNotFound(t, createTokenRequest(node1Client, "uid1", tokenRequestWithAudiences("")))
	})

	t.Run("uid in token request does not match pod uid is forbidden", func(t *testing.T) {
		createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, "random-uid", tokenRequestWithAudiences("")), "the UID in the bound object reference (random-uid) does not match the UID in record. The object might have been deleted and then recreated")
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("node requesting token for pod bound to different node is forbidden", func(t *testing.T) {
		pod := createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node2Client, pod.UID, tokenRequestWithAudiences("")), "node requested token bound to a pod scheduled on a different node")
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("service account token request is successful", func(t *testing.T) {
		// create a pod as an admin to add object references
		pod := createPod(t, superuserClient, nil)
		createServiceAccount(t, superuserClient, "ns", "default")
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")))
	})
}

// TestNodeRestrictionServiceAccountAudience is an integration test to verify that
// the NodeRestriction admission plugin
// - allows kubelet to request a token for a service account that's in the pod spec
//  1. pod --> ephemeral --> pvc --> pv --> csi --> driver --> tokenrequest with audience
//  2. pod --> pvc --> pv --> csi --> driver --> tokenrequest with audience
//  3. pod --> csi --> driver --> tokenrequest with audience
//  4. pod --> projected --> service account token with audience
//
// - allows kubelet to request a token for a service account based on subjectacccessreview
//  1. clusterrole and clusterrolebinding allowing a node to request specific audience for a service account
//  2. clusterrole and rolebinding allowing a node to request specific audience for a service account
//  3. role and rolebinding allowing a node to request specific audience for a service account in a namespace
//  4. clusterrole and rolebinding allowing a node to request specific audience for any service account in a namespace
//  5. clusterrole and rolebinding allowing any node to request any audience for a service account in a namespace
//  6. clusterrole and rolebinding allowing any node to request any audience for any service account in a namespace
//  7. api server audience "" configured with rbac role
//
// - forbids kubelet to request a token for a service account that's not in the pod spec
// when the ServiceAccountNodeAudienceRestriction feature is enabled.
func TestNodeRestrictionServiceAccountAudience(t *testing.T) {
	const (
		// Define credentials
		// Fake values for testing.
		tokenMaster = "master-token"
		tokenNode1  = "node1-token"
	)

	tokenFile, err := os.CreateTemp("", "kubeconfig")
	checkNilError(t, err)

	_, err = tokenFile.WriteString(strings.Join([]string{
		fmt.Sprintf(`%s,admin,uid1,"system:masters"`, tokenMaster),
		fmt.Sprintf(`%s,system:node:node1,uid3,"system:nodes"`, tokenNode1),
	}, "\n"))
	checkNilError(t, err)
	checkNilError(t, tokenFile.Close())

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountNodeAudienceRestriction, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--runtime-config=api/all=true",
		"--authorization-mode", "Node,RBAC",
		"--token-auth-file", tokenFile.Name(),
		"--enable-admission-plugins", "NodeRestriction",
		"--disable-admission-plugins", "TaintNodesByCondition",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	kubeConfigFile := createKubeConfigFileForRestConfig(t, server.ClientConfig)

	ctx := testContext(t)

	kcm := kubecontrollermanagertesting.StartTestServerOrDie(ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--controllers=ephemeral-volume-controller", // we need this controller to test the ephemeral volume source in the pod
		"--leader-elect=false",                      // KCM leader election calls os.Exit when it ends, so it is easier to just turn it off altogether
	})
	defer kcm.TearDownFn()

	// Build client config and superuser clientset
	clientConfig := server.ClientConfig
	superuserClient, _ := clientsetForToken(tokenMaster, clientConfig)

	if _, err := superuserClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "ns"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// RBAC permissions are required to test service account token requests
	// using the node client because we cannot rely on the node authorizer since we have not
	// configured the references needed for it to work.  This test is focused on exercising
	// the node admission logic, not the node authorizer logic.  We want to know what happens
	// in admission if the request was already authorized.
	configureRBACForServiceAccountToken(t, superuserClient)

	_, err = superuserClient.CoreV1().PersistentVolumeClaims("ns").Create(context.TODO(), &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "mypvc"},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
			VolumeName:  "mypv",
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)

	_, err = superuserClient.CoreV1().PersistentVolumes().Create(context.TODO(), &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "mypv"},
		Spec: corev1.PersistentVolumeSpec{
			AccessModes:            []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Capacity:               corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")},
			ClaimRef:               &corev1.ObjectReference{Namespace: "ns", Name: "mypvc"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{CSI: &corev1.CSIPersistentVolumeSource{Driver: "com.example.csi.mydriver", VolumeHandle: "handle"}},
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)

	_, err = superuserClient.CoreV1().PersistentVolumeClaims("ns").Create(context.TODO(), &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "mypvc-azurefile"},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
			VolumeName:  "mypv-azurefile",
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)

	_, err = superuserClient.CoreV1().PersistentVolumes().Create(context.TODO(), &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "mypv-azurefile"},
		Spec: corev1.PersistentVolumeSpec{
			AccessModes:            []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
			Capacity:               corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")},
			ClaimRef:               &corev1.ObjectReference{Namespace: "ns", Name: "mypvc-azurefile"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{AzureFile: &corev1.AzureFilePersistentVolumeSource{ShareName: "share", SecretName: "secret"}},
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)

	node1Client, _ := clientsetForToken(tokenNode1, clientConfig)
	createNode(t, node1Client, "node1")
	createServiceAccount(t, superuserClient, "ns", "default")

	t.Run("projected volume source with empty audience works", func(t *testing.T) {
		projectedVolumeSourceEmptyAudience := &corev1.ProjectedVolumeSource{Sources: []corev1.VolumeProjection{{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Audience: "", Path: "path"}}}}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{Projected: projectedVolumeSourceEmptyAudience}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")))
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("projected volume source with non-empty audience works", func(t *testing.T) {
		projectedVolumeSource := &corev1.ProjectedVolumeSource{Sources: []corev1.VolumeProjection{{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Audience: "projected-audience", Path: "path"}}}}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{Projected: projectedVolumeSource}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("projected-audience")))
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("pod --> csi --> driver --> tokenrequest with audience forbidden - CSI driver not found", func(t *testing.T) {
		csiDriverVolumeSource := &corev1.CSIVolumeSource{Driver: "com.example.csi.mydriver"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{CSI: csiDriverVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidrivernotfound-audience")), `error validating audience "csidrivernotfound-audience": csidriver.storage.k8s.io "com.example.csi.mydriver" not found`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("pod --> csi --> driver --> tokenrequest with audience works", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "com.example.csi.mydriver")
		csiDriverVolumeSource := &corev1.CSIVolumeSource{Driver: "com.example.csi.mydriver"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{CSI: csiDriverVolumeSource}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience")))
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("pod --> pvc --> pv --> csi --> driver --> tokenrequest with audience forbidden - CSI driver not found", func(t *testing.T) {
		persistentVolumeClaimVolumeSource := &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: persistentVolumeClaimVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("pvc-csidrivernotfound-audience")), `error validating audience "pvc-csidrivernotfound-audience": csidriver.storage.k8s.io "com.example.csi.mydriver" not found`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("pod --> pvc --> pv --> csi --> driver --> tokenrequest with audience forbidden - pvc not found", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "pvcnotfound-audience", "com.example.csi.mydriver")
		persistentVolumeClaimVolumeSource := &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc1"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: persistentVolumeClaimVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("pvcnotfound-audience")), `error validating audience "pvcnotfound-audience": persistentvolumeclaim "mypvc1" not found`)
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("pod --> pvc --> pv --> csi --> driver --> tokenrequest with audience works", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "pvccsidriver-audience", "com.example.csi.mydriver")
		persistentVolumeClaimVolumeSource := &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: persistentVolumeClaimVolumeSource}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("pvccsidriver-audience")))
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("pod --> ephemeral --> pvc --> pv --> csi --> driver --> tokenrequest with audience forbidden - CSI driver not found", func(t *testing.T) {
		ephemeralVolumeSource := &corev1.EphemeralVolumeSource{VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
			Spec: corev1.PersistentVolumeClaimSpec{
				AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
				Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
				VolumeName:  "mypv",
			}}}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{Ephemeral: ephemeralVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("ephemeral-csidrivernotfound-audience")), `error validating audience "ephemeral-csidrivernotfound-audience": csidriver.storage.k8s.io "com.example.csi.mydriver" not found`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("pod --> ephemeral --> pvc --> pv --> csi --> driver --> tokenrequest with audience works", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "ephemeralcsidriver-audience", "com.example.csi.mydriver")
		ephemeralVolumeSource := &corev1.EphemeralVolumeSource{VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
			Spec: corev1.PersistentVolumeClaimSpec{
				AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
				Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
				VolumeName:  "mypv",
			}}}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{Ephemeral: ephemeralVolumeSource}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("ephemeralcsidriver-audience")))
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("csidriver exists but tokenrequest audience not found should be forbidden", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "com.example.csi.mydriver")
		pod := createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience-not-found")), `audience "csidriver-audience-not-found" not found in pod spec volume`)
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("pvc and csidriver exists but tokenrequest audience not found should be forbidden", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "com.example.csi.mydriver")
		persistentVolumeClaimVolumeSource := &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc"}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: persistentVolumeClaimVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience-not-found")), `audience "csidriver-audience-not-found" not found in pod spec volume`)
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("ephemeral volume source with audience not found should be forbidden", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "com.example.csi.mydriver")
		ephemeralVolumeSource := &corev1.EphemeralVolumeSource{VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
			Spec: corev1.PersistentVolumeClaimSpec{
				AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
				Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
				VolumeName:  "mypv",
			}}}
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{Ephemeral: ephemeralVolumeSource}}})
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience-not-found")), `audience "csidriver-audience-not-found" not found in pod spec volume`)
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "com.example.csi.mydriver")
	})

	t.Run("intree pv to csi migration, pod --> csi --> driver --> tokenrequest with audience works", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "file.csi.azure.com")
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "mypvc-azurefile"}}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience")))
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "file.csi.azure.com")
	})

	t.Run("intree inline volume to csi migration, pod --> csi --> driver --> tokenrequest with audience works", func(t *testing.T) {
		createCSIDriver(t, superuserClient, "csidriver-audience", "file.csi.azure.com")
		pod := createPod(t, superuserClient, []corev1.Volume{{Name: "foo", VolumeSource: corev1.VolumeSource{AzureFile: &corev1.AzureFileVolumeSource{ShareName: "default", SecretName: "mypvsecret"}}}})
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("csidriver-audience")))
		deletePod(t, superuserClient, "pod1")
		deleteCSIDriver(t, superuserClient, "file.csi.azure.com")
	})

	t.Run("token request with multiple audiences should be forbidden", func(t *testing.T) {
		pod := createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1", "audience2")), "node may only request 0 or 1 audiences")
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("clusterrole and clusterrolebinding allowing node1 to request audience1 should work", func(t *testing.T) {
		cr := &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"audience1"},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{"some-random-name"},
			}},
		}
		crb := &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1"},
			Subjects:   []rbacv1.Subject{{Kind: "User", Name: "system:node:node1"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "audience-access-for-node1"},
		}

		createServiceAccount(t, superuserClient, "ns", "some-random-name")
		pod := createPod(t, superuserClient, nil, podWithServiceAccountName("some-random-name"))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("some-random-name")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2"), tokenRequestWithName("some-random-name")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACClusterRole(t, cr, superuserClient)
		createRBACClusterRoleBinding(t, crb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("some-random-name")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2"), tokenRequestWithName("some-random-name")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACClusterRole(t, cr, superuserClient)
		deleteRBACClusterRoleBinding(t, crb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("some-random-name")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
		deleteServiceAccount(t, superuserClient, "ns", "some-random-name")
	})

	t.Run("clusterrole and rolebinding in a namespace allowing node1 to request audience1 should work", func(t *testing.T) {
		cr := &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"audience1"},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{"default"},
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "User", Name: "system:node:node1"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "audience-access-for-node1"},
		}

		pod := createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACClusterRole(t, cr, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACClusterRole(t, cr, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("role and rolebinding in a namespace allowing node1 to request audience1 should work", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"audience1"},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{"default"},
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "User", Name: "system:node:node1"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-node1"},
		}

		pod := createPod(t, superuserClient, nil)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("clusterrole and rolebinding in a namespace allowing node1 to request audience1, wildcard for service account names", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"audience1"},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{}, // resource name empty means all service accounts
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "User", Name: "system:node:node1"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-node1"},
		}

		createServiceAccount(t, superuserClient, "ns", "custom-sa")
		pod := createPod(t, superuserClient, nil, podWithServiceAccountName("custom-sa"))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("custom-sa")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2"), tokenRequestWithName("custom-sa")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("custom-sa")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2"), tokenRequestWithName("custom-sa")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1"), tokenRequestWithName("custom-sa")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
		deleteServiceAccount(t, superuserClient, "ns", "custom-sa")
	})

	t.Run("clusterrole and rolebinding in a namespace allowing nodes to request any audience, wildcard for audiences", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-nodes", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"*"}, // wildcard for audiences
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{"default"},
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "Group", Name: "system:nodes"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-nodes"},
		}

		pod := createPod(t, superuserClient, nil)

		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		randomAudience := rand.String(10)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences(randomAudience)), `audience "`+randomAudience+`" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")))
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")))
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")))
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences(randomAudience)))

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("clusterrole and rolebinding in a namespace allowing nodes to request any audience, any service account, wildcard for audiences", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-nodes", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"*"}, // wildcard for audiences
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{}, // empty resource name means all service accounts
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "Group", Name: "system:nodes"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-nodes"},
		}

		pod := createPod(t, superuserClient, nil)

		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")), `audience "audience2" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")))
		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience2")))

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("api server audience configured with rbac role", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-nodes", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{""},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{}, // empty resource name means all service accounts
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "Group", Name: "system:nodes"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-nodes"},
		}

		pod := createPod(t, superuserClient, nil, podWithAutoMountServiceAccountToken(false))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")), `audience "" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("")), `audience "" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})

	t.Run("audience with : and '/' configured with rbac role", func(t *testing.T) {
		role := &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-nodes", Namespace: "ns"},
			Rules: []rbacv1.PolicyRule{{
				APIGroups:     []string{""},
				Resources:     []string{"myaud://audience1/audience2.com"},
				Verbs:         []string{"request-serviceaccounts-token-audience"},
				ResourceNames: []string{}, // empty resource name means all service accounts
			}},
		}
		rb := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "audience-access-for-node1", Namespace: "ns"},
			Subjects:   []rbacv1.Subject{{Kind: "Group", Name: "system:nodes"}},
			RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "audience-access-for-nodes"},
		}

		pod := createPod(t, superuserClient, nil, podWithAutoMountServiceAccountToken(false))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("myaud://audience1/audience2.com")), `audience "myaud://audience1/audience2.com" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		createRBACRole(t, role, superuserClient)
		createRBACRoleBinding(t, rb, superuserClient)

		expectAllowed(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("myaud://audience1/audience2.com")))
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("audience1")), `audience "audience1" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)

		deleteRBACRole(t, role, superuserClient)
		deleteRBACRoleBinding(t, rb, superuserClient)
		// After the delete use a expectedForbiddenMessage to wait for the RBAC authorizer to catch up.
		expectedForbiddenMessage(t, createTokenRequest(node1Client, pod.UID, tokenRequestWithAudiences("myaud://audience1/audience2.com")), `audience "myaud://audience1/audience2.com" not found in pod spec volume, system:node:node1 is not authorized to request tokens for this audience`)
		deletePod(t, superuserClient, "pod1")
	})
}

func createKubeConfigFileForRestConfig(t *testing.T, restConfig *rest.Config) string {
	t.Helper()

	clientConfig := kubeconfig.CreateKubeConfig(restConfig)

	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig.yaml")
	if err := clientcmd.WriteToFile(*clientConfig, kubeConfigFile); err != nil {
		t.Fatal(err)
	}
	return kubeConfigFile
}

type podOptions struct {
	serviceAccountName           string
	autoMountServiceAccountToken *bool
}

type podOption func(*podOptions)

func podWithServiceAccountName(name string) podOption {
	return func(opts *podOptions) {
		opts.serviceAccountName = name
	}
}

func podWithAutoMountServiceAccountToken(autoMount bool) podOption {
	return func(opts *podOptions) {
		opts.autoMountServiceAccountToken = ptr.To(autoMount)
	}
}

func createPod(t *testing.T, client clientset.Interface, volumes []corev1.Volume, opts ...podOption) *corev1.Pod {
	t.Helper()

	options := &podOptions{
		serviceAccountName:           "default",
		autoMountServiceAccountToken: ptr.To(true),
	}

	for _, opt := range opts {
		opt(options)
	}

	pod, err := client.CoreV1().Pods("ns").Create(context.TODO(), &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
		Spec: corev1.PodSpec{
			NodeName:                     "node1",
			Containers:                   []corev1.Container{{Name: "image", Image: "busybox"}},
			ServiceAccountName:           options.serviceAccountName,
			Volumes:                      volumes,
			AutomountServiceAccountToken: options.autoMountServiceAccountToken,
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)
	return pod
}

func deletePod(t *testing.T, client clientset.Interface, podName string) {
	t.Helper()

	checkNilError(t, client.CoreV1().Pods("ns").Delete(context.TODO(), podName, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
}

func createNode(t *testing.T, client clientset.Interface, nodeName string) {
	t.Helper()

	_, err := client.CoreV1().Nodes().Create(context.TODO(), &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}, metav1.CreateOptions{})
	checkNilError(t, err)
}

type tokenRequestOptions struct {
	name      string
	audiences []string
}

type tokenRequestOption func(*tokenRequestOptions)

func tokenRequestWithName(name string) tokenRequestOption {
	return func(opts *tokenRequestOptions) {
		opts.name = name
	}
}

func tokenRequestWithAudiences(audiences ...string) tokenRequestOption {
	return func(opts *tokenRequestOptions) {
		opts.audiences = audiences
	}
}

func createTokenRequest(client clientset.Interface, uid types.UID, opts ...tokenRequestOption) func() error {
	options := &tokenRequestOptions{
		name:      "default",
		audiences: []string{},
	}

	for _, opt := range opts {
		opt(options)
	}

	return func() error {
		_, err := client.CoreV1().ServiceAccounts("ns").CreateToken(context.TODO(), options.name, &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					Name:       "pod1",
					APIVersion: "v1",
					UID:        uid,
				},
				Audiences: options.audiences,
			},
		}, metav1.CreateOptions{})
		return err
	}
}

func createCSIDriver(t *testing.T, client clientset.Interface, audience, driverName string) {
	t.Helper()

	_, err := client.StorageV1().CSIDrivers().Create(context.TODO(), &storagev1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{Name: driverName},
		Spec: storagev1.CSIDriverSpec{
			TokenRequests: []storagev1.TokenRequest{{Audience: audience}},
		},
	}, metav1.CreateOptions{})
	checkNilError(t, err)
}

func deleteCSIDriver(t *testing.T, client clientset.Interface, driverName string) {
	t.Helper()

	checkNilError(t, client.StorageV1().CSIDrivers().Delete(context.TODO(), driverName, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
}

func createServiceAccount(t *testing.T, client clientset.Interface, namespace, name string) {
	t.Helper()

	_, err := client.CoreV1().ServiceAccounts(namespace).Create(context.TODO(), &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}, metav1.CreateOptions{})
	checkNilError(t, err)
}

func deleteServiceAccount(t *testing.T, client clientset.Interface, namespace, name string) {
	t.Helper()

	checkNilError(t, client.CoreV1().ServiceAccounts(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
}

func configureRBACForServiceAccountToken(t *testing.T, client clientset.Interface) {
	t.Helper()

	_, err := client.RbacV1().ClusterRoles().Update(context.TODO(), &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: "system:node:node1"},
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{""},
			Resources: []string{"serviceaccounts/token"},
			Verbs:     []string{"create"},
		}},
	}, metav1.UpdateOptions{})
	checkNilError(t, err)

	_, err = client.RbacV1().ClusterRoleBindings().Update(context.TODO(), &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "system:node"},
		Subjects: []rbacv1.Subject{{
			APIGroup: rbacv1.GroupName,
			Kind:     "Group",
			Name:     "system:nodes",
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     "system:node",
		},
	}, metav1.UpdateOptions{})
	checkNilError(t, err)
}

func createRBACClusterRole(t *testing.T, clusterrole *rbacv1.ClusterRole, client clientset.Interface) {
	t.Helper()

	_, err := client.RbacV1().ClusterRoles().Create(context.TODO(), clusterrole, metav1.CreateOptions{})
	checkNilError(t, err)
}

func createRBACClusterRoleBinding(t *testing.T, clusterrolebinding *rbacv1.ClusterRoleBinding, client clientset.Interface) {
	t.Helper()

	_, err := client.RbacV1().ClusterRoleBindings().Create(context.TODO(), clusterrolebinding, metav1.CreateOptions{})
	checkNilError(t, err)
}

func createRBACRole(t *testing.T, role *rbacv1.Role, client clientset.Interface) {
	t.Helper()

	_, err := client.RbacV1().Roles(role.Namespace).Create(context.TODO(), role, metav1.CreateOptions{})
	checkNilError(t, err)
}

func createRBACRoleBinding(t *testing.T, rolebinding *rbacv1.RoleBinding, client clientset.Interface) {
	t.Helper()

	_, err := client.RbacV1().RoleBindings(rolebinding.Namespace).Create(context.TODO(), rolebinding, metav1.CreateOptions{})
	checkNilError(t, err)
}

func deleteRBACClusterRole(t *testing.T, clusterrole *rbacv1.ClusterRole, client clientset.Interface) {
	t.Helper()

	checkNilError(t, client.RbacV1().ClusterRoles().Delete(context.TODO(), clusterrole.Name, metav1.DeleteOptions{}))
}

func deleteRBACClusterRoleBinding(t *testing.T, clusterrolebinding *rbacv1.ClusterRoleBinding, client clientset.Interface) {
	t.Helper()

	checkNilError(t, client.RbacV1().ClusterRoleBindings().Delete(context.TODO(), clusterrolebinding.Name, metav1.DeleteOptions{}))
}

func deleteRBACRole(t *testing.T, role *rbacv1.Role, client clientset.Interface) {
	t.Helper()

	checkNilError(t, client.RbacV1().Roles(role.Namespace).Delete(context.TODO(), role.Name, metav1.DeleteOptions{}))
}

func deleteRBACRoleBinding(t *testing.T, rolebinding *rbacv1.RoleBinding, client clientset.Interface) {
	t.Helper()

	checkNilError(t, client.RbacV1().RoleBindings(rolebinding.Namespace).Delete(context.TODO(), rolebinding.Name, metav1.DeleteOptions{}))
}
