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
	"strings"
	"testing"
	"time"

	coordination "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/api/resource/v1alpha2"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
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

	// Wait for a healthy server
	for {
		result := superuserClient.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.TODO())
		_, err := result.Raw()
		if err == nil {
			break
		}
		t.Log(err)
		time.Sleep(time.Second)
	}

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
	if _, err := superuserClient.ResourceV1alpha2().ResourceClaims("ns").Create(context.TODO(), &v1alpha2.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "mynamedresourceclaim"}, Spec: v1alpha2.ResourceClaimSpec{ResourceClassName: "example.com"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1alpha2().ResourceClaims("ns").Create(context.TODO(), &v1alpha2.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "mytemplatizedresourceclaim"}, Spec: v1alpha2.ResourceClaimSpec{ResourceClassName: "example.com"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	model := v1alpha2.ResourceModel{NamedResources: &v1alpha2.NamedResourcesResources{}}
	if _, err := superuserClient.ResourceV1alpha2().ResourceSlices().Create(context.TODO(), &v1alpha2.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "myslice1"}, NodeName: "node1", DriverName: "dra.example.com", ResourceModel: model}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := superuserClient.ResourceV1alpha2().ResourceSlices().Create(context.TODO(), &v1alpha2.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "myslice2"}, NodeName: "node2", DriverName: "dra.example.com", ResourceModel: model}, metav1.CreateOptions{}); err != nil {
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
			_, err := client.ResourceV1alpha2().ResourceClaims("ns").Get(context.TODO(), "mynamedresourceclaim", metav1.GetOptions{})
			return err
		}
	}
	getResourceClaimTemplate := func(client clientset.Interface) func() error {
		return func() error {
			_, err := client.ResourceV1alpha2().ResourceClaims("ns").Get(context.TODO(), "mytemplatizedresourceclaim", metav1.GetOptions{})
			return err
		}
	}
	deleteResourceSliceCollection := func(client clientset.Interface, nodeName *string) func() error {
		return func() error {
			var listOptions metav1.ListOptions
			if nodeName != nil {
				listOptions.FieldSelector = "nodeName=" + *nodeName
			}
			return client.ResourceV1alpha2().ResourceSlices().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, listOptions)
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
	slices, err := superuserClient.ResourceV1alpha2().ResourceSlices().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(slices.Items) != 1 {
		t.Fatalf("unexpected slices: %v", slices.Items)
	}
	if slices.Items[0].NodeName != "node2" {
		t.Fatal("wrong slice deleted")
	}

	// Superuser can delete.
	expectAllowed(t, deleteResourceSliceCollection(superuserClient, nil))
	slices, err = superuserClient.ResourceV1alpha2().ResourceSlices().List(context.TODO(), metav1.ListOptions{})
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
