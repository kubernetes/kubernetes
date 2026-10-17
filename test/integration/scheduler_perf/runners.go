/*
Copyright The Kubernetes Authors.

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

package benchmark

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type PrepareNodeStrategy interface {
	// Modify pre-created Node objects before the test starts.
	PreparePatch(node *v1.Node) []byte
	// Create or modify any objects that depend on the node before the test starts.
	// Caller will re-try when http.StatusConflict error is returned.
	PrepareDependentObjects(ctx context.Context, node *v1.Node, client clientset.Interface) error
	// Clean up any node modifications after the test finishes.
	CleanupNode(ctx context.Context, node *v1.Node) *v1.Node
	// Clean up any objects that depend on the node after the test finishes.
	// Caller will re-try when http.StatusConflict error is returned.
	CleanupDependentObjects(ctx context.Context, nodeName string, client clientset.Interface) error
}

type TrivialNodePrepareStrategy struct{}

var _ PrepareNodeStrategy = &TrivialNodePrepareStrategy{}

func (*TrivialNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	return []byte{}
}

func (*TrivialNodePrepareStrategy) CleanupNode(ctx context.Context, node *v1.Node) *v1.Node {
	nodeCopy := *node
	return &nodeCopy
}

func (*TrivialNodePrepareStrategy) PrepareDependentObjects(ctx context.Context, node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*TrivialNodePrepareStrategy) CleanupDependentObjects(ctx context.Context, nodeName string, client clientset.Interface) error {
	return nil
}

type LabelNodePrepareStrategy struct {
	LabelKey      string
	LabelValues   []string
	roundRobinIdx int
}

var _ PrepareNodeStrategy = &LabelNodePrepareStrategy{}

func NewLabelNodePrepareStrategy(labelKey string, labelValues ...string) *LabelNodePrepareStrategy {
	return &LabelNodePrepareStrategy{
		LabelKey:    labelKey,
		LabelValues: labelValues,
	}
}

func (s *LabelNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	labelString := fmt.Sprintf("{\"%v\":\"%v\"}", s.LabelKey, s.LabelValues[s.roundRobinIdx])
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	s.roundRobinIdx++
	if s.roundRobinIdx == len(s.LabelValues) {
		s.roundRobinIdx = 0
	}
	return []byte(patch)
}

func (s *LabelNodePrepareStrategy) CleanupNode(ctx context.Context, node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	if node.Labels != nil && len(node.Labels[s.LabelKey]) != 0 {
		delete(nodeCopy.Labels, s.LabelKey)
	}
	return nodeCopy
}

func (*LabelNodePrepareStrategy) PrepareDependentObjects(ctx context.Context, node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*LabelNodePrepareStrategy) CleanupDependentObjects(ctx context.Context, nodeName string, client clientset.Interface) error {
	return nil
}

// NodeAllocatableStrategy fills node.status.allocatable and csiNode.spec.drivers[*].allocatable.
// csiNode is created if it does not exist. On cleanup, any csiNode.spec.drivers[*].allocatable is
// set to nil.
type NodeAllocatableStrategy struct {
	// Node.status.allocatable to fill to all nodes.
	NodeAllocatable map[v1.ResourceName]string
	// Map <driver_name> -> VolumeNodeResources to fill into csiNode.spec.drivers[<driver_name>].
	CsiNodeAllocatable map[string]*storagev1.VolumeNodeResources
	// List of in-tree volume plugins migrated to CSI.
	MigratedPlugins []string
}

var _ PrepareNodeStrategy = &NodeAllocatableStrategy{}

func NewNodeAllocatableStrategy(nodeAllocatable map[v1.ResourceName]string, csiNodeAllocatable map[string]*storagev1.VolumeNodeResources, migratedPlugins []string) *NodeAllocatableStrategy {
	return &NodeAllocatableStrategy{
		NodeAllocatable:    nodeAllocatable,
		CsiNodeAllocatable: csiNodeAllocatable,
		MigratedPlugins:    migratedPlugins,
	}
}

func (s *NodeAllocatableStrategy) PreparePatch(node *v1.Node) []byte {
	newNode := node.DeepCopy()
	for name, value := range s.NodeAllocatable {
		newNode.Status.Allocatable[name] = resource.MustParse(value)
	}

	oldJSON, err := json.Marshal(node)
	if err != nil {
		panic(err)
	}
	newJSON, err := json.Marshal(newNode)
	if err != nil {
		panic(err)
	}

	patch, err := strategicpatch.CreateTwoWayMergePatch(oldJSON, newJSON, v1.Node{})
	if err != nil {
		panic(err)
	}
	return patch
}

func (s *NodeAllocatableStrategy) CleanupNode(ctx context.Context, node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	for name := range s.NodeAllocatable {
		delete(nodeCopy.Status.Allocatable, name)
	}
	return nodeCopy
}

func (s *NodeAllocatableStrategy) createCSINode(ctx context.Context, nodeName string, client clientset.Interface) error {
	csiNode := &storagev1.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
			Annotations: map[string]string{
				v1.MigratedPluginsAnnotationKey: strings.Join(s.MigratedPlugins, ","),
			},
		},
		Spec: storagev1.CSINodeSpec{
			Drivers: []storagev1.CSINodeDriver{},
		},
	}

	for driver, allocatable := range s.CsiNodeAllocatable {
		d := storagev1.CSINodeDriver{
			Name:        driver,
			Allocatable: allocatable,
			NodeID:      nodeName,
		}
		csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, d)
	}

	_, err := client.StorageV1().CSINodes().Create(ctx, csiNode, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		// Something created CSINode instance after we checked it did not exist.
		// Make the caller to re-try PrepareDependentObjects by returning Conflict error
		err = apierrors.NewConflict(storagev1.Resource("csinodes"), nodeName, err)
	}
	return err
}

func (s *NodeAllocatableStrategy) updateCSINode(ctx context.Context, csiNode *storagev1.CSINode, client clientset.Interface) error {
	for driverName, allocatable := range s.CsiNodeAllocatable {
		found := false
		for i, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				found = true
				csiNode.Spec.Drivers[i].Allocatable = allocatable
				break
			}
		}
		if !found {
			d := storagev1.CSINodeDriver{
				Name:        driverName,
				Allocatable: allocatable,
			}

			csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, d)
		}
	}
	csiNode.Annotations[v1.MigratedPluginsAnnotationKey] = strings.Join(s.MigratedPlugins, ",")

	_, err := client.StorageV1().CSINodes().Update(ctx, csiNode, metav1.UpdateOptions{})
	return err
}

func (s *NodeAllocatableStrategy) PrepareDependentObjects(ctx context.Context, node *v1.Node, client clientset.Interface) error {
	csiNode, err := client.StorageV1().CSINodes().Get(ctx, node.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return s.createCSINode(ctx, node.Name, client)
		}
		return err
	}
	return s.updateCSINode(ctx, csiNode, client)
}

func (s *NodeAllocatableStrategy) CleanupDependentObjects(ctx context.Context, nodeName string, client clientset.Interface) error {
	csiNode, err := client.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	for driverName := range s.CsiNodeAllocatable {
		for i, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				csiNode.Spec.Drivers[i].Allocatable = nil
			}
		}
	}
	return s.updateCSINode(ctx, csiNode, client)
}

// UniqueNodeLabelStrategy sets a unique label for each node.
type UniqueNodeLabelStrategy struct {
	LabelKey string
}

var _ PrepareNodeStrategy = &UniqueNodeLabelStrategy{}

func NewUniqueNodeLabelStrategy(labelKey string) *UniqueNodeLabelStrategy {
	return &UniqueNodeLabelStrategy{
		LabelKey: labelKey,
	}
}

func (s *UniqueNodeLabelStrategy) PreparePatch(*v1.Node) []byte {
	labelString := fmt.Sprintf("{\"%v\":\"%v\"}", s.LabelKey, string(uuid.NewUUID()))
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	return []byte(patch)
}

func (s *UniqueNodeLabelStrategy) CleanupNode(ctx context.Context, node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	if node.Labels != nil && len(node.Labels[s.LabelKey]) != 0 {
		delete(nodeCopy.Labels, s.LabelKey)
	}
	return nodeCopy
}

func (*UniqueNodeLabelStrategy) PrepareDependentObjects(ctx context.Context, node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*UniqueNodeLabelStrategy) CleanupDependentObjects(ctx context.Context, nodeName string, client clientset.Interface) error {
	return nil
}

const retries = 5

func DoPrepareNode(ctx context.Context, client clientset.Interface, node *v1.Node, strategy PrepareNodeStrategy) error {
	var err error
	patch := strategy.PreparePatch(node)
	if len(patch) == 0 {
		return nil
	}
	for range retries {
		if _, err = client.CoreV1().Nodes().Patch(ctx, node.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{}); err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("error while applying patch %v to Node %v: %v", string(patch), node.Name, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("too many conflicts when applying patch %v to Node %v: %s", string(patch), node.Name, err)
	}

	for range retries {
		if err = strategy.PrepareDependentObjects(ctx, node, client); err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("error while preparing objects for node %s: %s", node.Name, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("too many conflicts when creating objects for node %s: %s", node.Name, err)
	}
	return nil
}

type TestPodCreateStrategy func(ctx context.Context, client clientset.Interface, namespace string, podCount int) error

type CountToPodStrategy struct {
	Count    int
	Strategy TestPodCreateStrategy
}

type TestPodCreatorConfig map[string][]CountToPodStrategy

func NewTestPodCreatorConfig() *TestPodCreatorConfig {
	config := make(TestPodCreatorConfig)
	return &config
}

type CountToStrategy struct {
	Count    int
	Strategy PrepareNodeStrategy
}

type TestNodePreparer interface {
	PrepareNodes(ctx context.Context, nextNodeIndex int) error
	CleanupNodes(ctx context.Context) error
}

func (c *TestPodCreatorConfig) AddStrategy(
	namespace string, podCount int, strategy TestPodCreateStrategy) {
	(*c)[namespace] = append((*c)[namespace], CountToPodStrategy{Count: podCount, Strategy: strategy})
}

type TestPodCreator struct {
	Client clientset.Interface
	// namespace -> count -> strategy
	Config *TestPodCreatorConfig
}

func NewTestPodCreator(client clientset.Interface, config *TestPodCreatorConfig) *TestPodCreator {
	return &TestPodCreator{
		Client: client,
		Config: config,
	}
}

func (c *TestPodCreator) CreatePods(ctx context.Context) error {
	for ns, v := range *(c.Config) {
		for _, countToStrategy := range v {
			if err := countToStrategy.Strategy(ctx, c.Client, ns, countToStrategy.Count); err != nil {
				return err
			}
		}
	}
	return nil
}

func MakePodSpec() v1.PodSpec {
	return v1.PodSpec{
		Containers: []v1.Container{{
			Name:  "pause",
			Image: imageutils.GetE2EImage(imageutils.Pause),
			Ports: []v1.ContainerPort{{ContainerPort: 80}},
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("500Mi"),
				},
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("500Mi"),
				},
			},
		}},
	}
}

func makeCreatePod(client clientset.Interface, namespace string, podTemplate *v1.Pod) error {
	if err := testutils.CreatePodWithRetries(client, namespace, podTemplate); err != nil {
		return fmt.Errorf("error creating pod: %v", err)
	}
	return nil
}

func CreatePod(ctx context.Context, client clientset.Interface, namespace string, podCount int, podTemplate PodTemplate) error {
	var createError error
	lock := sync.Mutex{}
	createPodFunc := func(i int) {
		pod, err := podTemplate.GetPodTemplate(i, podCount)
		if err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = err
			return
		}
		pod = pod.DeepCopy()
		// client-go writes into the object that is passed to Create,
		// causing a data race unless we create a new copy for each
		// parallel call.
		if err := makeCreatePod(client, namespace, pod); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = err
		}
	}

	if podCount < 30 {
		workqueue.ParallelizeUntil(ctx, podCount, podCount, createPodFunc)
	} else {
		workqueue.ParallelizeUntil(ctx, 30, podCount, createPodFunc)
	}
	return createError
}

func CreatePodWithPersistentVolume(ctx context.Context, client clientset.Interface, namespace string, claimTemplate *v1.PersistentVolumeClaim, factory volumeFactory, podTemplate PodTemplate, count int, bindVolume bool) error {
	var createError error
	lock := sync.Mutex{}
	createPodFunc := func(i int) {
		pvcName := fmt.Sprintf("pvc-%d", i)
		// pvc
		pvc := claimTemplate.DeepCopy()
		pvc.Name = pvcName
		// pv
		pv := factory(i)
		// PVs are cluster-wide resources.
		// Prepend a namespace to make the name globally unique.
		pv.Name = fmt.Sprintf("%s-%s", namespace, pv.Name)
		pvs := pv.Spec.PersistentVolumeSource
		if pvs.CSI != nil {
			pvs.CSI.VolumeHandle = pv.Name
		}
		if bindVolume {
			// bind pv to "pvc-$i"
			pv.Spec.ClaimRef = &v1.ObjectReference{
				Kind:       "PersistentVolumeClaim",
				Namespace:  namespace,
				Name:       pvcName,
				APIVersion: "v1",
			}
			pv.Status.Phase = v1.VolumeBound

			// bind pvc to "pv-$i"
			pvc.Spec.VolumeName = pv.Name
			pvc.Status.Phase = v1.ClaimBound
		} else {
			pv.Status.Phase = v1.VolumeAvailable
		}

		// Create PVC first as it's referenced by the PV when the `bindVolume` is true.
		if err := testutils.CreatePersistentVolumeClaimWithRetries(client, namespace, pvc); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error creating PVC: %s", err)
			return
		}

		// We need to update statuses separately, as creating pv/pvc resets status to the default one.
		if _, err := client.CoreV1().PersistentVolumeClaims(namespace).UpdateStatus(ctx, pvc, metav1.UpdateOptions{}); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error updating PVC status: %s", err)
			return
		}

		if err := testutils.CreatePersistentVolumeWithRetries(client, pv); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error creating PV: %s", err)
			return
		}
		// We need to update statuses separately, as creating pv/pvc resets status to the default one.
		if _, err := client.CoreV1().PersistentVolumes().UpdateStatus(ctx, pv, metav1.UpdateOptions{}); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error updating PV status: %s", err)
			return
		}

		// pod
		pod, err := podTemplate.GetPodTemplate(i, count)
		if err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error getting pod template: %s", err)
			return
		}
		pod = pod.DeepCopy()
		pod.Spec.Volumes = []v1.Volume{
			{
				Name: "vol",
				VolumeSource: v1.VolumeSource{
					PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
						ClaimName: pvcName,
					},
				},
			},
		}
		if err := makeCreatePod(client, namespace, pod); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = err
			return
		}
	}

	if count < 30 {
		workqueue.ParallelizeUntil(ctx, count, count, createPodFunc)
	} else {
		workqueue.ParallelizeUntil(ctx, 30, count, createPodFunc)
	}
	return createError
}

func NewCustomCreatePodStrategy(podTemplate PodTemplate) TestPodCreateStrategy {
	return func(ctx context.Context, client clientset.Interface, namespace string, podCount int) error {
		return CreatePod(ctx, client, namespace, podCount, podTemplate)
	}
}

// volumeFactory creates an unique PersistentVolume for given integer.
type volumeFactory func(uniqueID int) *v1.PersistentVolume

// PodTemplate is responsible for creating a v1.Pod instance that is ready
// to be sent to the API server.
type PodTemplate interface {
	// GetPodTemplate returns a pod template for one out of many different pods.
	// Pods with numbers in the range [index, index+count-1] will be created
	// based on what GetPodTemplate returns. It gets called multiple times
	// with a fixed index and increasing count parameters. This number can,
	// but doesn't have to be, used to modify parts of the pod spec like
	// for example a named reference to some other object.
	GetPodTemplate(index, count int) (*v1.Pod, error)
}

// StaticPodTemplate returns an implementation of PodTemplate for a fixed pod that is the same regardless of the index.
func StaticPodTemplate(pod *v1.Pod) PodTemplate {
	return (*staticPodTemplate)(pod)
}

type staticPodTemplate v1.Pod

// GetPodTemplate implements [PodTemplate.GetPodTemplate] by returning the same pod
// for each call.
func (s *staticPodTemplate) GetPodTemplate(index, count int) (*v1.Pod, error) {
	return (*v1.Pod)(s), nil
}

func NewCreatePodWithPersistentVolumeStrategy(claimTemplate *v1.PersistentVolumeClaim, factory volumeFactory, podTemplate PodTemplate) TestPodCreateStrategy {
	return func(ctx context.Context, client clientset.Interface, namespace string, podCount int) error {
		return CreatePodWithPersistentVolume(ctx, client, namespace, claimTemplate, factory, podTemplate, podCount, true /* bindVolume */)
	}
}
