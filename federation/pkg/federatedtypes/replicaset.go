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

package federatedtypes

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/pkg/api"
	fedapi "k8s.io/kubernetes/federation/apis/federation"
	fedv1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/planner"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/podanalyzer"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	ReplicaSetKind                     = "replicaset"
	ReplicaSetControllerName           = "replicasets"
	FedReplicaSetPreferencesAnnotation = "federation.kubernetes.io/replica-set-preferences"
)

type replicaSetUserInfo struct {
	scheduleResult (map[string]int64)
	fedStatus      *extensionsv1.ReplicaSetStatus
}

func init() {
	RegisterFederatedType(ReplicaSetKind, ReplicaSetControllerName, []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource(ReplicaSetControllerName)}, NewReplicaSetAdapter)
}

type ReplicaSetAdapter struct {
	client         federationclientset.Interface
	defaultPlanner *planner.Planner
}

func NewReplicaSetAdapter(client federationclientset.Interface) FederatedTypeAdapter {
	return &ReplicaSetAdapter{
		client: client,
		defaultPlanner: planner.NewPlanner(&fedapi.FederatedReplicaSetPreferences{
			Clusters: map[string]fedapi.ClusterReplicaSetPreferences{
				"*": {Weight: 1},
			},
		})}
}

func (a *ReplicaSetAdapter) Kind() string {
	return ReplicaSetKind
}

func (a *ReplicaSetAdapter) ObjectType() pkgruntime.Object {
	return &extensionsv1.ReplicaSet{}
}

func (a *ReplicaSetAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*extensionsv1.ReplicaSet)
	return ok
}

func (a *ReplicaSetAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	copiedObj, err := api.Scheme.DeepCopy(obj)
	if err != nil {
		return nil
	}
	return copiedObj.(pkgruntime.Object)
}

//		lrs := &extensionsv1.ReplicaSet{
//			ObjectMeta: util.DeepCopyRelevantObjectMeta(frs.ObjectMeta),
//			Spec:       *util.DeepCopyApiTypeOrPanic(&replicaset.Spec).(*extensionsv1.ReplicaSetSpec),
//		}
//	return &extensionsv1.ReplicaSet{
//		ObjectMeta: util.DeepCopyRelevantObjectMeta(configmap.ObjectMeta),
//		Data:       configmap.Data,
//	}
//}

func (a *ReplicaSetAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	replicaset1 := obj1.(*extensionsv1.ReplicaSet)
	replicaset2 := obj2.(*extensionsv1.ReplicaSet)
	return fedutil.ObjectMetaAndSpecEquivalent(replicaset1, replicaset2)
}

func (a *ReplicaSetAdapter) NamespacedName(obj pkgruntime.Object) types.NamespacedName {
	replicaset := obj.(*extensionsv1.ReplicaSet)
	return types.NamespacedName{Namespace: replicaset.Namespace, Name: replicaset.Name}
}

func (a *ReplicaSetAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*extensionsv1.ReplicaSet).ObjectMeta
}

func (a *ReplicaSetAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	replicaset := obj.(*extensionsv1.ReplicaSet)
	return a.client.Extensions().ReplicaSets(replicaset.Namespace).Create(replicaset)
}

func (a *ReplicaSetAdapter) FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error {
	return a.client.Extensions().ReplicaSets(namespacedName.Namespace).Delete(namespacedName.Name, options)
}

func (a *ReplicaSetAdapter) FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return a.client.Extensions().ReplicaSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *ReplicaSetAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.Extensions().ReplicaSets(namespace).List(options)
}

func (a *ReplicaSetAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	replicaset := obj.(*extensionsv1.ReplicaSet)
	return a.client.Extensions().ReplicaSets(replicaset.Namespace).Update(replicaset)
}

func (a *ReplicaSetAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.Extensions().ReplicaSets(namespace).Watch(options)
}

func (a *ReplicaSetAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	replicaset := obj.(*extensionsv1.ReplicaSet)
	return client.Extensions().ReplicaSets(replicaset.Namespace).Create(replicaset)
}

func (a *ReplicaSetAdapter) ClusterDelete(client kubeclientset.Interface, nsName types.NamespacedName, options *metav1.DeleteOptions) error {
	return client.Extensions().ReplicaSets(nsName.Namespace).Delete(nsName.Name, options)
}

func (a *ReplicaSetAdapter) ClusterGet(client kubeclientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return client.Extensions().ReplicaSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *ReplicaSetAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.Extensions().ReplicaSets(namespace).List(options)
}

func (a *ReplicaSetAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	replicaset := obj.(*extensionsv1.ReplicaSet)
	return client.Extensions().ReplicaSets(replicaset.Namespace).Update(replicaset)
}

func (a *ReplicaSetAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.Extensions().ReplicaSets(namespace).Watch(options)
}

func (a *ReplicaSetAdapter) PrepareForUpdateFunc() func(obj pkgruntime.Object, key string, clusters []*fedv1beta1.Cluster, informer fedutil.FederatedInformer) (interface{}, error) {
	return func(obj pkgruntime.Object, key string, clusters []*fedv1beta1.Cluster, informer fedutil.FederatedInformer) (interface{}, error) {
		var clusterNames []string
		for _, cluster := range clusters {
			clusterNames = append(clusterNames, cluster.Name)
		}

		// Schedule the pods across the existing clusters.
		replicaSetGetter := func(clusterName, key string) (interface{}, bool, error) {
			return informer.GetTargetStore().GetByKey(clusterName, key)
		}
		podsGetter := func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error) {
			clientset, err := informer.GetClientsetForCluster(clusterName)
			if err != nil {
				return nil, err
			}
			selector, err := metav1.LabelSelectorAsSelector(replicaSet.Spec.Selector)
			if err != nil {
				return nil, fmt.Errorf("invalid selector: %v", err)
			}
			return clientset.Core().Pods(replicaSet.ObjectMeta.Namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		}
		current, estimatedCapacity, err := clustersReplicaState(clusterNames, key, replicaSetGetter, podsGetter)
		if err != nil {
			return nil, err
		}
		rs := obj.(*extensionsv1.ReplicaSet)
		return &replicaSetUserInfo{
			scheduleResult: a.schedule(rs, clusterNames, current, estimatedCapacity),
			fedStatus:      &extensionsv1.ReplicaSetStatus{ObservedGeneration: rs.Generation},
		}, nil
	}
}

func (a *ReplicaSetAdapter) UpdateObjectFunc() func(cluster *fedv1beta1.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, userInfo interface{}) (pkgruntime.Object, error) {
	return func(cluster *fedv1beta1.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, userInfo interface{}) (pkgruntime.Object, error) {
		rsUserInfo := userInfo.(*replicaSetUserInfo)
		rs := federationObjCopy.(*extensionsv1.ReplicaSet)
		if clusterObj != nil {
			rs = clusterObj.(*extensionsv1.ReplicaSet)
		}

		replicas, ok := rsUserInfo.scheduleResult[cluster.Name]
		if !ok {
			replicas = 0
		}
		specReplicas := int32(replicas)
		rs.Spec.Replicas = &specReplicas

		rsUserInfo.fedStatus.Replicas += rs.Status.Replicas
		rsUserInfo.fedStatus.FullyLabeledReplicas += rs.Status.FullyLabeledReplicas
		rsUserInfo.fedStatus.ReadyReplicas += rs.Status.ReadyReplicas
		rsUserInfo.fedStatus.AvailableReplicas += rs.Status.AvailableReplicas
		if replicas == 0 {
			return nil, nil
		}
		return rs, nil
	}
}

func (a *ReplicaSetAdapter) UpdateFinishedFunc() func(obj pkgruntime.Object, userInfo interface{}) error {
	return func(obj pkgruntime.Object, userInfo interface{}) error {
		rs := obj.(*extensionsv1.ReplicaSet)
		rsUserInfo := userInfo.(*replicaSetUserInfo)

		if rsUserInfo.fedStatus.Replicas != rs.Status.Replicas || rsUserInfo.fedStatus.FullyLabeledReplicas != rs.Status.FullyLabeledReplicas ||
			rsUserInfo.fedStatus.ReadyReplicas != rs.Status.ReadyReplicas || rsUserInfo.fedStatus.AvailableReplicas != rs.Status.AvailableReplicas {
			rs.Status = *rsUserInfo.fedStatus
			_, err := a.client.Extensions().ReplicaSets(rs.Namespace).UpdateStatus(rs)
			return err
		}
		return nil
	}
}

func parseFederationReplicaSetReference(frs *extensionsv1.ReplicaSet) (*fedapi.FederatedReplicaSetPreferences, error) {
	if frs.Annotations == nil {
		return nil, nil
	}
	frsPrefString, found := frs.Annotations[FedReplicaSetPreferencesAnnotation]
	if !found {
		return nil, nil
	}
	var frsPref fedapi.FederatedReplicaSetPreferences
	if err := json.Unmarshal([]byte(frsPrefString), &frsPref); err != nil {
		return nil, err
	}
	return &frsPref, nil
}

func (a *ReplicaSetAdapter) schedule(frs *extensionsv1.ReplicaSet, clusterNames []string,
	current map[string]int64, estimatedCapacity map[string]int64) map[string]int64 {
	// TODO: integrate real scheduler

	plnr := a.defaultPlanner
	frsPref, err := parseFederationReplicaSetReference(frs)
	if err != nil {
		glog.Info("Invalid ReplicaSet specific preference, use default. rs: %v, err: %v", frs, err)
	}
	if frsPref != nil { // create a new planner if user specified a preference
		plnr = planner.NewPlanner(frsPref)
	}

	replicas := int64(*frs.Spec.Replicas)
	scheduleResult, overflow := plnr.Plan(replicas, clusterNames, current, estimatedCapacity,
		frs.Namespace+"/"+frs.Name)
	// make sure the return contains clusters need to zero the replicas
	result := make(map[string]int64)
	for clusterName := range current {
		result[clusterName] = 0
	}
	for clusterName, replicas := range scheduleResult {
		result[clusterName] = replicas
	}
	for clusterName, replicas := range overflow {
		result[clusterName] += replicas
	}
	if glog.V(4) {
		buf := bytes.NewBufferString(fmt.Sprintf("Schedule - ReplicaSet: %s/%s\n", frs.Namespace, frs.Name))
		sort.Strings(clusterNames)
		for _, clusterName := range clusterNames {
			cur := current[clusterName]
			target := scheduleResult[clusterName]
			fmt.Fprintf(buf, "%s: current: %d target: %d", clusterName, cur, target)
			if over, found := overflow[clusterName]; found {
				fmt.Fprintf(buf, " overflow: %d", over)
			}
			if capacity, found := estimatedCapacity[clusterName]; found {
				fmt.Fprintf(buf, " capacity: %d", capacity)
			}
			fmt.Fprintf(buf, "\n")
		}
		glog.V(4).Infof(buf.String())
	}
	return result
}

// clusterReplicaState returns information about the scheduling state of the pods running in the federated clusters.
func clustersReplicaState(
	clusterNames []string,
	replicaSetKey string,
	replicaSetGetter func(clusterName string, key string) (interface{}, bool, error),
	podsGetter func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error)) (current map[string]int64, estimatedCapacity map[string]int64, err error) {

	current = make(map[string]int64)
	estimatedCapacity = make(map[string]int64)

	for _, clusterName := range clusterNames {
		lrsObj, exists, err := replicaSetGetter(clusterName, replicaSetKey)
		if err != nil {
			return nil, nil, err
		}
		if !exists {
			continue
		}
		lrs := lrsObj.(*extensionsv1.ReplicaSet)
		if int32(*lrs.Spec.Replicas) == lrs.Status.ReadyReplicas {
			current[clusterName] = int64(lrs.Status.ReadyReplicas)
		} else {
			pods, err := podsGetter(clusterName, lrs)
			if err != nil {
				return nil, nil, err
			}
			podStatus := podanalyzer.AnalyzePods(pods, time.Now())
			current[clusterName] = int64(podStatus.RunningAndReady) // include pending as well?
			unschedulable := int64(podStatus.Unschedulable)
			if unschedulable > 0 {
				estimatedCapacity[clusterName] = int64(*lrs.Spec.Replicas) - unschedulable
			}
		}
	}
	return current, estimatedCapacity, nil
}

func (a *ReplicaSetAdapter) NewTestObject(namespace string) pkgruntime.Object {
	replicas := int32(2)
	return &extensionsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-replicaset-",
			Namespace:    namespace,
		},
		Spec: extensionsv1.ReplicaSetSpec{
			Replicas: &replicas,
		},
	}
}
