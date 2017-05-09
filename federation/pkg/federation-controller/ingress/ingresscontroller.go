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

package ingress

import (
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/golang/glog"

	v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationcache "k8s.io/kubernetes/federation/client/cache"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	cache "k8s.io/kubernetes/pkg/client/cache"
	release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	ingressSyncPeriod = 10 * time.Minute
	clusterSyncPeriod = 10 * time.Minute

	// How long to wait before retrying the processing of an ingress change.
	// If this changes, the sleep in hack/jenkins/e2e.sh before downing a cluster
	// should be changed appropriately.
	minRetryDelay = 5 * time.Second
	maxRetryDelay = 300 * time.Second

	// clientRetryCount applies when accessing a remote kube-apiserver or federation apiserver.
	// TODO: quinton: All of the below belongs in a common config for all federation controllers.
	clientRetryCount = 5

	retryable = true

	doNotRetry = time.Duration(0)

	UserAgentName = "federation-ingress-controller"
	KubeAPIQPS    = 20.0
	KubeAPIBurst  = 30
)

type cachedIngress struct {
	// TODO:quinton: replace with mwieglus common library stuff
	lastState *v1beta1.Ingress
	// The state as successfully applied to the clusters
	// TODO:quinton: replace with mwieglus common library stuff
	appliedState *v1beta1.Ingress
	// cluster endpoint map hold subset info from kubernetes clusters
	// key clusterName
	// value is a flag indicating whether there is a ready address, 1 means there is ready address
	// TODO: quinton: Do we need this at all, and if so, make it a bool
	endpointMap map[string]int
	// ingressStatusMap holds ingress status info from kubernetes clusters, keyed on clusterName
	ingressStatusMap map[string]v1beta1.LoadBalancerStatus
	// Ensures only one goroutine can operate on this ingress at any given time.
	rwlock sync.Mutex
	// Controls error back-off for procceeding federation ingress to k8s clusters
	lastRetryDelay time.Duration
	// Controls error back-off for updating federation ingress back to federation apiserver
	lastFedUpdateDelay time.Duration
}

type ingressCache struct {
	rwlock sync.Mutex // protects ingressMap
	// fedIngressMap contains all ingresses received from federation apiserver
	// key ingressName
	fedIngressMap map[string]*cachedIngress
}

type IngressController struct {
	// TODO: quinton: Probably don't need DNS.
	dns              dnsprovider.Interface
	federationClient federation_release_1_4.Interface
	federationName   string
	// TODO: quinton: Probably don't need DNS.
	zoneName string
	// TODO: quinton: Probably don't need DNS.
	// each federation should be configured with a single zone (e.g. "mycompany.com")
	dnsZones     dnsprovider.Zones
	ingressCache *ingressCache
	clusterCache *clusterClientCache
	// A store of ingresses, populated by the ingressController
	ingressStore cache.StoreToIngressLister
	// Watches changes to all ingresses
	ingressController *framework.Controller
	// A store of clusters, populated by the ingressController
	// TODO: quinton: This does not beling here - move it to the common controller library/framework
	clusterStore federationcache.StoreToClusterLister
	// Watches changes to all clusters
	clusterController *framework.Controller
	eventBroadcaster  record.EventBroadcaster
	eventRecorder     record.EventRecorder
	// ingresses that need to be synced
	queue           *workqueue.Type
	knownClusterSet sets.String
}

// New returns a new ingress controller to keep cluster ingresses in sync with the registry.
func New(federationClient federation_release_1_4.Interface, dns dnsprovider.Interface, federationName, zoneName string) *IngressController {
	broadcaster := record.NewBroadcaster()
	// federationClient event is not supported yet
	// broadcaster.StartRecordingToSink(&unversioned_core.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: UserAgentName})

	s := &IngressController{
		// TODO:  quinton: Probably don't need DNS
		dns:              dns,
		federationClient: federationClient,
		// TODO:  quinton: Probably don't need DNS
		federationName: federationName,
		// TODO:  quinton: Probably don't need DNS
		zoneName:     zoneName,
		ingressCache: &ingressCache{fedIngressMap: make(map[string]*cachedIngress)},
		clusterCache: &clusterClientCache{
			rwlock:    sync.Mutex{},
			clientMap: make(map[string]*clusterCache),
		},
		eventBroadcaster: broadcaster,
		eventRecorder:    recorder,
		queue:            workqueue.New(),
		knownClusterSet:  make(sets.String),
	}
	s.ingressStore.Store, s.ingressController = framework.NewInformer(
		&cache.ListWatch{
			// TODO: quinton: Ingress is not in Core, it's in v1beta1
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return s.federationClient.Core().Ingresses(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return s.federationClient.Core().Ingresses(v1.NamespaceAll).Watch(options)
			},
		},
		&v1beta1.Ingress{},
		ingressSyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: s.enqueueIngress,
			UpdateFunc: func(old, cur interface{}) {
				// there is case that old and new are equals but we still catch the event now.
				if !reflect.DeepEqual(old, cur) {
					s.enqueueIngress(cur)
				}
			},
			DeleteFunc: s.enqueueIngress,
		},
	)
	s.clusterStore.Store, s.clusterController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (pkg_runtime.Object, error) {
				return s.federationClient.Federation().Clusters().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return s.federationClient.Federation().Clusters().Watch(options)
			},
		},
		&v1beta1.Cluster{},
		clusterSyncPeriod,
		framework.ResourceEventHandlerFuncs{
			DeleteFunc: s.clusterCache.delFromClusterSet,
			AddFunc:    s.clusterCache.addToClientMap,
			UpdateFunc: func(old, cur interface{}) {
				oldCluster, ok := old.(*v1beta1.Cluster)
				if !ok {
					return
				}
				curCluster, ok := cur.(*v1beta1.Cluster)
				if !ok {
					return
				}
				if !reflect.DeepEqual(oldCluster.Spec, curCluster.Spec) {
					// update when spec is changed
					s.clusterCache.addToClientMap(cur)
				}

				pred := getClusterConditionPredicate()
				// only update when condition changed to ready from not-ready
				if !pred(*oldCluster) && pred(*curCluster) {
					s.clusterCache.addToClientMap(cur)
				}
				// did not handle ready -> not-ready
				// how could we stop a controller?
			},
		},
	)
	return s
}

// obj could be an *api.Ingress, or a DeletionFinalStateUnknown marker item.
func (s *IngressController) enqueueIngress(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	s.queue.Add(key)
}

// Run starts a background goroutine that watches for changes to federation ingresses
// and ensures that they have Kubernetes ingresses created, updated or deleted appropriately.
// federationSyncPeriod controls how often we check the federation's ingresses to
// ensure that the correct Kubernetes ingresses exist.
// This is only necessary to fudge over failed watches.
// clusterSyncPeriod controls how often we check the federation's underlying clusters and
// their Kubernetes ingresses to ensure that matching ingresses created independently of the Federation
// (e.g. directly via the underlying cluster's API) are correctly accounted for.

// It's an error to call Run() more than once for a given IngressController
// object.
func (s *IngressController) Run(workers int, stopCh <-chan struct{}) error {
	if err := s.init(); err != nil {
		return err
	}
	defer runtime.HandleCrash()
	go s.ingressController.Run(stopCh)
	go s.clusterController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(s.fedIngressWorker, time.Second, stopCh)
	}
	go wait.Until(s.clusterEndpointWorker, time.Second, stopCh)
	go wait.Until(s.clusterIngressWorker, time.Second, stopCh)
	go wait.Until(s.clusterSyncLoop, time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down Federation Ingress Controller")
	s.queue.ShutDown()
	return nil
}

func (ingress *IngressController) init() error {
	// TODO: quinton: Should not need DNS - excise this...
	if ingress.federationName == "" {
		return fmt.Errorf("IngressController should not be run without federationName.")
	}
	if ingress.zoneName == "" {
		return fmt.Errorf("IngressController should not be run without zoneName.")
	}
	if ingress.dns == nil {
		return fmt.Errorf("IngressController should not be run without a dnsprovider.")
	}
	zones, ok := s.dns.Zones()
	if !ok {
		return fmt.Errorf("the dns provider does not support zone enumeration, which is required for creating dns records.")
	}
	s.dnsZones = zones
	if _, err := getDnsZone(s.zoneName, s.dnsZones); err != nil {
		glog.Infof("DNS zone %q not found.  Creating DNS zone %q.", s.zoneName, s.zoneName)
		managedZone, err := s.dnsZones.New(s.zoneName)
		if err != nil {
			return err
		}
		zone, err := s.dnsZones.Add(managedZone)
		if err != nil {
			return err
		}
		glog.Infof("DNS zone %q successfully created.  Note that DNS resolution will not work until you have registered this name with "+
			"a DNS registrar and they have changed the authoritative name servers for your domain to point to your DNS provider.", zone.Name())
	}
	return nil
}

// fedIngressWorker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that syncIngress is never invoked concurrently with the same key.
func (ingressCtl *IngressController) fedIngressWorker() {
	for {
		func() {
			key, quit := ingressCtl.queue.Get()
			if quit {
				return
			}

			defer ingressCtl.queue.Done(key)
			err := ingressCtl.syncIngress(key.(string))
			if err != nil {
				glog.Errorf("Error syncing ingress: %v", err)
			}
		}()
	}
}

func wantsDNSRecords(ingress *v1beta1.Ingress) bool {
	// TODO: quinton: This is rubbish - fix it.
	return ingress.Spec.Type == v1.ServiceTypeLoadBalancer
}

// processIngressForCluster creates or updates ingress to all registered running clusters,
// update DNS records and update the ingress info with DNS entries to federation apiserver.
// the function returns any error caught
func (ingressCtl *IngressController) processIngressForCluster(cachedIngress *cachedIngress, clusterName string, ingress *v1beta.Ingress, client *release_1_4.Clientset) error {
	glog.V(4).Infof("Process ingress %s/%s for cluster %s", ingress.Namespace, ingress.Name, clusterName)
	// Create or Update k8s Ingress
	err := s.ensureClusterIngress(cachedIngress, clusterName, ingress, client)
	if err != nil {
		glog.V(4).Infof("Failed to process ingress %s/%s for cluster %s", ingress.Namespace, ingress.Name, clusterName)
		return err
	}
	glog.V(4).Infof("Successfully process ingress %s/%s for cluster %s", ingress.Namespace, ingress.Name, clusterName)
	return nil
}

// updateFederationIngress Returns whatever error occurred along with a boolean indicator of whether it
// should be retried.
func (s *IngressController) updateFederationIngress(key string, cachedIngress *cachedIngress) (error, bool) {
	// Clone federation ingress, and create them in underlying k8s cluster
	clone, err := conversion.NewCloner().DeepCopy(cachedIngress.lastState)
	if err != nil {
		return err, !retryable
	}
	ingress, ok := clone.(*v1.Ingress)
	if !ok {
		return fmt.Errorf("Unexpected ingress cast error : %v\n", ingress), !retryable
	}

	// handle available clusters one by one
	var hasErr bool
	for clusterName, cache := range s.clusterCache.clientMap {
		go func(cache *clusterCache, clusterName string) {
			err = s.processIngressForCluster(cachedIngress, clusterName, ingress, cache.clientset)
			if err != nil {
				hasErr = true
			}
		}(cache, clusterName)
	}
	if hasErr {
		// detail error has been dumped inside the loop
		return fmt.Errorf("Ingress %s/%s was not successfully updated to all clusters", ingress.Namespace, ingress.Name), retryable
	}
	return nil, !retryable
}

func (s *IngressController) deleteFederationIngress(cachedIngress *cachedIngress) (error, bool) {
	// handle available clusters one by one
	var hasErr bool
	for clusterName, cluster := range s.clusterCache.clientMap {
		err := s.deleteClusterIngress(clusterName, cachedIngress, cluster.clientset)
		if err != nil {
			hasErr = true
		} else if err := s.ensureDnsRecords(clusterName, cachedIngress); err != nil {
			hasErr = true
		}
	}
	if hasErr {
		// detail error has been dumpped inside the loop
		return fmt.Errorf("Ingress %s/%s was not successfully updated to all clusters", cachedIngress.lastState.Namespace, cachedIngress.lastState.Name), retryable
	}
	return nil, !retryable
}

func (s *IngressController) deleteClusterIngress(clusterName string, cachedIngress *cachedIngress, clientset *release_1_4.Clientset) error {
	ingress := cachedIngress.lastState
	glog.V(4).Infof("Deleting ingress %s/%s from cluster %s", ingress.Namespace, ingress.Name, clusterName)
	var err error
	for i := 0; i < clientRetryCount; i++ {
		err = clientset.Core().Ingresss(ingress.Namespace).Delete(ingress.Name, &api.DeleteOptions{})
		if err == nil || errors.IsNotFound(err) {
			glog.V(4).Infof("Ingress %s/%s deleted from cluster %s", ingress.Namespace, ingress.Name, clusterName)
			delete(cachedIngress.endpointMap, clusterName)
			return nil
		}
		time.Sleep(cachedIngress.nextRetryDelay())
	}
	glog.V(4).Infof("Failed to delete ingress %s/%s from cluster %s, %+v", ingress.Namespace, ingress.Name, clusterName, err)
	return err
}

func (s *IngressController) ensureClusterIngress(cachedIngress *cachedIngress, clusterName string, ingress *v1.Ingress, client *release_1_4.Clientset) error {
	var err error
	var needUpdate bool
	for i := 0; i < clientRetryCount; i++ {
		svc, err := client.Core().Ingresss(ingress.Namespace).Get(ingress.Name)
		if err == nil {
			// ingress exists
			glog.V(5).Infof("Found ingress %s/%s from cluster %s", ingress.Namespace, ingress.Name, clusterName)
			//reserve immutable fields
			ingress.Spec.ClusterIP = svc.Spec.ClusterIP

			//reserve auto assigned field
			for i, oldPort := range svc.Spec.Ports {
				for _, port := range ingress.Spec.Ports {
					if port.NodePort == 0 {
						if !portEqualExcludeNodePort(&oldPort, &port) {
							svc.Spec.Ports[i] = port
							needUpdate = true
						}
					} else {
						if !portEqualForLB(&oldPort, &port) {
							svc.Spec.Ports[i] = port
							needUpdate = true
						}
					}
				}
			}

			if needUpdate {
				// we only apply spec update
				svc.Spec = ingress.Spec
				_, err = client.Core().Ingresss(svc.Namespace).Update(svc)
				if err == nil {
					glog.V(5).Infof("Ingress %s/%s successfully updated to cluster %s", svc.Namespace, svc.Name, clusterName)
					return nil
				} else {
					glog.V(4).Infof("Failed to update %+v", err)
				}
			} else {
				glog.V(5).Infof("Ingress %s/%s is not updated to cluster %s as the spec are identical", svc.Namespace, svc.Name, clusterName)
				return nil
			}
		} else if errors.IsNotFound(err) {
			// Create ingress if it is not found
			glog.Infof("Ingress '%s/%s' is not found in cluster %s, trying to create new",
				ingress.Namespace, ingress.Name, clusterName)
			ingress.ResourceVersion = ""
			_, err = client.Core().Ingresss(ingress.Namespace).Create(ingress)
			if err == nil {
				glog.V(5).Infof("Ingress %s/%s successfully created to cluster %s", ingress.Namespace, ingress.Name, clusterName)
				return nil
			}
			glog.V(4).Infof("Failed to create %+v", err)
			if errors.IsAlreadyExists(err) {
				glog.V(5).Infof("ingress %s/%s already exists in cluster %s", ingress.Namespace, ingress.Name, clusterName)
				return nil
			}
		}
		if errors.IsConflict(err) {
			glog.V(4).Infof("Not persisting update to ingress '%s/%s' that has been changed since we received it: %v",
				ingress.Namespace, ingress.Name, err)
		}
		// should we reuse same retry delay for all clusters?
		time.Sleep(cachedIngress.nextRetryDelay())
	}
	return err
}

func (s *ingressCache) allIngresss() []*cachedIngress {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	ingresss := make([]*cachedIngress, 0, len(s.fedIngressMap))
	for _, v := range s.fedIngressMap {
		ingresss = append(ingresss, v)
	}
	return ingresss
}

func (s *ingressCache) get(ingressName string) (*cachedIngress, bool) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	ingress, ok := s.fedIngressMap[ingressName]
	return ingress, ok
}

func (s *ingressCache) getOrCreate(ingressName string) *cachedIngress {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	ingress, ok := s.fedIngressMap[ingressName]
	if !ok {
		ingress = &cachedIngress{
			endpointMap:      make(map[string]int),
			ingressStatusMap: make(map[string]v1.LoadBalancerStatus),
		}
		s.fedIngressMap[ingressName] = ingress
	}
	return ingress
}

func (s *ingressCache) set(ingressName string, ingress *cachedIngress) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	s.fedIngressMap[ingressName] = ingress
}

func (s *ingressCache) delete(ingressName string) {
	s.rwlock.Lock()
	defer s.rwlock.Unlock()
	delete(s.fedIngressMap, ingressName)
}

// needsUpdateDNS check if the dns records of the given ingress should be updated
func (s *IngressController) needsUpdateDNS(oldIngress *v1.Ingress, newIngress *v1.Ingress) bool {
	if !wantsDNSRecords(oldIngress) && !wantsDNSRecords(newIngress) {
		return false
	}
	if wantsDNSRecords(oldIngress) != wantsDNSRecords(newIngress) {
		s.eventRecorder.Eventf(newIngress, v1.EventTypeNormal, "Type", "%v -> %v",
			oldIngress.Spec.Type, newIngress.Spec.Type)
		return true
	}
	if !portsEqualForLB(oldIngress, newIngress) || oldIngress.Spec.SessionAffinity != newIngress.Spec.SessionAffinity {
		return true
	}
	if !LoadBalancerIPsAreEqual(oldIngress, newIngress) {
		s.eventRecorder.Eventf(newIngress, v1.EventTypeNormal, "LoadbalancerIP", "%v -> %v",
			oldIngress.Spec.LoadBalancerIP, newIngress.Spec.LoadBalancerIP)
		return true
	}
	if len(oldIngress.Spec.ExternalIPs) != len(newIngress.Spec.ExternalIPs) {
		s.eventRecorder.Eventf(newIngress, v1.EventTypeNormal, "ExternalIP", "Count: %v -> %v",
			len(oldIngress.Spec.ExternalIPs), len(newIngress.Spec.ExternalIPs))
		return true
	}
	for i := range oldIngress.Spec.ExternalIPs {
		if oldIngress.Spec.ExternalIPs[i] != newIngress.Spec.ExternalIPs[i] {
			s.eventRecorder.Eventf(newIngress, v1.EventTypeNormal, "ExternalIP", "Added: %v",
				newIngress.Spec.ExternalIPs[i])
			return true
		}
	}
	if !reflect.DeepEqual(oldIngress.Annotations, newIngress.Annotations) {
		return true
	}
	if oldIngress.UID != newIngress.UID {
		s.eventRecorder.Eventf(newIngress, v1.EventTypeNormal, "UID", "%v -> %v",
			oldIngress.UID, newIngress.UID)
		return true
	}

	return false
}

func getPortsForLB(ingress *v1.Ingress) ([]*v1.IngressPort, error) {
	// TODO: quinton: Probably applies for DNS SVC records.  Come back to this.
	//var protocol api.Protocol

	ports := []*v1.IngressPort{}
	for i := range ingress.Spec.Ports {
		sp := &ingress.Spec.Ports[i]
		// The check on protocol was removed here.  The DNS provider itself is now responsible for all protocol validation
		ports = append(ports, sp)
	}
	return ports, nil
}

func portsEqualForLB(x, y *v1.Ingress) bool {
	xPorts, err := getPortsForLB(x)
	if err != nil {
		return false
	}
	yPorts, err := getPortsForLB(y)
	if err != nil {
		return false
	}
	return portSlicesEqualForLB(xPorts, yPorts)
}

func portSlicesEqualForLB(x, y []*v1.IngressPort) bool {
	if len(x) != len(y) {
		return false
	}

	for i := range x {
		if !portEqualForLB(x[i], y[i]) {
			return false
		}
	}
	return true
}

func portEqualForLB(x, y *v1.IngressPort) bool {
	// TODO: Should we check name?  (In theory, an LB could expose it)
	if x.Name != y.Name {
		return false
	}

	if x.Protocol != y.Protocol {
		return false
	}

	if x.Port != y.Port {
		return false
	}
	if x.NodePort != y.NodePort {
		return false
	}
	return true
}

func portEqualExcludeNodePort(x, y *v1.IngressPort) bool {
	// TODO: Should we check name?  (In theory, an LB could expose it)
	if x.Name != y.Name {
		return false
	}

	if x.Protocol != y.Protocol {
		return false
	}

	if x.Port != y.Port {
		return false
	}
	return true
}

func clustersFromList(list *v1beta1.ClusterList) []string {
	result := []string{}
	for ix := range list.Items {
		result = append(result, list.Items[ix].Name)
	}
	return result
}

// getClusterConditionPredicate filters all clusters that meet the condition that
// condition.type=Ready and condition.status=true
func getClusterConditionPredicate() federationcache.ClusterConditionPredicate {
	return func(cluster v1beta1.Cluster) bool {
		// If we have no info, don't accept
		if len(cluster.Status.Conditions) == 0 {
			return false
		}
		for _, cond := range cluster.Status.Conditions {
			//We consider the cluster for load balancing only when its ClusterReady condition status
			//is ConditionTrue
			if cond.Type == v1beta1.ClusterReady && cond.Status != v1.ConditionTrue {
				glog.V(4).Infof("Ignoring cluser %v with %v condition status %v", cluster.Name, cond.Type, cond.Status)
				return false
			}
		}
		return true
	}
}

// clusterSyncLoop observes running clusters changes, and apply all ingresss to new added cluster
// and add dns records for the changes
func (s *IngressController) clusterSyncLoop() {
	var ingresssToUpdate []*cachedIngress
	// should we remove cache for cluster from ready to not ready? should remove the condition predicate if no
	clusters, err := s.clusterStore.ClusterCondition(getClusterConditionPredicate()).List()
	if err != nil {
		glog.Infof("Fail to get cluster list")
		return
	}
	newClusters := clustersFromList(&clusters)
	var newSet, increase sets.String
	newSet = sets.NewString(newClusters...)
	if newSet.Equal(s.knownClusterSet) {
		// The set of cluster names in the ingresss in the federation hasn't changed, but we can retry
		// updating any ingresss that we failed to update last time around.
		ingresssToUpdate = s.updateDNSRecords(ingresssToUpdate, newClusters)
		return
	}
	glog.Infof("Detected change in list of cluster names. New  set: %v, Old set: %v", newSet, s.knownClusterSet)
	increase = newSet.Difference(s.knownClusterSet)
	// do nothing when cluster is removed.
	if increase != nil {
		// Try updating all ingresss, and save the ones that fail to try again next
		// round.
		ingresssToUpdate = s.ingressCache.allIngresss()
		numIngresss := len(ingresssToUpdate)
		for newCluster := range increase {
			glog.Infof("New cluster observed %s", newCluster)
			s.updateAllIngresssToCluster(ingresssToUpdate, newCluster)
		}
		ingresssToUpdate = s.updateDNSRecords(ingresssToUpdate, newClusters)
		glog.Infof("Successfully updated %d out of %d DNS records to direct traffic to the updated cluster",
			numIngresss-len(ingresssToUpdate), numIngresss)
	}
	s.knownClusterSet = newSet
}

func (s *IngressController) updateAllIngresssToCluster(ingresss []*cachedIngress, clusterName string) {
	cluster, ok := s.clusterCache.clientMap[clusterName]
	if ok {
		for _, cachedIngress := range ingresss {
			appliedState := cachedIngress.lastState
			s.processIngressForCluster(cachedIngress, clusterName, appliedState, cluster.clientset)
		}
	}
}

func (s *IngressController) removeAllIngresssFromCluster(ingresss []*cachedIngress, clusterName string) {
	client, ok := s.clusterCache.clientMap[clusterName]
	if ok {
		for _, cachedIngress := range ingresss {
			s.deleteClusterIngress(clusterName, cachedIngress, client.clientset)
		}
		glog.Infof("Synced all ingresss to cluster %s", clusterName)
	}
}

// updateDNSRecords updates all existing federation ingress DNS Records so that
// they will match the list of cluster names provided.
// Returns the list of ingresss that couldn't be updated.
func (s *IngressController) updateDNSRecords(ingresss []*cachedIngress, clusters []string) (ingresssToRetry []*cachedIngress) {
	for _, ingress := range ingresss {
		func() {
			ingress.rwlock.Lock()
			defer ingress.rwlock.Unlock()
			// If the applied state is nil, that means it hasn't yet been successfully dealt
			// with by the DNS Record reconciler. We can trust the DNS Record
			// reconciler to ensure the federation ingress's DNS records are created to target
			// the correct backend ingress IP's
			if ingress.appliedState == nil {
				return
			}
			if err := s.lockedUpdateDNSRecords(ingress, clusters); err != nil {
				glog.Errorf("External error while updating DNS Records: %v.", err)
				ingresssToRetry = append(ingresssToRetry, ingress)
			}
		}()
	}
	return ingresssToRetry
}

// lockedUpdateDNSRecords Updates the DNS records of a ingress, assuming we hold the mutex
// associated with the ingress.
func (s *IngressController) lockedUpdateDNSRecords(ingress *cachedIngress, clusterNames []string) error {
	if !wantsDNSRecords(ingress.appliedState) {
		return nil
	}
	ensuredCount := 0
	for key := range s.clusterCache.clientMap {
		for _, clusterName := range clusterNames {
			if key == clusterName {
				s.ensureDnsRecords(clusterName, ingress)
				ensuredCount += 1
			}
		}
	}
	if ensuredCount < len(clusterNames) {
		return fmt.Errorf("Failed to update DNS records for %d of %d clusters for ingress %v due to missing clients for those clusters",
			len(clusterNames)-ensuredCount, len(clusterNames), ingress)
	}
	return nil
}

func LoadBalancerIPsAreEqual(oldIngress, newIngress *v1.Ingress) bool {
	return oldIngress.Spec.LoadBalancerIP == newIngress.Spec.LoadBalancerIP
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedIngress) nextRetryDelay() time.Duration {
	s.lastRetryDelay = s.lastRetryDelay * 2
	if s.lastRetryDelay < minRetryDelay {
		s.lastRetryDelay = minRetryDelay
	}
	if s.lastRetryDelay > maxRetryDelay {
		s.lastRetryDelay = maxRetryDelay
	}
	return s.lastRetryDelay
}

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedIngress) resetRetryDelay() {
	s.lastRetryDelay = time.Duration(0)
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedIngress) nextFedUpdateDelay() time.Duration {
	s.lastFedUpdateDelay = s.lastFedUpdateDelay * 2
	if s.lastFedUpdateDelay < minRetryDelay {
		s.lastFedUpdateDelay = minRetryDelay
	}
	if s.lastFedUpdateDelay > maxRetryDelay {
		s.lastFedUpdateDelay = maxRetryDelay
	}
	return s.lastFedUpdateDelay
}

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedIngress) resetFedUpdateDelay() {
	s.lastFedUpdateDelay = time.Duration(0)
}

// Computes the next retry, using exponential backoff
// mutex must be held.
func (s *cachedIngress) nextDNSUpdateDelay() time.Duration {
	s.lastDNSUpdateDelay = s.lastDNSUpdateDelay * 2
	if s.lastDNSUpdateDelay < minRetryDelay {
		s.lastDNSUpdateDelay = minRetryDelay
	}
	if s.lastDNSUpdateDelay > maxRetryDelay {
		s.lastDNSUpdateDelay = maxRetryDelay
	}
	return s.lastDNSUpdateDelay
}

// resetRetryDelay Resets the retry exponential backoff.  mutex must be held.
func (s *cachedIngress) resetDNSUpdateDelay() {
	s.lastDNSUpdateDelay = time.Duration(0)
}

// syncIngress will sync the Ingress with the given key if it has had its expectations fulfilled,
// meaning it did not expect to see any more of its pods created or deleted. This function is not meant to be
// invoked concurrently with the same key.
func (s *IngressController) syncIngress(key string) error {
	startTime := time.Now()
	var cachedIngress *cachedIngress
	var retryDelay time.Duration
	defer func() {
		glog.V(4).Infof("Finished syncing ingress %q (%v)", key, time.Now().Sub(startTime))
	}()
	// obj holds the latest ingress info from apiserver
	obj, exists, err := s.ingressStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve ingress %v from store: %v", key, err)
		s.queue.Add(key)
		return err
	}

	if !exists {
		// ingress absence in store means watcher caught the deletion, ensure LB info is cleaned
		glog.Infof("Ingress has been deleted %v", key)
		err, retryDelay = s.processIngressDeletion(key)
	}

	if exists {
		ingress, ok := obj.(*v1.Ingress)
		if ok {
			cachedIngress = s.ingressCache.getOrCreate(key)
			err, retryDelay = s.processIngressUpdate(cachedIngress, ingress, key)
		} else {
			tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
			if !ok {
				return fmt.Errorf("Object contained wasn't a ingress or a deleted key: %+v", obj)
			}
			glog.Infof("Found tombstone for %v", key)
			err, retryDelay = s.processIngressDeletion(tombstone.Key)
		}
	}

	if retryDelay != 0 {
		s.enqueueIngress(obj)
	} else if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to process ingress. Not retrying: %v", err))
	}
	return nil
}

// processIngressUpdate returns an error if processing the ingress update failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry in that Duration.
func (s *IngressController) processIngressUpdate(cachedIngress *cachedIngress, ingress *v1.Ingress, key string) (error, time.Duration) {
	// Ensure that no other goroutine will interfere with our processing of the
	// ingress.
	cachedIngress.rwlock.Lock()
	defer cachedIngress.rwlock.Unlock()

	// Update the cached ingress (used above for populating synthetic deletes)
	// alway trust ingress, which is retrieve from ingressStore, which keeps the latest ingress info getting from apiserver
	// if the same ingress is changed before this go routine finished, there will be another queue entry to handle that.
	cachedIngress.lastState = ingress
	err, retry := s.updateFederationIngress(key, cachedIngress)
	if err != nil {
		message := "Error occurs when updating ingress to all clusters"
		if retry {
			message += " (will retry): "
		} else {
			message += " (will not retry): "
		}
		message += err.Error()
		s.eventRecorder.Event(ingress, v1.EventTypeWarning, "UpdateIngressFail", message)
		return err, cachedIngress.nextRetryDelay()
	}
	// Always update the cache upon success.
	// NOTE: Since we update the cached ingress if and only if we successfully
	// processed it, a cached ingress being nil implies that it hasn't yet
	// been successfully processed.

	cachedIngress.appliedState = ingress
	s.ingressCache.set(key, cachedIngress)
	glog.V(4).Infof("Successfully procceeded ingresss %s", key)
	cachedIngress.resetRetryDelay()
	return nil, doNotRetry
}

// processIngressDeletion returns an error if processing the ingress deletion failed, along with a time.Duration
// indicating whether processing should be retried; zero means no-retry; otherwise
// we should retry in that Duration.
func (s *IngressController) processIngressDeletion(key string) (error, time.Duration) {
	glog.V(2).Infof("Process ingress deletion for %v", key)
	cachedIngress, ok := s.ingressCache.get(key)
	if !ok {
		return fmt.Errorf("Ingress %s not in cache even though the watcher thought it was. Ignoring the deletion.", key), doNotRetry
	}
	ingress := cachedIngress.lastState
	cachedIngress.rwlock.Lock()
	defer cachedIngress.rwlock.Unlock()
	s.eventRecorder.Event(ingress, v1.EventTypeNormal, "DeletingDNSRecord", "Deleting DNS Records")
	// TODO should we delete dns info here or wait for endpoint changes? prefer here
	// or we do nothing for ingress deletion
	//err := s.dns.balancer.EnsureLoadBalancerDeleted(ingress)
	err, retry := s.deleteFederationIngress(cachedIngress)
	if err != nil {
		message := "Error occurs when deleting federation ingress"
		if retry {
			message += " (will retry): "
		} else {
			message += " (will not retry): "
		}
		s.eventRecorder.Event(ingress, v1.EventTypeWarning, "DeletingDNSRecordFailed", message)
		return err, cachedIngress.nextRetryDelay()
	}
	s.eventRecorder.Event(ingress, v1.EventTypeNormal, "DeletedDNSRecord", "Deleted DNS Records")
	s.ingressCache.delete(key)

	cachedIngress.resetRetryDelay()
	return nil, doNotRetry
}
