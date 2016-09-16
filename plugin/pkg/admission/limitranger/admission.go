/*
Copyright 2014 The Kubernetes Authors.

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

package limitranger

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	lru "github.com/hashicorp/golang-lru"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	limitRangerAnnotation = "kubernetes.io/limit-ranger"
)

func init() {
	admission.RegisterPlugin("LimitRanger", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		return NewLimitRanger(client, &DefaultLimitRangerActions{})
	})
}

// limitRanger enforces usage limits on a per resource basis in the namespace
type limitRanger struct {
	*admission.Handler
	client  clientset.Interface
	actions LimitRangerActions
	indexer cache.Indexer

	// liveLookups holds the last few live lookups we've done to help ammortize cost on repeated lookup failures.
	// This let's us handle the case of latent caches, by looking up actual results for a namespace on cache miss/no results.
	// We track the lookup result here so that for repeated requests, we don't look it up very often.
	liveLookupCache *lru.Cache
	liveTTL         time.Duration
}

type liveLookupEntry struct {
	expiry time.Time
	items  []*api.LimitRange
}

// Admit admits resources into cluster that do not violate any defined LimitRange in the namespace
func (l *limitRanger) Admit(a admission.Attributes) (err error) {
	if !l.actions.SupportsAttributes(a) {
		return nil
	}

	obj := a.GetObject()
	name := "Unknown"
	if obj != nil {
		name, _ = meta.NewAccessor().Name(obj)
		if len(name) == 0 {
			name, _ = meta.NewAccessor().GenerateName(obj)
		}
	}

	key := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Namespace: a.GetNamespace(),
			Name:      "",
		},
	}
	items, err := l.indexer.Index("namespace", key)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("unable to %s %v at this time because there was an error enforcing limit ranges", a.GetOperation(), a.GetResource()))
	}

	// if there are no items held in our indexer, check our live-lookup LRU, if that misses, do the live lookup to prime it.
	if len(items) == 0 {
		lruItemObj, ok := l.liveLookupCache.Get(a.GetNamespace())
		if !ok || lruItemObj.(liveLookupEntry).expiry.Before(time.Now()) {
			// TODO: If there are multiple operations at the same time and cache has just expired,
			// this may cause multiple List operations being issued at the same time.
			// If there is already in-flight List() for a given namespace, we should wait until
			// it is finished and cache is updated instead of doing the same, also to avoid
			// throttling - see #22422 for details.
			liveList, err := l.client.Core().LimitRanges(a.GetNamespace()).List(api.ListOptions{})
			if err != nil {
				return admission.NewForbidden(a, err)
			}
			newEntry := liveLookupEntry{expiry: time.Now().Add(l.liveTTL)}
			for i := range liveList.Items {
				newEntry.items = append(newEntry.items, &liveList.Items[i])
			}
			l.liveLookupCache.Add(a.GetNamespace(), newEntry)
			lruItemObj = newEntry
		}
		lruEntry := lruItemObj.(liveLookupEntry)

		for i := range lruEntry.items {
			items = append(items, lruEntry.items[i])
		}

	}

	// ensure it meets each prescribed min/max
	for i := range items {
		limitRange := items[i].(*api.LimitRange)

		if !l.actions.SupportsLimit(limitRange) {
			continue
		}

		err = l.actions.Limit(limitRange, a.GetResource().Resource, a.GetObject())
		if err != nil {
			return admission.NewForbidden(a, err)
		}
	}
	return nil
}

// NewLimitRanger returns an object that enforces limits based on the supplied limit function
func NewLimitRanger(client clientset.Interface, actions LimitRangerActions) (admission.Interface, error) {
	liveLookupCache, err := lru.New(10000)
	if err != nil {
		return nil, err
	}

	lw := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return client.Core().LimitRanges(api.NamespaceAll).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return client.Core().LimitRanges(api.NamespaceAll).Watch(options)
		},
	}
	indexer, reflector := cache.NewNamespaceKeyedIndexerAndReflector(lw, &api.LimitRange{}, 0)
	reflector.Run()

	if actions == nil {
		actions = &DefaultLimitRangerActions{}
	}

	return &limitRanger{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		client:          client,
		actions:         actions,
		indexer:         indexer,
		liveLookupCache: liveLookupCache,
		liveTTL:         time.Duration(30 * time.Second),
	}, nil
}

// Min returns the lesser of its 2 arguments
func Min(a int64, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// Max returns the greater of its 2 arguments
func Max(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// defaultContainerResourceRequirements returns the default requirements for a container
// the requirement.Limits are taken from the LimitRange defaults (if specified)
// the requirement.Requests are taken from the LimitRange default request (if specified)
func defaultContainerResourceRequirements(limitRange *api.LimitRange) api.ResourceRequirements {
	requirements := api.ResourceRequirements{}
	requirements.Requests = api.ResourceList{}
	requirements.Limits = api.ResourceList{}

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		if limit.Type == api.LimitTypeContainer {
			for k, v := range limit.DefaultRequest {
				value := v.Copy()
				requirements.Requests[k] = *value
			}
			for k, v := range limit.Default {
				value := v.Copy()
				requirements.Limits[k] = *value
			}
		}
	}
	return requirements
}

// mergeContainerResources handles defaulting all of the resources on a container.
func mergeContainerResources(container *api.Container, defaultRequirements *api.ResourceRequirements, annotationPrefix string, annotations []string) []string {
	setRequests := []string{}
	setLimits := []string{}
	if container.Resources.Limits == nil {
		container.Resources.Limits = api.ResourceList{}
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = api.ResourceList{}
	}
	for k, v := range defaultRequirements.Limits {
		_, found := container.Resources.Limits[k]
		if !found {
			container.Resources.Limits[k] = *v.Copy()
			setLimits = append(setLimits, string(k))
		}
	}
	for k, v := range defaultRequirements.Requests {
		_, found := container.Resources.Requests[k]
		if !found {
			container.Resources.Requests[k] = *v.Copy()
			setRequests = append(setRequests, string(k))
		}
	}
	if len(setRequests) > 0 {
		sort.Strings(setRequests)
		a := strings.Join(setRequests, ", ") + fmt.Sprintf(" request for %s %s", annotationPrefix, container.Name)
		annotations = append(annotations, a)
	}
	if len(setLimits) > 0 {
		sort.Strings(setLimits)
		a := strings.Join(setLimits, ", ") + fmt.Sprintf(" limit for %s %s", annotationPrefix, container.Name)
		annotations = append(annotations, a)
	}
	return annotations
}

// mergePodResourceRequirements merges enumerated requirements with default requirements
// it annotates the pod with information about what requirements were modified
func mergePodResourceRequirements(pod *api.Pod, defaultRequirements *api.ResourceRequirements) {
	annotations := []string{}

	for i := range pod.Spec.Containers {
		annotations = mergeContainerResources(&pod.Spec.Containers[i], defaultRequirements, "container", annotations)
	}

	for i := range pod.Spec.InitContainers {
		annotations = mergeContainerResources(&pod.Spec.InitContainers[i], defaultRequirements, "init container", annotations)
	}

	if len(annotations) > 0 {
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = make(map[string]string)
		}
		val := "LimitRanger plugin set: " + strings.Join(annotations, "; ")
		pod.ObjectMeta.Annotations[limitRangerAnnotation] = val
	}
}

// requestLimitEnforcedValues returns the specified values at a common precision to support comparability
func requestLimitEnforcedValues(requestQuantity, limitQuantity, enforcedQuantity resource.Quantity) (request, limit, enforced int64) {
	request = requestQuantity.Value()
	limit = limitQuantity.Value()
	enforced = enforcedQuantity.Value()
	// do a more precise comparison if possible (if the value won't overflow)
	if request <= resource.MaxMilliValue && limit <= resource.MaxMilliValue && enforced <= resource.MaxMilliValue {
		request = requestQuantity.MilliValue()
		limit = limitQuantity.MilliValue()
		enforced = enforcedQuantity.MilliValue()
	}
	return
}

// minConstraint enforces the min constraint over the specified resource
func minConstraint(limitType api.LimitType, resourceName api.ResourceName, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[resourceName]
	lim, limExists := limit[resourceName]
	observedReqValue, observedLimValue, enforcedValue := requestLimitEnforcedValues(req, lim, enforced)

	if !reqExists {
		return fmt.Errorf("minimum %s usage per %s is %s.  No request is specified.", resourceName, limitType, enforced.String())
	}
	if observedReqValue < enforcedValue {
		return fmt.Errorf("minimum %s usage per %s is %s, but request is %s.", resourceName, limitType, enforced.String(), req.String())
	}
	if limExists && (observedLimValue < enforcedValue) {
		return fmt.Errorf("minimum %s usage per %s is %s, but limit is %s.", resourceName, limitType, enforced.String(), lim.String())
	}
	return nil
}

// maxConstraint enforces the max constraint over the specified resource
func maxConstraint(limitType api.LimitType, resourceName api.ResourceName, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[resourceName]
	lim, limExists := limit[resourceName]
	observedReqValue, observedLimValue, enforcedValue := requestLimitEnforcedValues(req, lim, enforced)

	if !limExists {
		return fmt.Errorf("maximum %s usage per %s is %s.  No limit is specified.", resourceName, limitType, enforced.String())
	}
	if observedLimValue > enforcedValue {
		return fmt.Errorf("maximum %s usage per %s is %s, but limit is %s.", resourceName, limitType, enforced.String(), lim.String())
	}
	if reqExists && (observedReqValue > enforcedValue) {
		return fmt.Errorf("maximum %s usage per %s is %s, but request is %s.", resourceName, limitType, enforced.String(), req.String())
	}
	return nil
}

// limitRequestRatioConstraint enforces the limit to request ratio over the specified resource
func limitRequestRatioConstraint(limitType api.LimitType, resourceName api.ResourceName, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[resourceName]
	lim, limExists := limit[resourceName]
	observedReqValue, observedLimValue, _ := requestLimitEnforcedValues(req, lim, enforced)

	if !reqExists || (observedReqValue == int64(0)) {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but no request is specified or request is 0.", resourceName, limitType, enforced.String())
	}
	if !limExists || (observedLimValue == int64(0)) {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but no limit is specified or limit is 0.", resourceName, limitType, enforced.String())
	}

	observedRatio := float64(observedLimValue) / float64(observedReqValue)
	displayObservedRatio := observedRatio
	maxLimitRequestRatio := float64(enforced.Value())
	if enforced.Value() <= resource.MaxMilliValue {
		observedRatio = observedRatio * 1000
		maxLimitRequestRatio = float64(enforced.MilliValue())
	}

	if observedRatio > maxLimitRequestRatio {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but provided ratio is %f.", resourceName, limitType, enforced.String(), displayObservedRatio)
	}

	return nil
}

// sum takes the total of each named resource across all inputs
// if a key is not in each input, then the output resource list will omit the key
func sum(inputs []api.ResourceList) api.ResourceList {
	result := api.ResourceList{}
	keys := []api.ResourceName{}
	for i := range inputs {
		for k := range inputs[i] {
			keys = append(keys, k)
		}
	}
	for _, key := range keys {
		total, isSet := int64(0), true

		for i := range inputs {
			input := inputs[i]
			v, exists := input[key]
			if exists {
				if key == api.ResourceCPU {
					total = total + v.MilliValue()
				} else {
					total = total + v.Value()
				}
			} else {
				isSet = false
			}
		}

		if isSet {
			if key == api.ResourceCPU {
				result[key] = *(resource.NewMilliQuantity(total, resource.DecimalSI))
			} else {
				result[key] = *(resource.NewQuantity(total, resource.DecimalSI))
			}

		}
	}
	return result
}

// DefaultLimitRangerActions is the default implementation of LimitRangerActions.
type DefaultLimitRangerActions struct{}

// ensure DefaultLimitRangerActions implements the LimitRangerActions interface.
var _ LimitRangerActions = &DefaultLimitRangerActions{}

// Limit enforces resource requirements of incoming resources against enumerated constraints
// on the LimitRange.  It may modify the incoming object to apply default resource requirements
// if not specified, and enumerated on the LimitRange
func (d *DefaultLimitRangerActions) Limit(limitRange *api.LimitRange, resourceName string, obj runtime.Object) error {
	switch resourceName {
	case "pods":
		return PodLimitFunc(limitRange, obj.(*api.Pod))
	}
	return nil
}

// SupportsAttributes ignores all calls that do not deal with pod resources since that is
// all this supports now.  Also ignores any call that has a subresource defined.
func (d *DefaultLimitRangerActions) SupportsAttributes(a admission.Attributes) bool {
	if a.GetSubresource() != "" {
		return false
	}

	return a.GetKind().GroupKind() == api.Kind("Pod")
}

// SupportsLimit always returns true.
func (d *DefaultLimitRangerActions) SupportsLimit(limitRange *api.LimitRange) bool {
	return true
}

// PodLimitFunc enforces resource requirements enumerated by the pod against
// the specified LimitRange.  The pod may be modified to apply default resource
// requirements if not specified, and enumerated on the LimitRange
func PodLimitFunc(limitRange *api.LimitRange, pod *api.Pod) error {
	var errs []error

	defaultResources := defaultContainerResourceRequirements(limitRange)
	mergePodResourceRequirements(pod, &defaultResources)

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		limitType := limit.Type
		// enforce container limits
		if limitType == api.LimitTypeContainer {
			for j := range pod.Spec.Containers {
				container := &pod.Spec.Containers[j]
				for k, v := range limit.Min {
					if err := minConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
				for k, v := range limit.Max {
					if err := maxConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
				for k, v := range limit.MaxLimitRequestRatio {
					if err := limitRequestRatioConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
			}
			for j := range pod.Spec.InitContainers {
				container := &pod.Spec.InitContainers[j]
				for k, v := range limit.Min {
					if err := minConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
				for k, v := range limit.Max {
					if err := maxConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
				for k, v := range limit.MaxLimitRequestRatio {
					if err := limitRequestRatioConstraint(limitType, k, v, container.Resources.Requests, container.Resources.Limits); err != nil {
						errs = append(errs, err)
					}
				}
			}
		}

		// enforce pod limits on init containers
		if limitType == api.LimitTypePod {
			containerRequests, containerLimits := []api.ResourceList{}, []api.ResourceList{}
			for j := range pod.Spec.Containers {
				container := &pod.Spec.Containers[j]
				containerRequests = append(containerRequests, container.Resources.Requests)
				containerLimits = append(containerLimits, container.Resources.Limits)
			}
			podRequests := sum(containerRequests)
			podLimits := sum(containerLimits)
			for j := range pod.Spec.InitContainers {
				container := &pod.Spec.InitContainers[j]
				// take max(sum_containers, any_init_container)
				for k, v := range container.Resources.Requests {
					if v2, ok := podRequests[k]; ok {
						if v.Cmp(v2) > 0 {
							podRequests[k] = v
						}
					} else {
						podRequests[k] = v
					}
				}
				for k, v := range container.Resources.Limits {
					if v2, ok := podLimits[k]; ok {
						if v.Cmp(v2) > 0 {
							podLimits[k] = v
						}
					} else {
						podLimits[k] = v
					}
				}
			}
			for k, v := range limit.Min {
				if err := minConstraint(limitType, k, v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
			for k, v := range limit.Max {
				if err := maxConstraint(limitType, k, v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
			for k, v := range limit.MaxLimitRequestRatio {
				if err := limitRequestRatioConstraint(limitType, k, v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}
