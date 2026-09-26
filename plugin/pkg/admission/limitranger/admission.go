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
	"cmp"
	"context"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"strings"
	"time"

	"golang.org/x/sync/singleflight"
	inf "gopkg.in/inf.v0"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitailizer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/utils/lru"

	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	limitRangerAnnotation = "kubernetes.io/limit-ranger"
	// PluginName indicates name of admission plugin.
	PluginName = "LimitRanger"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewLimitRanger(&DefaultLimitRangerActions{})
	})
}

// LimitRanger enforces usage limits on a per resource basis in the namespace
type LimitRanger struct {
	*admission.Handler
	client  kubernetes.Interface
	actions LimitRangerActions
	lister  corev1listers.LimitRangeLister

	// liveLookups holds the last few live lookups we've done to help ammortize cost on repeated lookup failures.
	// This let's us handle the case of latent caches, by looking up actual results for a namespace on cache miss/no results.
	// We track the lookup result here so that for repeated requests, we don't look it up very often.
	liveLookupCache *lru.Cache
	group           singleflight.Group
	liveTTL         time.Duration
}

var _ admission.MutationInterface = &LimitRanger{}
var _ admission.ValidationInterface = &LimitRanger{}

var _ genericadmissioninitailizer.WantsExternalKubeInformerFactory = &LimitRanger{}
var _ genericadmissioninitailizer.WantsExternalKubeClientSet = &LimitRanger{}

type liveLookupEntry struct {
	expiry time.Time
	items  []*corev1.LimitRange
}

// SetExternalKubeInformerFactory registers an informer factory into the LimitRanger
func (l *LimitRanger) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	limitRangeInformer := f.Core().V1().LimitRanges()
	l.SetReadyFunc(limitRangeInformer.Informer().HasSynced)
	l.lister = limitRangeInformer.Lister()
}

// SetExternalKubeClientSet registers the client into LimitRanger
func (l *LimitRanger) SetExternalKubeClientSet(client kubernetes.Interface) {
	l.client = client
}

// ValidateInitialization verifies the LimitRanger object has been properly initialized
func (l *LimitRanger) ValidateInitialization() error {
	if l.lister == nil {
		return fmt.Errorf("missing limitRange lister")
	}
	if l.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

// Admit admits resources into cluster that do not violate any defined LimitRange in the namespace
func (l *LimitRanger) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	return l.runLimitFunc(a, func(limitRange *corev1.LimitRange) error {
		return l.actions.MutateLimit(limitRange, a.GetResource().Resource, a.GetObject())
	})
}

// Validate admits resources into cluster that do not violate any defined LimitRange in the namespace
func (l *LimitRanger) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	return l.runLimitFunc(a, func(limitRange *corev1.LimitRange) error {
		return l.actions.ValidateLimit(limitRange, a.GetResource().Resource, a.GetObject(), a.GetOldObject())
	})
}

func (l *LimitRanger) runLimitFunc(a admission.Attributes, limitFn func(limitRange *corev1.LimitRange) error) (err error) {
	if !l.actions.SupportsAttributes(a) {
		return nil
	}

	// ignore all objects marked for deletion
	oldObj := a.GetOldObject()
	if oldObj != nil {
		oldAccessor, err := meta.Accessor(oldObj)
		if err != nil {
			return admission.NewForbidden(a, err)
		}
		if oldAccessor.GetDeletionTimestamp() != nil {
			return nil
		}
	}

	items, err := l.GetLimitRanges(a)
	if err != nil {
		return err
	}

	// ensure it meets each prescribed min/max
	for i := range items {
		limitRange := items[i]

		if !l.actions.SupportsLimit(limitRange) {
			continue
		}

		err = limitFn(limitRange)
		if err != nil {
			return admission.NewForbidden(a, err)
		}
	}
	return nil
}

// GetLimitRanges returns a LimitRange object with the items held in
// the indexer if available, or do alive lookup of the value.
func (l *LimitRanger) GetLimitRanges(a admission.Attributes) ([]*corev1.LimitRange, error) {
	items, err := l.lister.LimitRanges(a.GetNamespace()).List(labels.Everything())
	if err != nil {
		return nil, admission.NewForbidden(a, fmt.Errorf("unable to %s %v at this time because there was an error enforcing limit ranges", a.GetOperation(), a.GetResource()))
	}

	// if there are no items held in our indexer, check our live-lookup LRU, if that misses, do the live lookup to prime it.
	if len(items) == 0 {
		lruItemObj, ok := l.liveLookupCache.Get(a.GetNamespace())
		if !ok || lruItemObj.(liveLookupEntry).expiry.Before(time.Now()) {
			// Fixed: #22422
			// use singleflight to alleviate simultaneous calls to
			lruItemObj, err, _ = l.group.Do(a.GetNamespace(), func() (interface{}, error) {
				liveList, err := l.client.CoreV1().LimitRanges(a.GetNamespace()).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					return nil, admission.NewForbidden(a, err)
				}
				newEntry := liveLookupEntry{expiry: time.Now().Add(l.liveTTL)}
				for i := range liveList.Items {
					newEntry.items = append(newEntry.items, &liveList.Items[i])
				}
				l.liveLookupCache.Add(a.GetNamespace(), newEntry)
				return newEntry, nil
			})
			if err != nil {
				return nil, err
			}
		}
		lruEntry := lruItemObj.(liveLookupEntry)

		items = append(items, lruEntry.items...)

	}

	return items, nil
}

// NewLimitRanger returns an object that enforces limits based on the supplied limit function
func NewLimitRanger(actions LimitRangerActions) (*LimitRanger, error) {
	liveLookupCache := lru.New(10000)

	if actions == nil {
		actions = &DefaultLimitRangerActions{}
	}

	return &LimitRanger{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		actions:         actions,
		liveLookupCache: liveLookupCache,
		liveTTL:         time.Duration(30 * time.Second),
	}, nil
}

// defaultContainerResourceRequirements returns the default requirements for a container
// the requirement.Limits are taken from the LimitRange defaults (if specified)
// the requirement.Requests are taken from the LimitRange default request (if specified)
func defaultContainerResourceRequirements(limitRange *corev1.LimitRange) api.ResourceRequirements {
	requirements := api.ResourceRequirements{}
	requirements.Requests = api.ResourceList{}
	requirements.Limits = api.ResourceList{}

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		if limit.Type == corev1.LimitTypeContainer {
			for k, v := range limit.DefaultRequest {
				requirements.Requests[api.ResourceName(k)] = v.DeepCopy()
			}
			for k, v := range limit.Default {
				requirements.Limits[api.ResourceName(k)] = v.DeepCopy()
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
			container.Resources.Limits[k] = v.DeepCopy()
			setLimits = append(setLimits, string(k))
		}
	}
	for k, v := range defaultRequirements.Requests {
		_, found := container.Resources.Requests[k]
		if !found {
			container.Resources.Requests[k] = v.DeepCopy()
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

// exceedsAllowed reports whether limit is greater than request times ratio,
// comparing magnitudes.
// Quantity has no multiply and inf.Dec.Mul adds the two scales as an int32, so
// the product is formed here with the scale kept in int64.
func exceedsAllowed(limit, request, ratio resource.Quantity) bool {
	// The ratio of a negative pair is the one between its magnitudes.
	limit, request = magnitude(limit), magnitude(request)
	limitDec, requestDec, ratioDec := limit.AsDec(), request.AsDec(), ratio.AsDec()
	scale := int64(requestDec.Scale()) + int64(ratioDec.Scale())
	productUnscaled := new(big.Int).Mul(requestDec.UnscaledBig(), ratioDec.UnscaledBig())
	if scale < math.MinInt32 {
		// The product has no representable scale, so compare it as digits and a scale.
		return cmpScaled(limitDec.UnscaledBig(), int64(limitDec.Scale()), productUnscaled, scale) > 0
	}
	product := new(inf.Dec)
	product.SetUnscaledBig(productUnscaled)
	product.SetScale(inf.Scale(scale))
	allowed := resource.NewDecimalQuantity(*product, resource.DecimalSI)
	return limit.Cmp(*allowed) > 0
}

// magnitude returns q without a negative sign, leaving q unchanged.
func magnitude(q resource.Quantity) resource.Quantity {
	if q.Sign() >= 0 {
		return q
	}
	q = q.DeepCopy()
	q.Neg()
	return q
}

// cmpScaled compares a*10^-sa with b*10^-sb for a non-negative a.
// A scale past inf.Scale has no Quantity for Cmp to take.
func cmpScaled(a *big.Int, sa int64, b *big.Int, sb int64) int {
	if a.Sign() <= 0 || b.Sign() <= 0 {
		return cmp.Compare(a.Sign(), b.Sign())
	}
	// 10^(ea-1) <= a*10^-sa < 10^ea, and likewise for b.
	ea := int64(len(a.Text(10))) - sa
	eb := int64(len(b.Text(10))) - sb
	if c := cmp.Compare(ea, eb); c != 0 {
		return c
	}
	// Equal exponents mean sb-sa is the digit-count difference, a small shift.
	if shift := sb - sa; shift >= 0 {
		a = mulPow10(a, shift)
	} else {
		b = mulPow10(b, -shift)
	}
	return a.Cmp(b)
}

const (
	// ratioScale is the six decimals the ratio message has always shown.
	ratioScale inf.Scale = 6
	// ratioFoldLimit is the scale difference past which QuoRound would build
	// the whole exponent.
	ratioFoldLimit int64 = 40
)

// ratioString formats the magnitude of limit over request to six decimals,
// of the mantissa once it uses an exponent.
func ratioString(limit, request resource.Quantity) string {
	limit, request = magnitude(limit), magnitude(request)
	limitDec, requestDec := limit.AsDec(), request.AsDec()
	exp := int64(requestDec.Scale()) - int64(limitDec.Scale())
	if exp >= -ratioFoldLimit && exp <= ratioFoldLimit {
		return new(inf.Dec).QuoRound(limitDec, requestDec, ratioScale, inf.RoundHalfEven).String()
	}
	// The ratio is the coefficient quotient times ten to the exp. Bring that
	// quotient into [1,10) first, or one below the six decimals reads as zero.
	num, den := limitDec.UnscaledBig(), requestDec.UnscaledBig()
	shift := int64(len(den.Text(10)) - len(num.Text(10)))
	if shift >= 0 {
		num = mulPow10(num, shift)
	} else {
		den = mulPow10(den, -shift)
	}
	if num.Cmp(den) < 0 {
		num = new(big.Int).Mul(num, big.NewInt(10))
		shift++
	}
	mantissa := new(inf.Dec).QuoRound(
		new(inf.Dec).SetUnscaledBig(num),
		new(inf.Dec).SetUnscaledBig(den),
		ratioScale, inf.RoundHalfEven)
	// Rounding can carry the mantissa to ten. Dividing it back moves the
	// opposite way from the shift above, so the exponent goes up by one.
	if ten := inf.NewDec(10, 0); mantissa.Cmp(ten) >= 0 {
		mantissa = new(inf.Dec).QuoRound(mantissa, ten, ratioScale, inf.RoundHalfEven)
		shift--
	}
	exp -= shift
	if exp == 0 {
		return mantissa.String()
	}
	return fmt.Sprintf("%se%d", mantissa, exp)
}

// mulPow10 returns a new big.Int holding x times ten to the n, for n >= 0.
func mulPow10(x *big.Int, n int64) *big.Int {
	return new(big.Int).Mul(x, new(big.Int).Exp(big.NewInt(10), big.NewInt(n), nil))
}

// minConstraint enforces the min constraint over the specified resource
func minConstraint(limitType string, resourceName string, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[api.ResourceName(resourceName)]
	lim, limExists := limit[api.ResourceName(resourceName)]

	if !reqExists {
		return fmt.Errorf("minimum %s usage per %s is %s.  No request is specified", resourceName, limitType, enforced.String())
	}
	if enforced.Cmp(req) > 0 {
		return fmt.Errorf("minimum %s usage per %s is %s, but request is %s", resourceName, limitType, enforced.String(), req.String())
	}
	if limExists && enforced.Cmp(lim) > 0 {
		return fmt.Errorf("minimum %s usage per %s is %s, but limit is %s", resourceName, limitType, enforced.String(), lim.String())
	}
	return nil
}

// maxRequestConstraint enforces the max constraint over the specified resource
// use when specify LimitType resource doesn't recognize limit values
func maxRequestConstraint(limitType string, resourceName string, enforced resource.Quantity, request api.ResourceList) error {
	req, reqExists := request[api.ResourceName(resourceName)]

	if !reqExists {
		return fmt.Errorf("maximum %s usage per %s is %s.  No request is specified", resourceName, limitType, enforced.String())
	}
	if req.Cmp(enforced) > 0 {
		return fmt.Errorf("maximum %s usage per %s is %s, but request is %s", resourceName, limitType, enforced.String(), req.String())
	}
	return nil
}

// maxConstraint enforces the max constraint over the specified resource
func maxConstraint(limitType string, resourceName string, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[api.ResourceName(resourceName)]
	lim, limExists := limit[api.ResourceName(resourceName)]

	if !limExists {
		return fmt.Errorf("maximum %s usage per %s is %s.  No limit is specified", resourceName, limitType, enforced.String())
	}
	if lim.Cmp(enforced) > 0 {
		return fmt.Errorf("maximum %s usage per %s is %s, but limit is %s", resourceName, limitType, enforced.String(), lim.String())
	}
	if reqExists && req.Cmp(enforced) > 0 {
		return fmt.Errorf("maximum %s usage per %s is %s, but request is %s", resourceName, limitType, enforced.String(), req.String())
	}
	return nil
}

// limitRequestRatioConstraint enforces the limit to request ratio over the specified resource
func limitRequestRatioConstraint(limitType string, resourceName string, enforced resource.Quantity, request api.ResourceList, limit api.ResourceList) error {
	req, reqExists := request[api.ResourceName(resourceName)]
	lim, limExists := limit[api.ResourceName(resourceName)]

	if !reqExists || req.Sign() == 0 {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but no request is specified or request is 0", resourceName, limitType, enforced.String())
	}
	if !limExists || lim.Sign() == 0 {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but no limit is specified or limit is 0", resourceName, limitType, enforced.String())
	}

	if exceedsAllowed(lim, req, enforced) {
		return fmt.Errorf("%s max limit to request ratio per %s is %s, but provided ratio is %s", resourceName, limitType, enforced.String(), ratioString(lim, req))
	}

	return nil
}

// DefaultLimitRangerActions is the default implementation of LimitRangerActions.
type DefaultLimitRangerActions struct{}

// ensure DefaultLimitRangerActions implements the LimitRangerActions interface.
var _ LimitRangerActions = &DefaultLimitRangerActions{}

// MutateLimit enforces resource requirements of incoming resources
// against enumerated constraints on the LimitRange.  It may modify
// the incoming object to apply default resource requirements if not
// specified, and enumerated on the LimitRange
func (d *DefaultLimitRangerActions) MutateLimit(limitRange *corev1.LimitRange, resourceName string, obj runtime.Object) error {
	switch resourceName {
	case "pods":
		return PodMutateLimitFunc(limitRange, obj.(*api.Pod))
	}
	return nil
}

// ValidateLimit verifies the resource requirements of incoming
// resources against enumerated constraints on the LimitRange are
// valid
func (d *DefaultLimitRangerActions) ValidateLimit(limitRange *corev1.LimitRange, resourceName string, obj, oldObj runtime.Object) error {
	switch resourceName {
	case "pods":
		oldPod, _ := oldObj.(*api.Pod)
		return PodValidateLimitFunc(limitRange, obj.(*api.Pod), oldPod)
	case "persistentvolumeclaims":
		oldPVC, _ := oldObj.(*api.PersistentVolumeClaim)
		return PersistentVolumeClaimValidateLimitFunc(limitRange, obj.(*api.PersistentVolumeClaim), oldPVC)
	}
	return nil
}

// SupportsAttributes ignores all calls that do not deal with pod resources or storage requests (PVCs).
// Also ignores any call that has a subresource defined.
func (d *DefaultLimitRangerActions) SupportsAttributes(a admission.Attributes) bool {
	// Handle in-place vertical scaling of pods, where users modify container
	// resources using the resize subresource.
	if a.GetSubresource() == "resize" && a.GetKind().GroupKind() == api.Kind("Pod") && a.GetOperation() == admission.Update {
		return true
	}

	// No other subresources are supported
	if a.GetSubresource() != "" {
		return false
	}

	// Since containers and initContainers cannot currently be added, removed, or updated, it is unnecessary
	// to mutate and validate limitrange on pod updates. Trying to mutate containers or initContainers on a pod
	// update request will always fail pod validation because those fields are immutable once the object is created.
	if a.GetKind().GroupKind() == api.Kind("Pod") && a.GetOperation() == admission.Update {
		return false
	}

	return a.GetKind().GroupKind() == api.Kind("Pod") || a.GetKind().GroupKind() == api.Kind("PersistentVolumeClaim")
}

// SupportsLimit always returns true.
func (d *DefaultLimitRangerActions) SupportsLimit(limitRange *corev1.LimitRange) bool {
	return true
}

// PersistentVolumeClaimValidateLimitFunc enforces storage limits for PVCs.
// Users request storage via pvc.Spec.Resources.Requests.  Min/Max is enforced by an admin with LimitRange.
// Claims will not be modified with default values because storage is a required part of pvc.Spec.
// All storage enforced values *only* apply to pvc.Spec.Resources.Requests.
// On update, oldPVC is the stored claim, and a request it already holds
// is not checked again.  oldPVC is nil on create.
func PersistentVolumeClaimValidateLimitFunc(limitRange *corev1.LimitRange, pvc, oldPVC *api.PersistentVolumeClaim) error {
	var errs []error
	requests := pvc.Spec.Resources.Requests
	var oldRequests api.ResourceList
	if oldPVC != nil {
		oldRequests = oldPVC.Spec.Resources.Requests
	}
	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		limitType := limit.Type
		if limitType == corev1.LimitTypePersistentVolumeClaim {
			for k, v := range limit.Min {
				// A stored request is valid as stored, so only a changed one is checked.
				if unchangedRequest(api.ResourceName(k), requests, oldRequests) {
					continue
				}
				// normal usage of minConstraint. pvc.Spec.Resources.Limits is not recognized as user input
				if err := minConstraint(string(limitType), string(k), v, requests, api.ResourceList{}); err != nil {
					errs = append(errs, err)
				}
			}
			for k, v := range limit.Max {
				if unchangedRequest(api.ResourceName(k), requests, oldRequests) {
					continue
				}
				// We want to enforce the max of the LimitRange against what
				// the user requested.
				if err := maxRequestConstraint(string(limitType), string(k), v, requests); err != nil {
					errs = append(errs, err)
				}
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

// unchangedRequest reports whether resourceName is requested in both lists
// with the same value.
func unchangedRequest(resourceName api.ResourceName, requests, oldRequests api.ResourceList) bool {
	req, reqExists := requests[resourceName]
	old, oldExists := oldRequests[resourceName]
	return reqExists && oldExists && req.Cmp(old) == 0
}

// PodMutateLimitFunc sets resource requirements enumerated by the pod against
// the specified LimitRange.  The pod may be modified to apply default resource
// requirements if not specified, and enumerated on the LimitRange
func PodMutateLimitFunc(limitRange *corev1.LimitRange, pod *api.Pod) error {
	defaultResources := defaultContainerResourceRequirements(limitRange)
	mergePodResourceRequirements(pod, &defaultResources)
	return nil
}

// PodValidateLimitFunc enforces resource requirements enumerated by the pod against
// the specified LimitRange.
//
// On update, oldPod is the stored pod, and a request or limit that a container
// or the pod as a whole already holds is not checked again: a stored value is
// valid by definition, so a resize that leaves it unchanged is not rejected by a
// constraint it no longer satisfies. Values the update changes are checked as on
// create. oldPod is nil on create.
func PodValidateLimitFunc(limitRange *corev1.LimitRange, pod, oldPod *api.Pod) error {
	var errs []error

	var oldContainers, oldInitContainers []api.Container
	if oldPod != nil {
		oldContainers = oldPod.Spec.Containers
		oldInitContainers = oldPod.Spec.InitContainers
	}

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		limitType := limit.Type
		// enforce container limits
		if limitType == corev1.LimitTypeContainer {
			errs = append(errs, validateContainerLimits(limit, pod.Spec.Containers, oldContainers)...)
			errs = append(errs, validateContainerLimits(limit, pod.Spec.InitContainers, oldInitContainers)...)
		}

		// enforce pod limits
		if limitType == corev1.LimitTypePod {
			opts := podResourcesOptions{
				PodLevelResourcesEnabled: feature.DefaultFeatureGate.Enabled(features.PodLevelResources),
			}
			var oldPodRequests, oldPodLimits api.ResourceList
			if oldPod != nil {
				oldPodRequests = podRequests(oldPod, opts)
				oldPodLimits = podLimits(oldPod, opts)
			}
			podRequests := podRequests(pod, opts)
			podLimits := podLimits(pod, opts)
			unchanged := func(resourceName corev1.ResourceName) bool {
				return oldPod != nil && resourceUnchanged(api.ResourceName(resourceName), podRequests, oldPodRequests, podLimits, oldPodLimits)
			}
			for k, v := range limit.Min {
				if unchanged(k) {
					continue
				}
				if err := minConstraint(string(limitType), string(k), v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
			for k, v := range limit.Max {
				if unchanged(k) {
					continue
				}
				if err := maxConstraint(string(limitType), string(k), v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
			for k, v := range limit.MaxLimitRequestRatio {
				if unchanged(k) {
					continue
				}
				if err := limitRequestRatioConstraint(string(limitType), string(k), v, podRequests, podLimits); err != nil {
					errs = append(errs, err)
				}
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

// validateContainerLimits enforces the min, max and limit-to-request ratio of a
// container-type LimitRangeItem on every container in containers. A value is
// skipped when the container of the same name in oldContainers holds the same
// request and limit for that resource; oldContainers is nil on create.
func validateContainerLimits(limit corev1.LimitRangeItem, containers, oldContainers []api.Container) []error {
	var errs []error
	limitType := string(limit.Type)
	for i := range containers {
		container := &containers[i]
		oldContainer := containerByName(oldContainers, container.Name)
		unchanged := func(resourceName corev1.ResourceName) bool {
			return oldContainer != nil && resourceUnchanged(api.ResourceName(resourceName), container.Resources.Requests, oldContainer.Resources.Requests, container.Resources.Limits, oldContainer.Resources.Limits)
		}
		for k, v := range limit.Min {
			if unchanged(k) {
				continue
			}
			if err := minConstraint(limitType, string(k), v, container.Resources.Requests, container.Resources.Limits); err != nil {
				errs = append(errs, err)
			}
		}
		for k, v := range limit.Max {
			if unchanged(k) {
				continue
			}
			if err := maxConstraint(limitType, string(k), v, container.Resources.Requests, container.Resources.Limits); err != nil {
				errs = append(errs, err)
			}
		}
		for k, v := range limit.MaxLimitRequestRatio {
			if unchanged(k) {
				continue
			}
			if err := limitRequestRatioConstraint(limitType, string(k), v, container.Resources.Requests, container.Resources.Limits); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errs
}

// containerByName returns the container called name, or nil if there is none.
func containerByName(containers []api.Container, name string) *api.Container {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	return nil
}

// resourceUnchanged reports whether both the request and the limit for
// resourceName are the same in the new and the old resource lists. Each of the
// min, max and ratio constraints reads both the request and the limit, so a
// value only counts as unchanged when neither moved.
func resourceUnchanged(resourceName api.ResourceName, requests, oldRequests, limits, oldLimits api.ResourceList) bool {
	return sameQuantity(resourceName, requests, oldRequests) && sameQuantity(resourceName, limits, oldLimits)
}

// sameQuantity reports whether resourceName is either absent from both lists or
// present in both with an equal quantity.
func sameQuantity(resourceName api.ResourceName, a, b api.ResourceList) bool {
	qa, inA := a[resourceName]
	qb, inB := b[resourceName]
	if inA != inB {
		return false
	}
	return !inA || qa.Cmp(qb) == 0
}

type podResourcesOptions struct {
	// PodLevelResourcesEnabled indicates that the PodLevelResources feature gate is
	// enabled.
	PodLevelResourcesEnabled bool
}

// podRequests is a simplified version of pkg/api/v1/resource/PodRequests that operates against the core version of
// pod. Any changes to that calculation should be reflected here.
// NOTE: We do not want to check status resources here, only the spec. This is equivalent to setting
// UseStatusResources=false in the common helper.
// TODO: Maybe we can consider doing a partial conversion of the pod to a v1
// type and then using the pkg/api/v1/resource/PodRequests.
// TODO(ndixita): PodRequests method exists in
// staging/src/k8s.io/component-helpers/resource/helpers.go. Refactor the code to
// avoid duplicating podRequests method.
func podRequests(pod *api.Pod, opts podResourcesOptions) api.ResourceList {
	reqs := api.ResourceList{}

	for _, container := range pod.Spec.Containers {
		containerReqs := container.Resources.Requests
		addResourceList(reqs, containerReqs)
	}

	restartableInitCotnainerReqs := api.ResourceList{}
	initContainerReqs := api.ResourceList{}
	// init containers define the minimum of any resource
	for _, container := range pod.Spec.InitContainers {
		containerReqs := container.Resources.Requests

		if container.RestartPolicy != nil && *container.RestartPolicy == api.ContainerRestartPolicyAlways {
			// and add them to the resulting cumulative container requests
			addResourceList(reqs, containerReqs)

			// track our cumulative restartable init container resources
			addResourceList(restartableInitCotnainerReqs, containerReqs)
			containerReqs = restartableInitCotnainerReqs
		} else {
			tmp := api.ResourceList{}
			addResourceList(tmp, containerReqs)
			addResourceList(tmp, restartableInitCotnainerReqs)
			containerReqs = tmp
		}

		maxResourceList(initContainerReqs, containerReqs)
	}

	maxResourceList(reqs, initContainerReqs)

	// If PodLevelResources feature is enabled and resources are set at pod-level,
	// override aggregated container requests of resources supported by pod-level
	// resources with quantities specified at pod-level.
	if opts.PodLevelResourcesEnabled && pod.Spec.Resources != nil {
		for resourceName, quantity := range pod.Spec.Resources.Requests {
			if isSupportedPodLevelResource(resourceName) {
				// override with pod-level resource requests
				reqs[resourceName] = quantity
			}
		}
	}

	return reqs
}

// podLimits is a simplified version of pkg/api/v1/resource/PodLimits that operates against the core version of
// pod. Any changes to that calculation should be reflected here.
// NOTE: We do not want to check status resources here, only the spec. This is equivalent to setting
// UseStatusResources=false in the common helper.
// TODO: Maybe we can consider doing a partial conversion of the pod to a v1
// type and then using the pkg/api/v1/resource/PodLimits.
// TODO(ndixita): PodLimits method exists in
// staging/src/k8s.io/component-helpers/resource/helpers.go. Refactor the code to
// avoid duplicating podLimits method.
func podLimits(pod *api.Pod, opts podResourcesOptions) api.ResourceList {
	limits := api.ResourceList{}

	for _, container := range pod.Spec.Containers {
		addResourceList(limits, container.Resources.Limits)
	}

	restartableInitContainerLimits := api.ResourceList{}
	initContainerLimits := api.ResourceList{}
	// init containers define the minimum of any resource
	for _, container := range pod.Spec.InitContainers {
		containerLimits := container.Resources.Limits
		// Is the init container marked as a sidecar?
		if container.RestartPolicy != nil && *container.RestartPolicy == api.ContainerRestartPolicyAlways {
			addResourceList(limits, containerLimits)

			// track our cumulative restartable init container resources
			addResourceList(restartableInitContainerLimits, containerLimits)
			containerLimits = restartableInitContainerLimits
		} else {
			tmp := api.ResourceList{}
			addResourceList(tmp, containerLimits)
			addResourceList(tmp, restartableInitContainerLimits)
			containerLimits = tmp
		}
		maxResourceList(initContainerLimits, containerLimits)
	}

	maxResourceList(limits, initContainerLimits)

	// If PodLevelResources feature is enabled and resources are set at pod-level,
	// override aggregated container limits of resources supported by pod-level
	// resources with quantities specified at pod-level.
	if opts.PodLevelResourcesEnabled && pod.Spec.Resources != nil {
		for resourceName, quantity := range pod.Spec.Resources.Limits {
			if isSupportedPodLevelResource(resourceName) {
				// override with pod-level resource limits
				limits[resourceName] = quantity
			}
		}
	}

	return limits
}

var supportedPodLevelResources = sets.New(api.ResourceCPU, api.ResourceMemory)

// isSupportedPodLevelResources checks if a given resource is supported by pod-level
// resource management through the PodLevelResources feature. Returns true if
// the resource is supported.
// isSupportedPodLevelResource method exists in
// staging/src/k8s.io/component-helpers/resource/helpers.go.
// isSupportedPodLevelResource is added here to avoid conversion of v1.
// Pod to api.Pod.
// TODO(ndixita): Find alternatives to avoid duplicating the code.
func isSupportedPodLevelResource(name api.ResourceName) bool {
	return supportedPodLevelResources.Has(name)
}

// addResourceList adds the resources in newList to list.
func addResourceList(list, newList api.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; !ok {
			list[name] = quantity.DeepCopy()
		} else {
			value.Add(quantity)
			list[name] = value
		}
	}
}

// maxResourceList sets list to the greater of list/newList for every resource in newList
func maxResourceList(list, newList api.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; !ok || quantity.Cmp(value) > 0 {
			list[name] = quantity.DeepCopy()
		}
	}
}
