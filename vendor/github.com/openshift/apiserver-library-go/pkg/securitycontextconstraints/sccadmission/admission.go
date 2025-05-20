package sccadmission

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	kutilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"
	rbacregistry "k8s.io/kubernetes/pkg/registry/rbac"

	securityv1 "github.com/openshift/api/security/v1"
	securityv1informer "github.com/openshift/client-go/security/informers/externalversions/security/v1"
	securityv1listers "github.com/openshift/client-go/security/listers/security/v1"

	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccmatching"
	sccsort "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util/sort"
)

const PluginName = "security.openshift.io/SecurityContextConstraint"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			return NewConstraint(), nil
		})
}

type constraint struct {
	*admission.Handler
	sccLister       securityv1listers.SecurityContextConstraintsLister
	namespaceLister corev1listers.NamespaceLister
	listersSynced   []cache.InformerSynced
	authorizer      authorizer.Authorizer
}

var (
	_ = initializer.WantsAuthorizer(&constraint{})
	_ = WantsSecurityInformer(&constraint{})
	_ = initializer.WantsExternalKubeInformerFactory(&constraint{})
	_ = admission.ValidationInterface(&constraint{})
	_ = admission.MutationInterface(&constraint{})
)

// NewConstraint creates a new SCC constraint admission plugin.
func NewConstraint() *constraint {
	return &constraint{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Admit determines if the pod should be admitted based on the requested security context
// and the available SCCs.
//
//  1. Find SCCs for the user.
//  2. Find SCCs for the SA.  If there is an error retrieving SA SCCs it is not fatal.
//  3. Remove duplicates between the user/SA SCCs.
//  4. Create the providers, includes setting pre-allocated values if necessary.
//  5. Try to generate and validate an SCC with providers.  If we find one then admit the pod
//     with the validated SCC.  If we don't find any reject the pod and give all errors from the
//     failed attempts.
//
// On updates, the BeforeUpdate of the pod strategy only zeroes out the status.  That means that
// any change that claims the pod is no longer privileged will be removed.  That should hold until
// we get a true old/new set of objects in.
func (c *constraint) Admit(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	if ignore, err := shouldIgnore(a); err != nil {
		return err
	} else if ignore {
		return nil
	}
	pod := a.GetObject().(*coreapi.Pod)

	// deny changes to required SCC annotation during updates
	if a.GetOperation() == admission.Update {
		oldPod := a.GetOldObject().(*coreapi.Pod)

		if pod.ObjectMeta.Annotations[securityv1.RequiredSCCAnnotation] != oldPod.ObjectMeta.Annotations[securityv1.RequiredSCCAnnotation] {
			return admission.NewForbidden(a, fmt.Errorf("invalid change of required security context constraint annotation: %v", securityv1.RequiredSCCAnnotation))
		}
	}

	// TODO(liggitt): allow spec mutation during initializing updates?
	specMutationAllowed := a.GetOperation() == admission.Create
	ephemeralContainersMutationAllowed := specMutationAllowed || (a.GetOperation() == admission.Update && a.GetSubresource() == "ephemeralcontainers")

	allowedPod, sccAnnotations, validationErrs, err := c.computeSecurityContext(ctx, a, pod, specMutationAllowed, ephemeralContainersMutationAllowed, pod.ObjectMeta.Annotations[securityv1.RequiredSCCAnnotation], "")
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	if allowedPod != nil {
		*pod = *allowedPod
		// annotate and accept the pod
		klog.V(4).Infof(
			"pod %s (generate: %s) validated against provider %s",
			pod.Name, pod.GenerateName, sccAnnotations[securityv1.ValidatedSCCAnnotation],
		)

		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = map[string]string{}
		}

		for key, value := range sccAnnotations {
			pod.ObjectMeta.Annotations[key] = value
		}

		return nil
	}

	// we didn't validate against any security context constraint provider, reject the pod and give the errors for each attempt
	klog.V(4).Infof("unable to validate pod %s (generate: %s) against any security context constraint: %v", pod.Name, pod.GenerateName, validationErrs.ToAggregate())
	return admission.NewForbidden(a, fmt.Errorf("unable to validate against any security context constraint: %v", validationErrs.ToAggregate()))
}

func (c *constraint) Validate(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	if ignore, err := shouldIgnore(a); err != nil {
		return err
	} else if ignore {
		return nil
	}
	pod := a.GetObject().(*coreapi.Pod)

	// this one is required
	requiredSCCName := pod.ObjectMeta.Annotations[securityv1.RequiredSCCAnnotation]
	// this one is non-binding
	validatedSCCNameHint := pod.ObjectMeta.Annotations[securityv1.ValidatedSCCAnnotation]
	if len(requiredSCCName) > 0 && requiredSCCName != validatedSCCNameHint {
		return admission.NewForbidden(a, fmt.Errorf("required scc/%v does not match the suggested scc/%v", requiredSCCName, validatedSCCNameHint))
	}

	// compute the context. Mutation is not allowed. ValidatedSCCAnnotation is used as a hint to gain same speed-up.
	allowedPod, _, validationErrs, err := c.computeSecurityContext(ctx, a, pod, false, false, requiredSCCName, validatedSCCNameHint)
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if allowedPod != nil && apiequality.Semantic.DeepEqual(pod, allowedPod) {
		return nil
	}

	// we didn't validate against any provider, reject the pod and give the errors for each attempt
	klog.V(4).Infof("unable to validate pod %s (generate: %s) in namespace %s against any security context constraint: %v", pod.Name, pod.GenerateName, a.GetNamespace(), validationErrs)
	return admission.NewForbidden(a, fmt.Errorf("unable to validate against any security context constraint: %v", validationErrs))
}

// these are the SCCs created by the cluster-kube-apiserver-operator.
// see the list in https://github.com/openshift/cluster-kube-apiserver-operator/blob/3b0218cf9778cbcf2650ad5aa4e01d7b40a2d05e/bindata/bootkube/scc-manifests/0000_20_kube-apiserver-operator_00_scc-restricted.yaml
// if these are not present, the lister isn't really finished listing.
var standardSCCNames = sets.NewString(
	"anyuid",
	"hostaccess",
	"hostmount-anyuid",
	"hostnetwork",
	"hostnetwork-v2",
	"nonroot",
	"nonroot-v2",
	"privileged",
	"restricted",
	"restricted-v2",
)

func requireStandardSCCs(sccs []*securityv1.SecurityContextConstraints, err error) error {
	if err != nil {
		return err
	}

	allCurrentSCCNames := sets.NewString()
	for _, curr := range sccs {
		allCurrentSCCNames.Insert(curr.Name)
	}

	missingSCCs := standardSCCNames.Difference(allCurrentSCCNames)
	if len(missingSCCs) == 0 {
		return nil
	}

	return fmt.Errorf("securitycontextconstraints.security.openshift.io cache is missing %v", strings.Join(missingSCCs.List(), ", "))
}

func (c *constraint) computeSecurityContext(
	ctx context.Context,
	a admission.Attributes,
	pod *coreapi.Pod,
	specMutationAllowed, ephemeralContainersMutationAllowed bool,
	requiredSCCName, validatedSCCHint string,
) (*coreapi.Pod, map[string]string, field.ErrorList, error) {
	// get all constraints that are usable by the user
	klog.V(4).Infof("getting security context constraints for pod %s (generate: %s) in namespace %s with user info %v", pod.Name, pod.GenerateName, a.GetNamespace(), a.GetUserInfo())

	if err := c.waitForReadyState(ctx); err != nil {
		return nil, nil, nil, admission.NewForbidden(a, err)
	}

	constraints, err := c.listSortedSCCs(requiredSCCName, validatedSCCHint, specMutationAllowed)
	if err != nil {
		return nil, nil, nil, admission.NewForbidden(a, err)
	}

	providers, errs := sccmatching.CreateProvidersFromConstraints(ctx, a.GetNamespace(), constraints, c.namespaceLister)
	logProviders(pod, providers, errs)
	if len(errs) > 0 {
		return nil, nil, nil, kutilerrors.NewAggregate(errs)
	}

	if len(providers) == 0 {
		return nil, nil, nil, admission.NewForbidden(a, fmt.Errorf("no SecurityContextConstraintsProvider available to validate pod request"))
	}

	// all containers in a single pod must validate under a single provider or we will reject the request
	var (
		allowedPod       *coreapi.Pod
		allowingProvider sccmatching.SecurityContextConstraintsProvider
		allowedForType   string
		validationErrs   field.ErrorList
	)

	sccChecker := newSCCAuthorizerChecker(c.authorizer, a, pod.Spec.ServiceAccountName)

	appliesToPod := func(provider sccmatching.SecurityContextConstraintsProvider, pod *coreapi.Pod) (podCopy *coreapi.Pod, errs field.ErrorList) {
		podCopy = pod.DeepCopy()
		if errs := sccmatching.AssignSecurityContext(provider, podCopy, field.NewPath(fmt.Sprintf("provider %s: ", provider.GetSCCName()))); len(errs) > 0 {
			return nil, errs
		}
		return podCopy, nil
	}

	var (
		restrictedSCCProvider   sccmatching.SecurityContextConstraintsProvider
		restrictedV2SCCProvider sccmatching.SecurityContextConstraintsProvider
		provider                sccmatching.SecurityContextConstraintsProvider
		denied                  = []string{}
		failures                = map[string]string{}
		i                       int
	)
loop:
	for i, provider = range providers {
		switch provider.GetSCCName() {
		case "restricted":
			restrictedSCCProvider = providers[i]
		case "restricted-v2":
			restrictedV2SCCProvider = providers[i]
		}

		currentType := sccChecker.allowedForType(ctx, provider)
		if currentType == allowedForNone {
			denied = append(denied, provider.GetSCCName())
			// this will cause every security context constraint attempted, in order, to the failure
			validationErrs = append(validationErrs,
				field.Forbidden(
					field.NewPath(fmt.Sprintf("provider %q", provider.GetSCCName())),
					"not usable by user or serviceaccount",
				),
			)

			continue
		}

		podCopy, errs := appliesToPod(provider, pod)
		if len(errs) > 0 {
			validationErrs = append(validationErrs, errs...)
			failures[provider.GetSCCName()] = errs.ToAggregate().Error()
			continue
		}

		// the entire pod validated
		switch {
		case specMutationAllowed:
			// if mutation is allowed, use the first found SCC that allows the pod.
			// This behavior is different from Kubernetes which tries to search a non-mutating provider
			// even on creating. We prefer most restrictive SCC in this case even if it mutates a pod.
			allowedPod = podCopy
			allowingProvider = provider
			allowedForType = currentType
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s with mutation", pod.Name, pod.GenerateName, provider.GetSCCName())
			break loop
		case ephemeralContainersMutationAllowed:
			podCopyCopy := podCopy.DeepCopy()
			// check if, possibly, only the ephemeral containers were mutated
			podCopyCopy.Spec.EphemeralContainers = pod.Spec.EphemeralContainers
			if apiequality.Semantic.DeepEqual(pod, podCopyCopy) {
				allowedPod = podCopy
				allowingProvider = provider
				allowedForType = currentType
				klog.V(5).Infof("pod %s (generate: %s) validated against provider %s with ephemeralContainers mutation", pod.Name, pod.GenerateName, provider.GetSCCName())
				break loop
			}
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s, but required pod mutation outside ephemeralContainers, skipping", pod.Name, pod.GenerateName, provider.GetSCCName())
			failures[provider.GetSCCName()] = "failures final validation after mutating admission"
		case apiequality.Semantic.DeepEqual(pod, podCopy):
			// if we don't allow mutation, only use the validated pod if it didn't require any spec changes
			allowedPod = podCopy
			allowingProvider = provider
			allowedForType = currentType
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s without mutation", pod.Name, pod.GenerateName, provider.GetSCCName())
			break loop
		default:
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s, but required mutation, skipping", pod.Name, pod.GenerateName, provider.GetSCCName())
			failures[provider.GetSCCName()] = "failures final validation after mutating admission"
		}
	}

	// if we have restricted-v2, and we're not allowed (this means restricted-v2 did not match) and the user cannot use restricted-v1
	// then we should check to see if restricted-v1 would allow the pod.  If so, prepend a specific failure message.
	userCannotUseForRestricted := sets.NewString(denied...).Has("restricted")
	hasRestrictedV2 := restrictedV2SCCProvider != nil
	isAllowed := allowingProvider != nil
	if hasRestrictedV2 && !isAllowed && userCannotUseForRestricted {
		// restrictedSCCProvider is never nil because the loop above only adds "restricted" to the denied list if it found "restricted"
		_, restrictedErrs := appliesToPod(restrictedSCCProvider, pod)
		if len(restrictedErrs) == 0 {
			// this means that restricted-v1 works, so we should indicate that the pod would have been admitted otherwise
			validationErrs = append(validationErrs,
				field.Forbidden(
					field.NewPath("no access to scc/restricted"),
					"the pod fails to validate against the `restricted-v2` security context constraint, "+
						"but would validate successfully against the `restricted` security context constraint",
				),
			)
		}
	}

	// add audit annotations
	if specMutationAllowed {
		// find next provider that was not chosen
		var nextNotChosenProvider sccmatching.SecurityContextConstraintsProvider
		for _, provider := range providers[i+1:] {
			if sccChecker.allowedForType(ctx, provider) == allowedForNone {
				continue
			}
			if _, errs := appliesToPod(provider, pod); len(errs) == 0 {
				nextNotChosenProvider = provider
				break
			}
		}

		a.AddAnnotation("securityserviceconstraints.admission.openshift.io/denied", strings.Join(denied, ","))
		for sccName, reason := range failures {
			a.AddAnnotation(fmt.Sprintf("securitycontextconstraints.admission.openshift.io/too-restrictive-%s", sccName), reason)
		}

		if allowingProvider != nil && nextNotChosenProvider != nil {
			chosen := allowingProvider.GetSCC()
			next := nextNotChosenProvider.GetSCC()
			if chosen != nil && next != nil {
				_, reason := sccsort.ByPriority([]*securityv1.SecurityContextConstraints{chosen, next}).LessWithReason(0, 1)
				if len(reason) == 0 {
					reason = "unknown"
				} else {
					reason = fmt.Sprintf("%q is most restrictive, not denied, and chosen over %q because %q %s", chosen.Name, next.Name, chosen.Name, reason)
				}
				a.AddAnnotation("securitycontextconstraints.admission.openshift.io/reason", reason)
			}
		} else if allowingProvider != nil {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/reason", fmt.Sprintf("%q is the only one not too restrictive and not denied", allowingProvider.GetSCCName()))
		} else if len(failures) == 0 {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/reason", "all denied")
		} else {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/reason", "all too restrictive or denied")
		}
	} else if len(validatedSCCHint) != 0 && (allowingProvider == nil || allowingProvider.GetSCCName() != validatedSCCHint) {
		if reason, ok := failures[validatedSCCHint]; ok {
			a.AddAnnotation(fmt.Sprintf("securitycontextconstraints.admission.openshift.io/too-restrictive-%s", validatedSCCHint), reason)
		} else {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/denied-validation", fmt.Sprintf("originally chosen %q got denied in final validation after mutating admission", validatedSCCHint))
		}

		if allowingProvider != nil {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/reason-validation", fmt.Sprintf("originally chosen %q did not pass final validation after mutating admission, but %q did instead", validatedSCCHint, allowingProvider.GetSCCName()))
		} else {
			a.AddAnnotation("securitycontextconstraints.admission.openshift.io/denied-validation", fmt.Sprintf("originally chosen %q got denied in final validation after mutating admission, and no other matched", validatedSCCHint))
		}
	}

	if allowedPod == nil || allowingProvider == nil {
		return nil, nil, validationErrs, nil
	}

	if !specMutationAllowed {
		// the finally chosen SCC. Note that we are not allowed to set an annotation multiple times, hence only for !specMutationAllowed
		a.AddAnnotation("securitycontextconstraints.admission.openshift.io/chosen", allowingProvider.GetSCCName())
	}

	podAnnotations := map[string]string{
		securityv1.ValidatedSCCAnnotation:                  allowingProvider.GetSCCName(),
		"security.openshift.io/validated-scc-subject-type": allowedForType,
	}

	return allowedPod, podAnnotations, validationErrs, nil
}

var ignoredSubresources = sets.NewString(
	"exec",
	"attach",
	"binding",
	"eviction",
	"log",
	"portforward",
	"proxy",
	"status",
)

var ignoredAnnotations = sets.NewString(
	"k8s.ovn.org/pod-networks",
)

func shouldIgnore(a admission.Attributes) (bool, error) {
	if a.GetResource().GroupResource() != coreapi.Resource("pods") {
		return true, nil
	}

	if subresource := a.GetSubresource(); len(subresource) != 0 && ignoredSubresources.Has(subresource) {
		return true, nil
	}

	pod, ok := a.GetObject().(*coreapi.Pod)
	// if we can't convert then fail closed since we've already checked that this is supposed to be a pod object.
	// this shouldn't normally happen during admission but could happen if an integrator passes a versioned
	// pod object rather than an internal object.
	if !ok {
		return false, admission.NewForbidden(a, fmt.Errorf("object was marked as kind pod but was unable to be converted: %v", a.GetObject()))
	}
	// ignore all Windows pods
	// TODO: This can be expanded to other OS'es later if needed
	if pod.Spec.OS != nil && pod.Spec.OS.Name == coreapi.Windows {
		return true, nil
	}

	if a.GetOperation() == admission.Update {
		oldPod, ok := a.GetOldObject().(*coreapi.Pod)
		if !ok {
			return false, admission.NewForbidden(a, fmt.Errorf("object was marked as kind pod but was unable to be converted: %v", a.GetOldObject()))
		}

		// never ignore any spec changes
		if !kapihelper.Semantic.DeepEqual(pod.Spec, oldPod.Spec) {
			return false, nil
		}

		// see if we are only doing meta changes that should be ignored during admission
		// for example, the OVN controller adds informative networking annotations that shouldn't cause the pod to go through admission again
		if shouldIgnoreMetaChanges(pod, oldPod) {
			return true, nil
		}
	}

	return false, nil
}

func shouldIgnoreMetaChanges(newPod, oldPod *coreapi.Pod) bool {
	// check if we're adding or changing only annotations from the ignore list
	for key, newVal := range newPod.ObjectMeta.Annotations {
		if oldVal, ok := oldPod.ObjectMeta.Annotations[key]; ok && newVal == oldVal {
			continue
		}

		if !ignoredAnnotations.Has(key) {
			return false
		}
	}

	// check if we're removing only annotations from the ignore list
	for key := range oldPod.ObjectMeta.Annotations {
		if _, ok := newPod.ObjectMeta.Annotations[key]; ok {
			continue
		}

		if !ignoredAnnotations.Has(key) {
			return false
		}
	}

	newPodCopy := newPod.DeepCopyObject()
	newPodCopyMeta, err := meta.Accessor(newPodCopy)
	if err != nil {
		return false
	}
	newPodCopyMeta.SetAnnotations(oldPod.ObjectMeta.Annotations)

	// see if we are only updating the ownerRef. Garbage collection does this
	// and we should allow it in general, since you had the power to update and the power to delete.
	// The worst that happens is that you delete something, but you aren't controlling the privileged object itself
	res := rbacregistry.IsOnlyMutatingGCFields(newPodCopy, oldPod, kapihelper.Semantic)

	return res
}

// SetSecurityInformers implements WantsSecurityInformer interface for constraint.
func (c *constraint) SetSecurityInformers(informers securityv1informer.SecurityContextConstraintsInformer) {
	c.sccLister = informers.Lister()
	c.listersSynced = append(c.listersSynced, informers.Informer().HasSynced)
}

func (c *constraint) SetExternalKubeInformerFactory(informers informers.SharedInformerFactory) {
	c.namespaceLister = informers.Core().V1().Namespaces().Lister()
	c.listersSynced = append(c.listersSynced, informers.Core().V1().Namespaces().Informer().HasSynced)
}

func (c *constraint) SetAuthorizer(authorizer authorizer.Authorizer) {
	c.authorizer = authorizer
}

// ValidateInitialization implements InitializationValidator interface for constraint.
func (c *constraint) ValidateInitialization() error {
	if c.sccLister == nil {
		return fmt.Errorf("%s requires an sccLister", PluginName)
	}
	if c.listersSynced == nil {
		return fmt.Errorf("%s requires an sccSynced", PluginName)
	}
	if c.namespaceLister == nil {
		return fmt.Errorf("%s requires a namespaceLister", PluginName)
	}
	if c.authorizer == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	return nil
}

func (c *constraint) listSortedSCCs(
	requiredSCCName, validatedSCCHint string,
	specMutationAllowed bool,
) ([]*securityv1.SecurityContextConstraints, error) {
	var err error
	var constraints []*securityv1.SecurityContextConstraints

	if len(requiredSCCName) > 0 {
		requiredSCC, err := c.sccLister.Get(requiredSCCName)
		if err != nil {
			return nil, fmt.Errorf("failed to retrieve the required SCC %q: %w", requiredSCCName, err)
		}
		constraints = []*securityv1.SecurityContextConstraints{requiredSCC}
	} else {
		constraints, err = c.sccLister.List(labels.Everything())
		if err != nil {
			return nil, err
		}
	}

	if len(constraints) == 0 {
		return nil, fmt.Errorf("no SecurityContextConstraints found in cluster")
	}

	sort.Sort(sccsort.ByPriority(constraints))

	if specMutationAllowed {
		return constraints, nil
	}

	// If mutation is not allowed and validatedSCCHint is provided, check the validated policy first.
	// Keep the order the same for everything else
	sort.SliceStable(constraints, func(i, j int) bool {
		// disregard the ephemeral containers here, the rest of the pod should still
		// not get mutated and so we are primarily interested in the SCC that matched previously
		if constraints[i].Name == validatedSCCHint {
			return true
		}
		if constraints[j].Name == validatedSCCHint {
			return false
		}
		return i < j
	})

	return constraints, nil
}

// waitForReadyState ensures the admission controller has a complete and
// consistent view of SCCs before making admission decisions. It first waits for
// the internal cache to sync, then verifies all standard SCCs are present.
func (c *constraint) waitForReadyState(ctx context.Context) error {
	const (
		interval  = 1 * time.Second
		timeout   = 10 * time.Second
		immediate = true
	)

	err := wait.PollUntilContextTimeout(ctx, interval, timeout, immediate, func(ctx context.Context) (bool, error) {
		for _, syncFunc := range c.listersSynced {
			if !syncFunc() {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("securitycontextconstraints.security.openshift.io cache is not synchronized")
	}

	// wait a few seconds until the synchronized list returns all the required SCCs created by the kas-o.
	// If this doesn't happen, then indicate which ones are missing.  This seems odd, but our CI system suggests that this happens occasionally.
	// If the SCCs were all deleted, then no pod will pass SCC admission until the SCCs are recreated, but the kas-o (which recreates them)
	// bypasses SCC admission, so this does not create a cycle.
	var requiredSCCErr error
	err = wait.PollUntilContextTimeout(ctx, interval, timeout, immediate, func(context.Context) (bool, error) {
		if requiredSCCErr = requireStandardSCCs(c.sccLister.List(labels.Everything())); requiredSCCErr != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		if requiredSCCErr != nil {
			return requiredSCCErr
		}
		return fmt.Errorf("securitycontextconstraints.security.openshift.io required check failed oddly")
	}

	return nil
}

// logProviders logs what providers were found for the pod as well as any errors that were encountered
// while creating providers.
func logProviders(pod *coreapi.Pod, providers []sccmatching.SecurityContextConstraintsProvider, providerCreationErrs []error) {
	names := make([]string, len(providers))
	for i, p := range providers {
		names[i] = p.GetSCCName()
	}
	klog.V(4).Infof("validating pod %s (generate: %s) against providers %s", pod.Name, pod.GenerateName, strings.Join(names, ","))

	for _, err := range providerCreationErrs {
		klog.V(2).Infof("provider creation error: %v", err)
	}
}
