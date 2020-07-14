package sccadmission

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"
	rbacregistry "k8s.io/kubernetes/pkg/registry/rbac"
	"k8s.io/kubernetes/pkg/serviceaccount"

	securityv1 "github.com/openshift/api/security/v1"
	securityv1informer "github.com/openshift/client-go/security/informers/externalversions/security/v1"
	securityv1listers "github.com/openshift/client-go/security/listers/security/v1"

	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccmatching"
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
	client     kubernetes.Interface
	sccLister  securityv1listers.SecurityContextConstraintsLister
	sccSynced  cache.InformerSynced
	authorizer authorizer.Authorizer
}

var (
	_ = initializer.WantsAuthorizer(&constraint{})
	_ = initializer.WantsExternalKubeClientSet(&constraint{})
	_ = WantsSecurityInformer(&constraint{})
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
// 1.  Find SCCs for the user.
// 2.  Find SCCs for the SA.  If there is an error retrieving SA SCCs it is not fatal.
// 3.  Remove duplicates between the user/SA SCCs.
// 4.  Create the providers, includes setting pre-allocated values if necessary.
// 5.  Try to generate and validate an SCC with providers.  If we find one then admit the pod
//     with the validated SCC.  If we don't find any reject the pod and give all errors from the
//     failed attempts.
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

	// TODO(liggitt): allow spec mutation during initializing updates?
	specMutationAllowed := a.GetOperation() == admission.Create

	allowedPod, sccName, validationErrs, err := c.computeSecurityContext(ctx, a, pod, specMutationAllowed, "")
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	if allowedPod != nil {
		*pod = *allowedPod
		// annotate and accept the pod
		klog.V(4).Infof("pod %s (generate: %s) validated against provider %s", pod.Name, pod.GenerateName, sccName)
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = map[string]string{}
		}
		pod.ObjectMeta.Annotations[securityv1.ValidatedSCCAnnotation] = sccName
		return nil
	}

	// we didn't validate against any security context constraint provider, reject the pod and give the errors for each attempt
	klog.V(4).Infof("unable to validate pod %s (generate: %s) against any security context constraint: %v", pod.Name, pod.GenerateName, validationErrs)
	return admission.NewForbidden(a, fmt.Errorf("unable to validate against any security context constraint: %v", validationErrs))
}

func (c *constraint) Validate(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	if ignore, err := shouldIgnore(a); err != nil {
		return err
	} else if ignore {
		return nil
	}
	pod := a.GetObject().(*coreapi.Pod)

	// compute the context. Mutation is not allowed. ValidatedSCCAnnotation is used as a hint to gain same speed-up.
	allowedPod, _, validationErrs, err := c.computeSecurityContext(ctx, a, pod, false, pod.ObjectMeta.Annotations[securityv1.ValidatedSCCAnnotation])
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

func (c *constraint) computeSecurityContext(ctx context.Context, a admission.Attributes, pod *coreapi.Pod, specMutationAllowed bool, validatedSCCHint string) (*coreapi.Pod, string, field.ErrorList, error) {
	// get all constraints that are usable by the user
	klog.V(4).Infof("getting security context constraints for pod %s (generate: %s) in namespace %s with user info %v", pod.Name, pod.GenerateName, a.GetNamespace(), a.GetUserInfo())

	err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		return c.sccSynced(), nil
	})
	if err != nil {
		return nil, "", nil, admission.NewForbidden(a, fmt.Errorf("securitycontextconstraints.security.openshift.io cache is not synchronized"))
	}

	constraints, err := sccmatching.NewDefaultSCCMatcher(c.sccLister, nil).FindApplicableSCCs(ctx, a.GetNamespace())
	if err != nil {
		return nil, "", nil, admission.NewForbidden(a, err)
	}
	if len(constraints) == 0 {
		sccs, err := c.sccLister.List(labels.Everything())
		if err != nil {
			return nil, "", nil, admission.NewForbidden(a, err)
		}
		if len(sccs) == 0 {
			return nil, "", nil, admission.NewForbidden(a, fmt.Errorf("no SecurityContextConstraints found in cluster"))
		}
		return nil, "", nil, admission.NewForbidden(a, fmt.Errorf("no SecurityContextConstraints found in namespace %s", a.GetNamespace()))
	}

	// If mutation is not allowed and validatedSCCHint is provided, check the validated policy first.
	// Keep the other the same for everything else
	sort.SliceStable(constraints, func(i, j int) bool {
		if !specMutationAllowed {
			if constraints[i].Name == validatedSCCHint {
				return true
			}
			if constraints[j].Name == validatedSCCHint {
				return false
			}
		}
		return i < j
	})

	providers, errs := sccmatching.CreateProvidersFromConstraints(a.GetNamespace(), constraints, c.client)
	logProviders(pod, providers, errs)

	if len(providers) == 0 {
		return nil, "", nil, admission.NewForbidden(a, fmt.Errorf("no SecurityContextConstraintsProvider available to validate pod request"))
	}

	// all containers in a single pod must validate under a single provider or we will reject the request
	var (
		allowedPod       *coreapi.Pod
		allowingProvider sccmatching.SecurityContextConstraintsProvider
		validationErrs   field.ErrorList
		saUserInfo       user.Info
	)

	userInfo := a.GetUserInfo()
	if len(pod.Spec.ServiceAccountName) > 0 {
		saUserInfo = serviceaccount.UserInfo(a.GetNamespace(), pod.Spec.ServiceAccountName, "")
	}

loop:
	for _, provider := range providers {
		// Get the SCC attributes required to decide whether the SCC applies for current user/SA
		sccName := provider.GetSCCName()
		sccUsers := provider.GetSCCUsers()
		sccGroups := provider.GetSCCGroups()

		// continue to the next provider if the current SCC one does not apply to either the user or the serviceaccount
		if !sccmatching.ConstraintAppliesTo(ctx, sccName, sccUsers, sccGroups, userInfo, a.GetNamespace(), c.authorizer) &&
			!(saUserInfo != nil && sccmatching.ConstraintAppliesTo(ctx, sccName, sccUsers, sccGroups, saUserInfo, a.GetNamespace(), c.authorizer)) {
			continue
		}

		podCopy := pod.DeepCopy()
		if errs := sccmatching.AssignSecurityContext(provider, podCopy, field.NewPath(fmt.Sprintf("provider %s: ", sccName))); len(errs) > 0 {
			validationErrs = append(validationErrs, errs...)
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
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s with mutation", pod.Name, pod.GenerateName, sccName)
			break loop
		case apiequality.Semantic.DeepEqual(pod, podCopy):
			// if we don't allow mutation, only use the validated pod if it didn't require any spec changes
			allowedPod = podCopy
			allowingProvider = provider
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s without mutation", pod.Name, pod.GenerateName, sccName)
			break loop
		default:
			klog.V(5).Infof("pod %s (generate: %s) validated against provider %s, but required mutation, skipping", pod.Name, pod.GenerateName, sccName)
		}
	}

	if allowedPod == nil || allowingProvider == nil {
		return nil, "", validationErrs, nil
	}
	return allowedPod, allowingProvider.GetSCCName(), validationErrs, nil
}

func shouldIgnore(a admission.Attributes) (bool, error) {
	if a.GetResource().GroupResource() != coreapi.Resource("pods") {
		return true, nil
	}
	if len(a.GetSubresource()) != 0 {
		return true, nil
	}

	_, ok := a.GetObject().(*coreapi.Pod)
	// if we can't convert then fail closed since we've already checked that this is supposed to be a pod object.
	// this shouldn't normally happen during admission but could happen if an integrator passes a versioned
	// pod object rather than an internal object.
	if !ok {
		return false, admission.NewForbidden(a, fmt.Errorf("object was marked as kind pod but was unable to be converted: %v", a.GetObject()))
	}

	// if this is an update, see if we are only updating the ownerRef.  Garbage collection does this
	// and we should allow it in general, since you had the power to update and the power to delete.
	// The worst that happens is that you delete something, but you aren't controlling the privileged object itself
	if a.GetOperation() == admission.Update && rbacregistry.IsOnlyMutatingGCFields(a.GetObject(), a.GetOldObject(), kapihelper.Semantic) {
		return true, nil
	}

	return false, nil
}

// SetSecurityInformers implements WantsSecurityInformer interface for constraint.
func (c *constraint) SetSecurityInformers(informers securityv1informer.SecurityContextConstraintsInformer) {
	c.sccLister = informers.Lister()
	c.sccSynced = informers.Informer().HasSynced
}

func (c *constraint) SetExternalKubeClientSet(client kubernetes.Interface) {
	c.client = client
}

func (c *constraint) SetAuthorizer(authorizer authorizer.Authorizer) {
	c.authorizer = authorizer
}

// ValidateInitialization implements InitializationValidator interface for constraint.
func (c *constraint) ValidateInitialization() error {
	if c.sccLister == nil {
		return fmt.Errorf("%s requires an sccLister", PluginName)
	}
	if c.sccSynced == nil {
		return fmt.Errorf("%s requires an sccSynced", PluginName)
	}
	if c.client == nil {
		return fmt.Errorf("%s requires a client", PluginName)
	}
	if c.authorizer == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
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
		klog.V(4).Infof("provider creation error: %v", err)
	}
}
