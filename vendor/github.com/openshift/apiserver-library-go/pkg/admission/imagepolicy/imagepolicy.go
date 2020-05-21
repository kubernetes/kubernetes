package imagepolicy

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"strings"
	"time"

	lru "github.com/hashicorp/golang-lru"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/klog"
	kapi "k8s.io/kubernetes/pkg/apis/core"

	imagev1 "github.com/openshift/api/image/v1"
	imagepolicy "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1"
	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/validation"
	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators"
	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/rules"
	imagev1client "github.com/openshift/client-go/image/clientset/versioned"
	imagev1typedclient "github.com/openshift/client-go/image/clientset/versioned/typed/image/v1"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	"github.com/openshift/library-go/pkg/image/imageutil"
	"github.com/openshift/library-go/pkg/image/reference"
)

func Register(plugins *admission.Plugins) {
	plugins.Register(imagepolicy.PluginName,
		func(input io.Reader) (admission.Interface, error) {
			config := &imagepolicy.ImagePolicyConfig{}
			if input != nil {
				configContent, err := ioutil.ReadAll(input)
				if err != nil {
					return nil, err
				}
				scheme := runtime.NewScheme()
				utilruntime.Must(imagepolicy.Install(scheme))
				codecs := serializer.NewCodecFactory(scheme)
				err = runtime.DecodeInto(codecs.UniversalDecoder(imagepolicy.GroupVersion), configContent, config)
				if err != nil {
					return nil, err
				}
			}

			imagepolicy.SetDefaults_ImagePolicyConfig(config)
			if errs := validation.Validate(config); len(errs) > 0 {
				return nil, errs.ToAggregate()
			}
			klog.V(5).Infof("%s admission controller loaded with config: %#v", imagepolicy.PluginName, config)
			return NewImagePolicyPlugin(config)
		})
}

type ImagePolicyPlugin struct {
	*admission.Handler
	config *imagepolicy.ImagePolicyConfig
	Client imagev1client.Interface

	accepter rules.Accepter

	integratedRegistryMatcher integratedRegistryMatcher

	NsLister corev1listers.NamespaceLister
	resolver imageResolver

	imageMutators imagereferencemutators.ImageMutators
}

var _ = initializer.WantsExternalKubeInformerFactory(&ImagePolicyPlugin{})
var _ = admissionrestconfig.WantsRESTClientConfig(&ImagePolicyPlugin{})
var _ = WantsInternalImageRegistry(&ImagePolicyPlugin{})
var _ = WantsImageMutators(&ImagePolicyPlugin{})
var _ = admission.ValidationInterface(&ImagePolicyPlugin{})
var _ = admission.MutationInterface(&ImagePolicyPlugin{})

type integratedRegistryMatcher struct {
	rules.RegistryMatcher
}

// imageResolver abstracts identifying an image for a particular reference.
type imageResolver interface {
	ResolveObjectReference(ref *kapi.ObjectReference, defaultNamespace string, forceResolveLocalNames bool) (*rules.ImagePolicyAttributes, error)
}

// imageResolutionPolicy determines whether an image should be resolved
type imageResolutionPolicy interface {
	// RequestsResolution returns true if you should attempt to resolve image pull specs
	RequestsResolution(metav1.GroupResource) bool
	// FailOnResolutionFailure returns true if you should fail when resolution fails
	FailOnResolutionFailure(metav1.GroupResource) bool
	// RewriteImagePullSpec returns true if you should rewrite image pull specs when resolution succeeds
	RewriteImagePullSpec(attr *rules.ImagePolicyAttributes, isUpdate bool, gr metav1.GroupResource) bool
}

// ImagePolicyPlugin returns an admission controller for pods that controls what images are allowed to run on the
// cluster.
func NewImagePolicyPlugin(parsed *imagepolicy.ImagePolicyConfig) (*ImagePolicyPlugin, error) {
	m := integratedRegistryMatcher{
		RegistryMatcher: rules.NewRegistryMatcher(nil),
	}
	accepter, err := rules.NewExecutionRulesAccepter(parsed.ExecutionRules, m)
	if err != nil {
		return nil, err
	}

	return &ImagePolicyPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		config:  parsed,

		accepter: accepter,

		integratedRegistryMatcher: m,
	}, nil
}

func (a *ImagePolicyPlugin) SetInternalImageRegistry(internalImageRegistryName string) {
	a.integratedRegistryMatcher.RegistryMatcher = rules.RegistryNameMatcher(internalImageRegistryName)
}

func (a *ImagePolicyPlugin) SetImageMutators(imageMutators imagereferencemutators.ImageMutators) {
	a.imageMutators = imageMutators
}

func (a *ImagePolicyPlugin) SetRESTClientConfig(restClientConfig rest.Config) {
	var err error
	a.Client, err = imagev1client.NewForConfig(&restClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
}

func (a *ImagePolicyPlugin) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	a.NsLister = kubeInformers.Core().V1().Namespaces().Lister()
}

// Validate ensures that all required interfaces have been provided, or returns an error.
func (a *ImagePolicyPlugin) ValidateInitialization() error {
	if a.Client == nil {
		return fmt.Errorf("%s needs an Openshift client", imagepolicy.PluginName)
	}
	if a.NsLister == nil {
		return fmt.Errorf("%s needs a namespace lister", imagepolicy.PluginName)
	}
	if a.imageMutators == nil {
		return fmt.Errorf("%s needs an image mutators", imagepolicy.PluginName)
	}
	imageResolver, err := newImageResolutionCache(a.Client.ImageV1(), a.integratedRegistryMatcher)
	if err != nil {
		return fmt.Errorf("unable to create image policy controller: %v", err)
	}
	a.resolver = imageResolver
	return nil
}

// Admit attempts to apply the image policy to the incoming resource.
func (a *ImagePolicyPlugin) Admit(ctx context.Context, attr admission.Attributes, _ admission.ObjectInterfaces) error {
	return a.admit(ctx, attr, true)
}

// Validate attempts to apply the image policy to the incoming resource.
func (a *ImagePolicyPlugin) Validate(ctx context.Context, attr admission.Attributes, _ admission.ObjectInterfaces) error {
	return a.admit(ctx, attr, false)
}

func (a *ImagePolicyPlugin) admit(ctx context.Context, attr admission.Attributes, mutationAllowed bool) error {
	switch attr.GetOperation() {
	case admission.Create, admission.Update:
		if len(attr.GetSubresource()) > 0 {
			return nil
		}
		// only create and update are tested, and only on core resources
		// TODO: scan all resources
		// TODO: Create a general equivalence map for admission - operation X on subresource Y is equivalent to reduced operation
	default:
		return nil
	}

	policy := resolutionConfig{a.config}

	schemagr := attr.GetResource().GroupResource()
	apigr := metav1.GroupResource{Resource: schemagr.Resource, Group: schemagr.Group}

	if !a.accepter.Covers(apigr) && !policy.Covers(apigr) {
		return nil
	}

	klog.V(5).Infof("running image policy admission for %s:%s/%s", attr.GetKind(), attr.GetNamespace(), attr.GetName())
	m, err := a.imageMutators.GetImageReferenceMutator(attr.GetObject(), attr.GetOldObject())
	if err != nil {
		return apierrs.NewForbidden(schemagr, attr.GetName(), fmt.Errorf("unable to apply image policy against objects of type %T: %v", attr.GetObject(), err))
	}

	if !mutationAllowed {
		m = &mutationPreventer{m}
	}

	annotations, _ := a.imageMutators.GetAnnotationAccessor(attr.GetObject())

	// load exclusion rules from the namespace cache
	var excluded sets.String
	if ns := attr.GetNamespace(); len(ns) > 0 {
		if ns, err := a.NsLister.Get(ns); err == nil {
			if value := ns.Annotations[imagepolicy.IgnorePolicyRulesAnnotation]; len(value) > 0 {
				excluded = sets.NewString(strings.Split(value, ",")...)
			}
		}
	}

	if err := accept(a.accepter, policy, a.resolver, m, annotations, attr, excluded); err != nil {
		return err
	}

	return nil
}

type mutationPreventer struct {
	m imagereferencemutators.ImageReferenceMutator
}

func (m *mutationPreventer) Mutate(fn imagereferencemutators.ImageReferenceMutateFunc) field.ErrorList {
	return m.m.Mutate(func(ref *kapi.ObjectReference) error {
		original := ref.DeepCopy()
		if err := fn(ref); err != nil {
			return fmt.Errorf("error in image policy validation: %v", err)
		}
		if !reflect.DeepEqual(ref, original) {
			klog.V(2).Infof("disallowed mutation in image policy validation: %s", diff.ObjectGoPrintSideBySide(original, ref))
			return fmt.Errorf("this image is prohibited by policy (changed after admission)")
		}
		return nil
	})
}

type imageResolutionCache struct {
	imageClient imagev1typedclient.ImageV1Interface
	integrated  rules.RegistryMatcher
	expiration  time.Duration

	cache *lru.Cache
}

type imageCacheEntry struct {
	expires time.Time
	image   *imagev1.Image
}

// newImageResolutionCache creates a new resolver that caches frequently loaded images for one minute.
func newImageResolutionCache(imageClient imagev1typedclient.ImageV1Interface, integratedRegistry rules.RegistryMatcher) (*imageResolutionCache, error) {
	imageCache, err := lru.New(128)
	if err != nil {
		return nil, err
	}
	return &imageResolutionCache{
		imageClient: imageClient,
		integrated:  integratedRegistry,
		cache:       imageCache,
		expiration:  time.Minute,
	}, nil
}

var now = time.Now

// ResolveObjectReference converts a reference into an image API or returns an error. If the kind is not recognized
// this method will return an error to prevent references that may be images from being ignored.
func (c *imageResolutionCache) ResolveObjectReference(ref *kapi.ObjectReference, defaultNamespace string, forceResolveLocalNames bool) (*rules.ImagePolicyAttributes, error) {
	switch ref.Kind {
	case "ImageStreamTag":
		ns := ref.Namespace
		if len(ns) == 0 {
			ns = defaultNamespace
		}
		name, tag, ok := imageutil.SplitImageStreamTag(ref.Name)
		if !ok {
			return &rules.ImagePolicyAttributes{IntegratedRegistry: true}, fmt.Errorf("references of kind ImageStreamTag must be of the form NAME:TAG")
		}
		return c.resolveImageStreamTag(ns, name, tag, false, false)

	case "ImageStreamImage":
		ns := ref.Namespace
		if len(ns) == 0 {
			ns = defaultNamespace
		}
		name, id, ok := imageutil.SplitImageStreamImage(ref.Name)
		if !ok {
			return &rules.ImagePolicyAttributes{IntegratedRegistry: true}, fmt.Errorf("references of kind ImageStreamImage must be of the form NAME@DIGEST")
		}
		return c.resolveImageStreamImage(ns, name, id)

	case "DockerImage":
		ref, err := reference.Parse(ref.Name)
		if err != nil {
			return nil, err
		}
		return c.resolveImageReference(ref, defaultNamespace, forceResolveLocalNames)

	default:
		return nil, fmt.Errorf("image policy does not allow image references of kind %q", ref.Kind)
	}
}

// Resolve converts an image reference into a resolved image or returns an error. Only images located in the internal
// registry or those with a digest can be resolved - all other scenarios will return an error.
func (c *imageResolutionCache) resolveImageReference(ref reference.DockerImageReference, defaultNamespace string, forceResolveLocalNames bool) (*rules.ImagePolicyAttributes, error) {
	// images by ID can be checked for policy
	if len(ref.ID) > 0 {
		now := now()
		if value, ok := c.cache.Get(ref.ID); ok {
			cached := value.(imageCacheEntry)
			if now.Before(cached.expires) {
				return &rules.ImagePolicyAttributes{Name: ref, Image: cached.image}, nil
			}
		}
		image, err := c.imageClient.Images().Get(context.TODO(), ref.ID, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		c.cache.Add(ref.ID, imageCacheEntry{expires: now.Add(c.expiration), image: image})
		return &rules.ImagePolicyAttributes{Name: ref, Image: image, IntegratedRegistry: c.integrated.Matches(ref.Registry)}, nil
	}

	// an image spec that points to the internal registry is by definition also an imagestreamtag reference,
	// so attempt to resolve it as such.
	fullReference := c.integrated.Matches(ref.Registry)
	// if we've been explicitly told to treat this image spec as an imagestreamtag reference, or if it is a single
	// segment value, attempt to resolve the value as an imagestream tag that will ultimately resolve to an image.
	partialReference := forceResolveLocalNames || (len(ref.Registry) == 0 && len(ref.Namespace) == 0 && len(ref.Name) > 0)

	// if we can't treat it as an imagestreamtag reference, and since we don't have an imageid (checked earlier),
	// we aren't going to be able to resolve this value to an image.
	if !fullReference && !partialReference {
		return nil, fmt.Errorf("(%s) could not be resolved to an exact image reference", ref.Exact())
	}

	tag := ref.Tag
	if len(tag) == 0 {
		tag = imagev1.DefaultImageTag
	}
	if len(ref.Namespace) == 0 || forceResolveLocalNames {
		ref.Namespace = defaultNamespace
	}

	return c.resolveImageStreamTag(ref.Namespace, ref.Name, tag, partialReference, forceResolveLocalNames)
}

// resolveImageStreamTag loads an image stream tag and creates a fully qualified image stream image reference,
// or returns an error.
func (c *imageResolutionCache) resolveImageStreamTag(namespace, name, tag string, partial, forceResolveLocalNames bool) (*rules.ImagePolicyAttributes, error) {
	attrs := &rules.ImagePolicyAttributes{IntegratedRegistry: true}
	resolved, err := c.imageClient.ImageStreamTags(namespace).Get(context.TODO(), imageutil.JoinImageStreamTag(name, tag), metav1.GetOptions{})
	if err != nil {
		if partial {
			attrs.IntegratedRegistry = false
		}
		// if a stream exists, resolves names, and a registry is installed, change the reference to be a pointer
		// to the internal registry. This prevents the lookup from going to the original location, which is consistent
		// with the intent of resolving local names.
		if isImageStreamTagNotFound(err) {
			if stream, err := c.imageClient.ImageStreams(namespace).Get(context.TODO(), name, metav1.GetOptions{}); err == nil && (forceResolveLocalNames || stream.Spec.LookupPolicy.Local) && len(stream.Status.DockerImageRepository) > 0 {
				if ref, err := reference.Parse(stream.Status.DockerImageRepository); err == nil {
					klog.V(4).Infof("%s/%s:%s points to a local name resolving stream, but the tag does not exist", namespace, name, tag)
					ref.Tag = tag
					attrs.Name = ref
					attrs.LocalRewrite = true
					return attrs, nil
				}
			}
		}
		return attrs, err
	}
	if partial {
		if !forceResolveLocalNames && !resolved.LookupPolicy.Local {
			attrs.IntegratedRegistry = false
			return attrs, fmt.Errorf("ImageStreamTag does not allow local references and the resource did not request image stream resolution")
		}
		attrs.LocalRewrite = true
	}
	ref, err := reference.Parse(resolved.Image.DockerImageReference)
	if err != nil {
		return attrs, fmt.Errorf("image reference %s could not be parsed: %v", resolved.Image.DockerImageReference, err)
	}
	ref.Tag = ""
	ref.ID = resolved.Image.Name

	now := now()
	c.cache.Add(resolved.Image.Name, imageCacheEntry{expires: now.Add(c.expiration), image: &resolved.Image})

	attrs.Name = ref
	attrs.Image = &resolved.Image
	return attrs, nil
}

// resolveImageStreamImage loads an image stream image if it exists, or returns an error.
func (c *imageResolutionCache) resolveImageStreamImage(namespace, name, id string) (*rules.ImagePolicyAttributes, error) {
	attrs := &rules.ImagePolicyAttributes{IntegratedRegistry: true}
	resolved, err := c.imageClient.ImageStreamImages(namespace).Get(context.TODO(), imageutil.JoinImageStreamImage(name, id), metav1.GetOptions{})
	if err != nil {
		return attrs, err
	}
	ref, err := reference.Parse(resolved.Image.DockerImageReference)
	if err != nil {
		return attrs, fmt.Errorf("ImageStreamTag could not be resolved: %v", err)
	}
	now := now()
	c.cache.Add(resolved.Image.Name, imageCacheEntry{expires: now.Add(c.expiration), image: &resolved.Image})

	attrs.Name = ref
	attrs.Image = &resolved.Image
	return attrs, nil
}

// isImageStreamTagNotFound returns true iff the tag is missing but the image stream
// exists.
func isImageStreamTagNotFound(err error) bool {
	if err == nil || !apierrs.IsNotFound(err) {
		return false
	}
	status, ok := err.(apierrs.APIStatus)
	if !ok {
		return false
	}
	details := status.Status().Details
	if details == nil {
		return false
	}
	return details.Kind == "imagestreamtags" && details.Group == "image.openshift.io"
}

// resolutionConfig translates an ImagePolicyConfig into imageResolutionPolicy
type resolutionConfig struct {
	config *imagepolicy.ImagePolicyConfig
}

// Covers returns true if the resolver specifically should touch this resource.
func (config resolutionConfig) Covers(gr metav1.GroupResource) bool {
	for _, rule := range config.config.ResolutionRules {
		if resolutionRuleCoversResource(rule.TargetResource, gr) {
			return true
		}
	}
	return false
}

// RequestsResolution is true if the policy demands it or if any rule covers it.
func (config resolutionConfig) RequestsResolution(gr metav1.GroupResource) bool {
	if RequestsResolution(config.config.ResolveImages) {
		return true
	}
	for _, rule := range config.config.ResolutionRules {
		if resolutionRuleCoversResource(rule.TargetResource, gr) {
			return true
		}
	}
	return false
}

// FailOnResolutionFailure does not depend on the nested rules.
func (config resolutionConfig) FailOnResolutionFailure(gr metav1.GroupResource) bool {
	return FailOnResolutionFailure(config.config.ResolveImages)
}

var skipImageRewriteOnUpdate = map[metav1.GroupResource]struct{}{
	// Job template specs are immutable, they cannot be updated.
	{Group: "batch", Resource: "jobs"}: {},
	// Build specs are immutable, they cannot be updated.
	{Group: "build.openshift.io", Resource: "builds"}: {},
	// TODO: remove when statefulsets allow spec.template updates in 3.7
	{Group: "apps", Resource: "statefulsets"}: {},
}

// RewriteImagePullSpec applies to implicit rewrite attributes and local resources as well as if the policy requires it.
// If a local name check is requested and a rule matches true is returned. The policy default resolution is only respected
// if a resource isn't covered by a rule - if pods have a rule with DoNotAttempt and the global policy is RequiredRewrite,
// no pods will be rewritten.
func (config resolutionConfig) RewriteImagePullSpec(attr *rules.ImagePolicyAttributes, isUpdate bool, gr metav1.GroupResource) bool {
	if isUpdate {
		if _, ok := skipImageRewriteOnUpdate[gr]; ok {
			return false
		}
	}
	hasMatchingRule := false
	for _, rule := range config.config.ResolutionRules {
		if !resolutionRuleCoversResource(rule.TargetResource, gr) {
			continue
		}
		if rule.LocalNames && attr.LocalRewrite {
			return true
		}
		if RewriteImagePullSpec(rule.Policy) {
			return true
		}
		hasMatchingRule = true
	}
	if hasMatchingRule {
		return false
	}
	return RewriteImagePullSpec(config.config.ResolveImages)
}

// resolutionRuleCoversResource implements wildcard checking on Resource names
func resolutionRuleCoversResource(rule metav1.GroupResource, gr metav1.GroupResource) bool {
	return rule.Group == gr.Group && (rule.Resource == gr.Resource || rule.Resource == "*")
}
