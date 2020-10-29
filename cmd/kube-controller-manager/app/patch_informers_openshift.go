package app

import (
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	appclient "github.com/openshift/client-go/apps/clientset/versioned"
	appinformer "github.com/openshift/client-go/apps/informers/externalversions"
	authorizationclient "github.com/openshift/client-go/authorization/clientset/versioned"
	authorizationinformer "github.com/openshift/client-go/authorization/informers/externalversions"
	buildclient "github.com/openshift/client-go/build/clientset/versioned"
	buildinformer "github.com/openshift/client-go/build/informers/externalversions"
	imageclient "github.com/openshift/client-go/image/clientset/versioned"
	imageinformer "github.com/openshift/client-go/image/informers/externalversions"
	networkclient "github.com/openshift/client-go/network/clientset/versioned"
	networkinformer "github.com/openshift/client-go/network/informers/externalversions"
	oauthclient "github.com/openshift/client-go/oauth/clientset/versioned"
	oauthinformer "github.com/openshift/client-go/oauth/informers/externalversions"
	quotaclient "github.com/openshift/client-go/quota/clientset/versioned"
	quotainformer "github.com/openshift/client-go/quota/informers/externalversions"
	routeclient "github.com/openshift/client-go/route/clientset/versioned"
	routeinformer "github.com/openshift/client-go/route/informers/externalversions"
	securityclient "github.com/openshift/client-go/security/clientset/versioned"
	securityinformer "github.com/openshift/client-go/security/informers/externalversions"
	templateclient "github.com/openshift/client-go/template/clientset/versioned"
	templateinformer "github.com/openshift/client-go/template/informers/externalversions"
	userclient "github.com/openshift/client-go/user/clientset/versioned"
	userinformer "github.com/openshift/client-go/user/informers/externalversions"
)

type externalKubeInformersWithExtraGenerics struct {
	informers.SharedInformerFactory
	genericResourceInformer GenericResourceInformer
}

func (i externalKubeInformersWithExtraGenerics) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	return i.genericResourceInformer.ForResource(resource)
}

func (i externalKubeInformersWithExtraGenerics) Start(stopCh <-chan struct{}) {
	i.SharedInformerFactory.Start(stopCh)
	i.genericResourceInformer.Start(stopCh)
}

type GenericResourceInformer interface {
	ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error)
	Start(stopCh <-chan struct{})
}

// genericResourceInformerFunc will handle a cast to a matching type
type genericResourceInformerFunc func(resource schema.GroupVersionResource) (informers.GenericInformer, error)

func (fn genericResourceInformerFunc) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	return fn(resource)
}

// this is a temporary condition until we rewrite enough of generation to auto-conform to the required interface and no longer need the internal version shim
func (fn genericResourceInformerFunc) Start(stopCh <-chan struct{}) {}

type genericInformers struct {
	// this is a temporary condition until we rewrite enough of generation to auto-conform to the required interface and no longer need the internal version shim
	startFn func(stopCh <-chan struct{})
	generic []GenericResourceInformer
}

func newGenericInformers(startFn func(stopCh <-chan struct{}), informers ...GenericResourceInformer) genericInformers {
	return genericInformers{
		startFn: startFn,
		generic: informers,
	}
}

func (i genericInformers) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	var firstErr error
	for _, generic := range i.generic {
		informer, err := generic.ForResource(resource)
		if err == nil {
			return informer, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	klog.V(4).Infof("Couldn't find informer for %v", resource)
	return nil, firstErr
}

func (i genericInformers) Start(stopCh <-chan struct{}) {
	i.startFn(stopCh)
	for _, generic := range i.generic {
		generic.Start(stopCh)
	}
}

// informers is a convenient way for us to keep track of the informers, but
// is intentionally private.  We don't want to leak it out further than this package.
// Everything else should say what it wants.
type combinedInformers struct {
	externalKubeInformers  informers.SharedInformerFactory
	appInformers           appinformer.SharedInformerFactory
	authorizationInformers authorizationinformer.SharedInformerFactory
	buildInformers         buildinformer.SharedInformerFactory
	imageInformers         imageinformer.SharedInformerFactory
	networkInformers       networkinformer.SharedInformerFactory
	oauthInformers         oauthinformer.SharedInformerFactory
	quotaInformers         quotainformer.SharedInformerFactory
	routeInformers         routeinformer.SharedInformerFactory
	securityInformers      securityinformer.SharedInformerFactory
	templateInformers      templateinformer.SharedInformerFactory
	userInformers          userinformer.SharedInformerFactory
}

func newInformerFactory(clientConfig *rest.Config) (informers.SharedInformerFactory, error) {
	kubeClient, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	appClient, err := appclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	authorizationClient, err := authorizationclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	buildClient, err := buildclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	imageClient, err := imageclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	networkClient, err := networkclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	oauthClient, err := oauthclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	quotaClient, err := quotaclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	routerClient, err := routeclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	securityClient, err := securityclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	templateClient, err := templateclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	userClient, err := userclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	// TODO find a single place to create and start informers.  During the 1.7 rebase this will come more naturally in a config object,
	// before then we should try to eliminate our direct to storage access.  It's making us do weird things.
	const defaultInformerResyncPeriod = 10 * time.Minute

	trim := func(obj interface{}) (interface{}, error) {
		if accessor, err := meta.Accessor(obj); err == nil {
			accessor.SetManagedFields(nil)
		}
		return obj, nil
	}

	ci := &combinedInformers{
		externalKubeInformers:  informers.NewSharedInformerFactoryWithOptions(kubeClient, defaultInformerResyncPeriod, informers.WithTransform(trim)),
		appInformers:           appinformer.NewSharedInformerFactoryWithOptions(appClient, defaultInformerResyncPeriod, appinformer.WithTransform(trim)),
		authorizationInformers: authorizationinformer.NewSharedInformerFactoryWithOptions(authorizationClient, defaultInformerResyncPeriod, authorizationinformer.WithTransform(trim)),
		buildInformers:         buildinformer.NewSharedInformerFactoryWithOptions(buildClient, defaultInformerResyncPeriod, buildinformer.WithTransform(trim)),
		imageInformers:         imageinformer.NewSharedInformerFactoryWithOptions(imageClient, defaultInformerResyncPeriod, imageinformer.WithTransform(trim)),
		networkInformers:       networkinformer.NewSharedInformerFactoryWithOptions(networkClient, defaultInformerResyncPeriod, networkinformer.WithTransform(trim)),
		oauthInformers:         oauthinformer.NewSharedInformerFactoryWithOptions(oauthClient, defaultInformerResyncPeriod, oauthinformer.WithTransform(trim)),
		quotaInformers:         quotainformer.NewSharedInformerFactoryWithOptions(quotaClient, defaultInformerResyncPeriod, quotainformer.WithTransform(trim)),
		routeInformers:         routeinformer.NewSharedInformerFactoryWithOptions(routerClient, defaultInformerResyncPeriod, routeinformer.WithTransform(trim)),
		securityInformers:      securityinformer.NewSharedInformerFactoryWithOptions(securityClient, defaultInformerResyncPeriod, securityinformer.WithTransform(trim)),
		templateInformers:      templateinformer.NewSharedInformerFactoryWithOptions(templateClient, defaultInformerResyncPeriod, templateinformer.WithTransform(trim)),
		userInformers:          userinformer.NewSharedInformerFactoryWithOptions(userClient, defaultInformerResyncPeriod, userinformer.WithTransform(trim)),
	}

	return externalKubeInformersWithExtraGenerics{
		SharedInformerFactory:   ci.GetExternalKubeInformers(),
		genericResourceInformer: ci.ToGenericInformer(),
	}, nil
}

func (i *combinedInformers) GetExternalKubeInformers() informers.SharedInformerFactory {
	return i.externalKubeInformers
}
func (i *combinedInformers) GetAppInformers() appinformer.SharedInformerFactory {
	return i.appInformers
}
func (i *combinedInformers) GetAuthorizationInformers() authorizationinformer.SharedInformerFactory {
	return i.authorizationInformers
}
func (i *combinedInformers) GetBuildInformers() buildinformer.SharedInformerFactory {
	return i.buildInformers
}
func (i *combinedInformers) GetImageInformers() imageinformer.SharedInformerFactory {
	return i.imageInformers
}
func (i *combinedInformers) GetNetworkInformers() networkinformer.SharedInformerFactory {
	return i.networkInformers
}
func (i *combinedInformers) GetOauthInformers() oauthinformer.SharedInformerFactory {
	return i.oauthInformers
}
func (i *combinedInformers) GetQuotaInformers() quotainformer.SharedInformerFactory {
	return i.quotaInformers
}
func (i *combinedInformers) GetRouteInformers() routeinformer.SharedInformerFactory {
	return i.routeInformers
}
func (i *combinedInformers) GetSecurityInformers() securityinformer.SharedInformerFactory {
	return i.securityInformers
}
func (i *combinedInformers) GetTemplateInformers() templateinformer.SharedInformerFactory {
	return i.templateInformers
}
func (i *combinedInformers) GetUserInformers() userinformer.SharedInformerFactory {
	return i.userInformers
}

// Start initializes all requested informers.
func (i *combinedInformers) Start(stopCh <-chan struct{}) {
	i.externalKubeInformers.Start(stopCh)
	i.appInformers.Start(stopCh)
	i.authorizationInformers.Start(stopCh)
	i.buildInformers.Start(stopCh)
	i.imageInformers.Start(stopCh)
	i.networkInformers.Start(stopCh)
	i.oauthInformers.Start(stopCh)
	i.quotaInformers.Start(stopCh)
	i.routeInformers.Start(stopCh)
	i.securityInformers.Start(stopCh)
	i.templateInformers.Start(stopCh)
	i.userInformers.Start(stopCh)
}

func (i *combinedInformers) ToGenericInformer() GenericResourceInformer {
	return newGenericInformers(
		i.Start,
		i.GetExternalKubeInformers(),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetAppInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetAuthorizationInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetBuildInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetImageInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetNetworkInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetOauthInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetQuotaInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetRouteInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetSecurityInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetTemplateInformers().ForResource(resource)
		}),
		genericResourceInformerFunc(func(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
			return i.GetUserInformers().ForResource(resource)
		}),
	)
}
