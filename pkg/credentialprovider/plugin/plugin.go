/*
Copyright 2020 The Kubernetes Authors.

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

package plugin

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/cryptobyte"
	"golang.org/x/sync/singleflight"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	credentialproviderapi "k8s.io/kubelet/pkg/apis/credentialprovider"
	"k8s.io/kubelet/pkg/apis/credentialprovider/install"
	credentialproviderv1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1"
	credentialproviderv1alpha1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1"
	credentialproviderv1beta1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1beta1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigv1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1beta1"
	"k8s.io/utils/clock"
)

const (
	// globalCacheKey is the key used for caching credentials that are not specific to a registry or image.
	// angle brackets are used to avoid conflicts with actual image or registry names.
	globalCacheKey     = "<global>"
	cachePurgeInterval = time.Minute * 15
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme, serializer.EnableStrict)

	apiVersions = map[string]schema.GroupVersion{
		credentialproviderv1alpha1.SchemeGroupVersion.String(): credentialproviderv1alpha1.SchemeGroupVersion,
		credentialproviderv1beta1.SchemeGroupVersion.String():  credentialproviderv1beta1.SchemeGroupVersion,
		credentialproviderv1.SchemeGroupVersion.String():       credentialproviderv1.SchemeGroupVersion,
	}
)

// GetServiceAccountFunc is a function type that returns a service account for the given namespace and name.
type GetServiceAccountFunc func(namespace, name string) (*v1.ServiceAccount, error)

// getServiceAccountTokenFunc is a function type that returns a service account token for the given namespace and name.
type getServiceAccountTokenFunc func(namespace, name string, tr *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error)

type cacheKeyParams struct {
	namespace          string
	serviceAccountName string
	serviceAccountUID  types.UID
	saAnnotations      map[string]string
	podName            string
	podUID             types.UID
	saTokenHash        string
	cacheType          kubeletconfig.ServiceAccountTokenCacheType
}

type cacheKeyBuilder struct {
	builder *cryptobyte.Builder
}

func newCacheKeyBuilder() *cacheKeyBuilder {
	return &cacheKeyBuilder{builder: cryptobyte.NewBuilder(nil)}
}

func (c *cacheKeyBuilder) addString(value string) *cacheKeyBuilder {
	c.builder.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte(value))
	})
	return c
}

func (c *cacheKeyBuilder) addAnnotations(annotations map[string]string) *cacheKeyBuilder {
	keys := sets.StringKeySet(annotations).List()
	c.builder.AddUint32(uint32(len(keys)))
	for _, k := range keys {
		c.builder.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes([]byte(k))
		})
		c.builder.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes([]byte(annotations[k]))
		})
	}
	return c
}

func (c *cacheKeyBuilder) build() (string, error) {
	keyBytes, err := c.builder.Bytes()
	if err != nil {
		return "", err
	}
	return string(keyBytes), nil
}

func init() {
	install.Install(scheme)
	kubeletconfig.AddToScheme(scheme)
	kubeletconfigv1alpha1.AddToScheme(scheme)
	kubeletconfigv1beta1.AddToScheme(scheme)
	kubeletconfigv1.AddToScheme(scheme)
}

// RegisterCredentialProviderPlugins is called from kubelet to register external credential provider
// plugins according to the CredentialProviderConfig config file.
func RegisterCredentialProviderPlugins(pluginConfigPath, pluginBinDir string,
	getServiceAccountToken getServiceAccountTokenFunc,
	getServiceAccount GetServiceAccountFunc,
) error {
	if _, err := os.Stat(pluginBinDir); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("plugin binary directory %s did not exist", pluginBinDir)
		}

		return fmt.Errorf("error inspecting binary directory %s: %w", pluginBinDir, err)
	}

	credentialProviderConfig, configHash, err := readCredentialProviderConfig(pluginConfigPath)
	if err != nil {
		return err
	}

	saTokenForCredentialProvidersFeatureEnabled := utilfeature.DefaultFeatureGate.Enabled(features.KubeletServiceAccountTokenForCredentialProviders)
	if errs := validateCredentialProviderConfig(credentialProviderConfig, saTokenForCredentialProvidersFeatureEnabled); len(errs) > 0 {
		return fmt.Errorf("failed to validate credential provider config: %v", errs.ToAggregate())
	}

	// Register metrics for credential providers
	registerMetrics()

	// Record the hash of the credential provider configuration.
	// This hash is exposed via metrics as an external API to allow monitoring of configuration changes.
	recordCredentialProviderConfigHash(configHash)

	for _, provider := range credentialProviderConfig.Providers {
		// Considering Windows binary with suffix ".exe", LookPath() helps to find the correct path.
		// LookPath() also calls os.Stat().
		pluginBin, err := exec.LookPath(filepath.Join(pluginBinDir, provider.Name))
		if err != nil {
			if errors.Is(err, os.ErrNotExist) || errors.Is(err, exec.ErrNotFound) {
				return fmt.Errorf("plugin binary executable %s did not exist", pluginBin)
			}

			return fmt.Errorf("error inspecting binary executable %s: %w", pluginBin, err)
		}

		plugin, err := newPluginProvider(pluginBinDir, provider, getServiceAccountToken, getServiceAccount)
		if err != nil {
			return fmt.Errorf("error initializing plugin provider %s: %w", provider.Name, err)
		}

		registerCredentialProviderPlugin(provider.Name, plugin)
	}

	return nil
}

// newPluginProvider returns a new pluginProvider based on the credential provider config.
func newPluginProvider(pluginBinDir string, provider kubeletconfig.CredentialProvider,
	getServiceAccountToken getServiceAccountTokenFunc,
	getServiceAccount GetServiceAccountFunc,
) (*pluginProvider, error) {
	mediaType := "application/json"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}

	gv, ok := apiVersions[provider.APIVersion]
	if !ok {
		return nil, fmt.Errorf("invalid apiVersion: %q", provider.APIVersion)
	}

	clock := clock.RealClock{}
	return &pluginProvider{
		clock:                clock,
		matchImages:          provider.MatchImages,
		cache:                cache.NewExpirationStore(cacheKeyFunc, &cacheExpirationPolicy{clock: clock}),
		defaultCacheDuration: provider.DefaultCacheDuration.Duration,
		lastCachePurge:       clock.Now(),
		plugin: &execPlugin{
			name:         provider.Name,
			apiVersion:   provider.APIVersion,
			encoder:      codecs.EncoderForVersion(info.Serializer, gv),
			pluginBinDir: pluginBinDir,
			args:         provider.Args,
			envVars:      provider.Env,
			environ:      os.Environ,
		},
		serviceAccountProvider: newServiceAccountProvider(provider, getServiceAccount, getServiceAccountToken),
	}, nil
}

// pluginProvider is the plugin-based implementation of the DockerConfigProvider interface.
type pluginProvider struct {
	clock clock.Clock

	sync.Mutex

	group singleflight.Group

	// matchImages defines the matching image URLs this plugin should operate against.
	// The plugin provider will not return any credentials for images that do not match
	// against this list of match URLs.
	matchImages []string

	// cache stores DockerConfig entries with an expiration time based on the cache duration
	// returned from the credential provider plugin.
	cache cache.Store
	// defaultCacheDuration is the default duration credentials are cached in-memory if the auth plugin
	// response did not provide a cache duration for credentials.
	defaultCacheDuration time.Duration

	// plugin is the exec implementation of the credential providing plugin.
	plugin Plugin

	// lastCachePurge is the last time cache is cleaned for expired entries.
	lastCachePurge time.Time

	// serviceAccountProvider holds the logic for handling service account tokens when needed.
	serviceAccountProvider *serviceAccountProvider
}

type serviceAccountProvider struct {
	audience                             string
	requireServiceAccount                bool
	getServiceAccountFunc                GetServiceAccountFunc
	getServiceAccountTokenFunc           getServiceAccountTokenFunc
	requiredServiceAccountAnnotationKeys []string
	optionalServiceAccountAnnotationKeys []string
	cacheType                            kubeletconfig.ServiceAccountTokenCacheType
}

func newServiceAccountProvider(
	provider kubeletconfig.CredentialProvider,
	getServiceAccount GetServiceAccountFunc,
	getServiceAccountToken getServiceAccountTokenFunc,
) *serviceAccountProvider {
	featureGateEnabled := utilfeature.DefaultFeatureGate.Enabled(features.KubeletServiceAccountTokenForCredentialProviders)
	serviceAccountTokenAudienceSet := provider.TokenAttributes != nil && len(provider.TokenAttributes.ServiceAccountTokenAudience) > 0

	if !featureGateEnabled || !serviceAccountTokenAudienceSet {
		return nil
	}

	return &serviceAccountProvider{
		audience:                             provider.TokenAttributes.ServiceAccountTokenAudience,
		requireServiceAccount:                *provider.TokenAttributes.RequireServiceAccount,
		getServiceAccountFunc:                getServiceAccount,
		getServiceAccountTokenFunc:           getServiceAccountToken,
		requiredServiceAccountAnnotationKeys: provider.TokenAttributes.RequiredServiceAccountAnnotationKeys,
		optionalServiceAccountAnnotationKeys: provider.TokenAttributes.OptionalServiceAccountAnnotationKeys,
		cacheType:                            provider.TokenAttributes.CacheType,
	}
}

type requiredAnnotationNotFoundError string

func (e requiredAnnotationNotFoundError) Error() string {
	return fmt.Sprintf("required annotation %s not found", string(e))
}

func isRequiredAnnotationNotFoundError(err error) bool {
	var requiredAnnotationNotFoundErr requiredAnnotationNotFoundError
	return errors.As(err, &requiredAnnotationNotFoundErr)
}

// getServiceAccountData returns the service account UID and required annotations for the service account.
// If the service account does not exist, an error is returned.
// saAnnotations is a map of annotation keys and values that the plugin requires to generate credentials
// that's defined in the tokenAttributes in the credential provider config.
// requiredServiceAccountAnnotationKeys are the keys that are required to be present in the service account.
// If any of the keys defined in this list are not present in the service account, kubelet will not invoke the plugin
// and will return an error.
// optionalServiceAccountAnnotationKeys are the keys that are optional to be present in the service account.
// If present, they will be added to the saAnnotations map.
func (s *serviceAccountProvider) getServiceAccountData(namespace, name string) (types.UID, map[string]string, error) {
	sa, err := s.getServiceAccountFunc(namespace, name)
	if err != nil {
		return "", nil, err
	}

	saAnnotations := make(map[string]string, len(s.requiredServiceAccountAnnotationKeys)+len(s.optionalServiceAccountAnnotationKeys))
	for _, k := range s.requiredServiceAccountAnnotationKeys {
		val, ok := sa.Annotations[k]
		if !ok {
			return "", nil, requiredAnnotationNotFoundError(k)
		}
		saAnnotations[k] = val
	}

	for _, k := range s.optionalServiceAccountAnnotationKeys {
		if val, ok := sa.Annotations[k]; ok {
			saAnnotations[k] = val
		}
	}

	return sa.UID, saAnnotations, nil
}

// getServiceAccountToken returns a service account token for the service account.
func (s *serviceAccountProvider) getServiceAccountToken(podNamespace, podName, serviceAccountName string, serviceAccountUID, podUID types.UID) (string, error) {
	tr, err := s.getServiceAccountTokenFunc(podNamespace, serviceAccountName, &authenticationv1.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{
			UID: serviceAccountUID,
		},
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{s.audience},
			// expirationSeconds is not set explicitly here. It has the same default value of "ExpirationSeconds" in the TokenRequestSpec.
			BoundObjectRef: &authenticationv1.BoundObjectReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        podUID,
			},
		},
	})

	if err != nil {
		return "", err
	}

	return tr.Status.Token, nil
}

// cacheEntry is the cache object that will be stored in cache.Store.
type cacheEntry struct {
	key         string
	credentials credentialprovider.DockerConfig
	expiresAt   time.Time
}

// cacheKeyFunc extracts AuthEntry.MatchKey as the cache key function for the plugin provider.
func cacheKeyFunc(obj interface{}) (string, error) {
	key := obj.(*cacheEntry).key
	return key, nil
}

// cacheExpirationPolicy defines implements cache.ExpirationPolicy, determining expiration based on the expiresAt timestamp.
type cacheExpirationPolicy struct {
	clock clock.Clock
}

// IsExpired returns true if the current time is after cacheEntry.expiresAt, which is determined by the
// cache duration returned from the credential provider plugin response.
func (c *cacheExpirationPolicy) IsExpired(entry *cache.TimestampedEntry) bool {
	return c.clock.Now().After(entry.Obj.(*cacheEntry).expiresAt)
}

// perPluginProvider holds the shared pluginProvider and the per-request information
// like podName, podNamespace, podUID and serviceAccountName.
// This is used to provide the per-request information to the pluginProvider.provide method, so
// that the plugin can use this information to get the pod's service account and generate bound service account tokens
// for plugins running in service account token mode.
type perPodPluginProvider struct {
	name string

	provider *pluginProvider

	podNamespace string
	podName      string
	podUID       types.UID

	serviceAccountName string
}

// provideWithCoordinates returns the DockerConfig and, if available, the ServiceAccountCoordinates
// used for credential resolution. If ServiceAccountCoordinates is nil, it means no service account
// context was used (e.g., the plugin is not operating in service account token mode or no service
// account was provided for the request).
func (p *perPodPluginProvider) provideWithCoordinates(image string) (credentialprovider.DockerConfig, *credentialprovider.ServiceAccountCoordinates) {
	return p.provider.provide(image, p.podNamespace, p.podName, p.podUID, p.serviceAccountName)
}

// provide returns a credentialprovider.DockerConfig based on the credentials returned
// from cache or the exec plugin. The returned ServiceAccountCoordinates may be nil.
// If ServiceAccountCoordinates is nil, it means no service account context was used
// (e.g., the plugin is not operating in service account token mode or no service account
// was provided for the request).
func (p *pluginProvider) provide(image, podNamespace, podName string, podUID types.UID, serviceAccountName string) (credentialprovider.DockerConfig, *credentialprovider.ServiceAccountCoordinates) {
	if !p.isImageAllowed(image) {
		return credentialprovider.DockerConfig{}, nil
	}

	var serviceAccountUID types.UID
	var serviceAccountToken string
	var saAnnotations map[string]string
	var err error
	var serviceAccountCacheKey string
	var serviceAccountTokenHash string
	var serviceAccountCoordinates *credentialprovider.ServiceAccountCoordinates

	if p.serviceAccountProvider != nil {
		if len(serviceAccountName) == 0 && p.serviceAccountProvider.requireServiceAccount {
			klog.V(5).Infof("Service account name is empty for pod %s/%s", podNamespace, podName)
			return credentialprovider.DockerConfig{}, nil
		}

		// If the service account name is empty and the plugin has indicated that invoking the plugin
		// without a service account is allowed, we will continue without generating a service account token.
		// This is useful for plugins that are running in service account token mode and are also used
		// to pull images for pods without service accounts (e.g., static pods).
		if len(serviceAccountName) > 0 {
			if serviceAccountUID, saAnnotations, err = p.serviceAccountProvider.getServiceAccountData(podNamespace, serviceAccountName); err != nil {
				if isRequiredAnnotationNotFoundError(err) {
					// The required annotation could be a mechanism for individual workloads to opt in to using service account tokens
					// for image pull. If any of the required annotation is missing, we will not invoke the plugin. We will log the error
					// at higher verbosity level as it could be noisy.
					klog.V(5).Infof("Failed to get service account data %s/%s: %v", podNamespace, serviceAccountName, err)
					return credentialprovider.DockerConfig{}, nil
				}

				klog.Errorf("Failed to get service account %s/%s: %v", podNamespace, serviceAccountName, err)
				return credentialprovider.DockerConfig{}, nil
			}

			if serviceAccountToken, err = p.serviceAccountProvider.getServiceAccountToken(podNamespace, podName, serviceAccountName, serviceAccountUID, podUID); err != nil {
				klog.Errorf("Error getting service account token %s/%s: %v", podNamespace, serviceAccountName, err)
				return credentialprovider.DockerConfig{}, nil
			}
			serviceAccountTokenHash = getHashIfNotEmpty(serviceAccountToken)

			c := cacheKeyParams{
				namespace:          podNamespace,
				serviceAccountName: serviceAccountName,
				serviceAccountUID:  serviceAccountUID,
				saAnnotations:      saAnnotations,
				podName:            podName,
				podUID:             podUID,
				saTokenHash:        serviceAccountTokenHash,
				cacheType:          p.serviceAccountProvider.cacheType,
			}

			serviceAccountCacheKey, err = generateServiceAccountCacheKey(c)
			if err != nil {
				klog.Errorf("Error generating service account cache key: %v", err)
				return credentialprovider.DockerConfig{}, nil
			}

			serviceAccountCoordinates = &credentialprovider.ServiceAccountCoordinates{
				UID:       string(serviceAccountUID),
				Namespace: podNamespace,
				Name:      serviceAccountName,
			}
		}
	}

	// Check if the credentials are cached and return them if found.
	cachedConfig, found, errCache := p.getCachedCredentials(image, serviceAccountCacheKey)
	if errCache != nil {
		klog.Errorf("Failed to get cached docker config: %v", err)
		return credentialprovider.DockerConfig{}, nil
	}

	if found {
		return cachedConfig, serviceAccountCoordinates
	}

	// ExecPlugin is wrapped in single flight to exec plugin once for concurrent same image request.
	// The caveat here is we don't know cacheKeyType yet, so if cacheKeyType is registry/global and credentials saved in cache
	// on per registry/global basis then exec will be called for all requests if requests are made concurrently.
	// foo.bar.registry
	// foo.bar.registry/image1
	// foo.bar.registry/image2
	// When the plugin is operating in the service account token mode, the singleflight key is the image plus the serviceAccountCacheKey
	// which is generated from the service account namespace, name, uid and the annotations passed to the plugin.
	singleFlightKey := image
	if p.serviceAccountProvider != nil && len(serviceAccountName) > 0 {
		// When the plugin is operating in the service account token mode, the singleflight key is the
		// image + sa annotations + sa token.
		// This does mean the singleflight key is different for each image pull request (even if the image is the same)
		// and the workload is using the same service account.
		if singleFlightKey, err = generateSingleFlightKey(image, serviceAccountTokenHash, saAnnotations); err != nil {
			klog.Errorf("Error generating singleflight key: %v", err)
			return credentialprovider.DockerConfig{}, nil
		}
	}
	res, err, _ := p.group.Do(singleFlightKey, func() (interface{}, error) {
		return p.plugin.ExecPlugin(context.Background(), image, serviceAccountToken, saAnnotations)
	})

	if err != nil {
		klog.Errorf("Failed getting credential from external registry credential provider: %v", err)
		return credentialprovider.DockerConfig{}, nil
	}

	response, ok := res.(*credentialproviderapi.CredentialProviderResponse)
	if !ok {
		klog.Errorf("Invalid response type returned by external credential provider")
		return credentialprovider.DockerConfig{}, nil
	}
	if len(serviceAccountToken) > 0 && p.serviceAccountProvider.cacheType != kubeletconfig.TokenServiceAccountTokenCacheType {
		// validate that the response credentials are not the echoed token back verbatim when cache
		// type is not token. Only Token cache type is valid when the service account token
		// is returned as the registry credentials.
		for _, authConfig := range response.Auth {
			if authConfig.Password == serviceAccountToken {
				klog.Errorf("Credential provider plugin returned the service account token as the password for image %q, which is not allowed when service account cache type is not set to 'Token'", image)
				return credentialprovider.DockerConfig{}, nil
			}
		}
	}

	var cacheKey string
	switch cacheKeyType := response.CacheKeyType; cacheKeyType {
	case credentialproviderapi.ImagePluginCacheKeyType:
		cacheKey = image
	case credentialproviderapi.RegistryPluginCacheKeyType:
		registry := parseRegistry(image)
		cacheKey = registry
	case credentialproviderapi.GlobalPluginCacheKeyType:
		cacheKey = globalCacheKey
	default:
		klog.Errorf("credential provider plugin did not return a valid cacheKeyType: %q", cacheKeyType)
		return credentialprovider.DockerConfig{}, nil
	}

	dockerConfig := make(credentialprovider.DockerConfig, len(response.Auth))
	for matchImage, authConfig := range response.Auth {
		dockerConfig[matchImage] = credentialprovider.DockerConfigEntry{
			Username: authConfig.Username,
			Password: authConfig.Password,
		}
	}

	// cache duration was explicitly 0 so don't cache this response at all.
	if response.CacheDuration != nil && response.CacheDuration.Duration == 0 {
		return dockerConfig, serviceAccountCoordinates
	}

	var expiresAt time.Time
	// nil cache duration means use the default cache duration
	if response.CacheDuration == nil {
		if p.defaultCacheDuration == 0 {
			return dockerConfig, serviceAccountCoordinates
		}
		expiresAt = p.clock.Now().Add(p.defaultCacheDuration)
	} else {
		expiresAt = p.clock.Now().Add(response.CacheDuration.Duration)
	}

	cacheKey, err = generateCacheKey(cacheKey, serviceAccountCacheKey)
	if err != nil {
		klog.Errorf("Error generating cache key: %v", err)
		return credentialprovider.DockerConfig{}, nil
	}

	cachedEntry := &cacheEntry{
		key:         cacheKey,
		credentials: dockerConfig,
		expiresAt:   expiresAt,
	}

	if err := p.cache.Add(cachedEntry); err != nil {
		klog.Errorf("Error adding auth entry to cache: %v", err)
	}

	return dockerConfig, serviceAccountCoordinates
}

// isImageAllowed returns true if the image matches against the list of allowed matches by the plugin.
func (p *pluginProvider) isImageAllowed(image string) bool {
	for _, matchImage := range p.matchImages {
		if matched, _ := credentialprovider.URLsMatchStr(matchImage, image); matched {
			return true
		}
	}

	return false
}

// getCachedCredentials returns a credentialprovider.DockerConfig if cached from the plugin.
func (p *pluginProvider) getCachedCredentials(image, serviceAccountCacheKey string) (credentialprovider.DockerConfig, bool, error) {
	p.Lock()
	if p.clock.Now().After(p.lastCachePurge.Add(cachePurgeInterval)) {
		// NewExpirationCache purges expired entries when List() is called
		// The expired entry in the cache is removed only when Get or List called on it.
		// List() is called on some interval to remove those expired entries on which Get is never called.
		_ = p.cache.List()
		p.lastCachePurge = p.clock.Now()
	}
	p.Unlock()

	baseCacheKeyCandidates := []string{
		image,
		parseRegistry(image),
		globalCacheKey,
	}

	for _, baseCacheKey := range baseCacheKeyCandidates {
		if config, found, err := p.lookupCredentialsInCache(baseCacheKey, serviceAccountCacheKey); err != nil {
			return nil, false, err
		} else if found {
			return config, true, nil
		}
	}

	return nil, false, nil
}

// lookupCredentialsInCache performs a single cache lookup for the given base key and service account cache key
func (p *pluginProvider) lookupCredentialsInCache(baseKey, serviceAccountCacheKey string) (credentialprovider.DockerConfig, bool, error) {
	cacheKey, err := generateCacheKey(baseKey, serviceAccountCacheKey)
	if err != nil {
		return nil, false, fmt.Errorf("error generating cache key: %w", err)
	}

	obj, found, err := p.cache.GetByKey(cacheKey)
	if err != nil {
		return nil, false, err
	}

	if found {
		return obj.(*cacheEntry).credentials, true, nil
	}

	return nil, false, nil
}

// Plugin is the interface calling ExecPlugin. This is mainly for testability
// so tests don't have to actually exec any processes.
type Plugin interface {
	ExecPlugin(ctx context.Context, image, serviceAccountToken string, serviceAccountAnnotations map[string]string) (*credentialproviderapi.CredentialProviderResponse, error)
}

// execPlugin is the implementation of the Plugin interface that execs a credential provider plugin based
// on it's name provided in CredentialProviderConfig. It is assumed that the executable is available in the
// plugin directory provided by the kubelet.
type execPlugin struct {
	name         string
	apiVersion   string
	encoder      runtime.Encoder
	args         []string
	envVars      []kubeletconfig.ExecEnvVar
	pluginBinDir string
	environ      func() []string
}

// ExecPlugin executes the plugin binary with arguments and environment variables specified in CredentialProviderConfig:
//
//	$ ENV_NAME=ENV_VALUE <plugin-name> args[0] args[1] <<<request
//
// The plugin is expected to receive the CredentialProviderRequest API via stdin from the kubelet and
// return CredentialProviderResponse via stdout.
func (e *execPlugin) ExecPlugin(ctx context.Context, image, serviceAccountToken string, serviceAccountAnnotations map[string]string) (*credentialproviderapi.CredentialProviderResponse, error) {
	klog.V(5).Infof("Getting image %s credentials from external exec plugin %s", image, e.name)

	authRequest := &credentialproviderapi.CredentialProviderRequest{Image: image, ServiceAccountToken: serviceAccountToken, ServiceAccountAnnotations: serviceAccountAnnotations}
	data, err := e.encodeRequest(authRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to encode auth request: %w", err)
	}

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	stdin := bytes.NewBuffer(data)

	// Use a catch-all timeout of 1 minute for all exec-based plugins, this should leave enough
	// head room in case a plugin needs to retry a failed request while ensuring an exec plugin
	// does not run forever. In the future we may want this timeout to be tweakable from the plugin
	// config file.
	ctx, cancel := context.WithTimeout(ctx, 1*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, filepath.Join(e.pluginBinDir, e.name), e.args...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = stdout, stderr, stdin

	var configEnvVars []string
	for _, v := range e.envVars {
		configEnvVars = append(configEnvVars, fmt.Sprintf("%s=%s", v.Name, v.Value))
	}

	// Append current system environment variables, to the ones configured in the
	// credential provider file. Failing to do so may result in unsuccessful execution
	// of the provider binary, see https://github.com/kubernetes/kubernetes/issues/102750
	// also, this behaviour is inline with Credential Provider Config spec
	cmd.Env = mergeEnvVars(e.environ(), configEnvVars)

	if err = e.runPlugin(ctx, cmd, image); err != nil {
		return nil, fmt.Errorf("%w: %s", err, stderr.String())
	}

	data = stdout.Bytes()
	// check that the response apiVersion matches what is expected
	gvk, err := json.DefaultMetaFactory.Interpret(data)
	if err != nil {
		return nil, fmt.Errorf("error reading GVK from response: %w", err)
	}

	if gvk.GroupVersion().String() != e.apiVersion {
		return nil, fmt.Errorf("apiVersion from credential plugin response did not match expected apiVersion:%s, actual apiVersion:%s", e.apiVersion, gvk.GroupVersion().String())
	}

	response, err := e.decodeResponse(data)
	if err != nil {
		// err is explicitly not wrapped since it may contain credentials in the response.
		return nil, errors.New("error decoding credential provider plugin response from stdout")
	}

	return response, nil
}

func (e *execPlugin) runPlugin(ctx context.Context, cmd *exec.Cmd, image string) error {
	startTime := time.Now()
	defer func() {
		kubeletCredentialProviderPluginDuration.WithLabelValues(e.name).Observe(time.Since(startTime).Seconds())
	}()

	err := cmd.Run()
	if ctx.Err() != nil {
		kubeletCredentialProviderPluginErrors.WithLabelValues(e.name).Inc()
		return fmt.Errorf("error execing credential provider plugin %s for image %s: %w", e.name, image, ctx.Err())
	}
	if err != nil {
		kubeletCredentialProviderPluginErrors.WithLabelValues(e.name).Inc()
		return fmt.Errorf("error execing credential provider plugin %s for image %s: %w", e.name, image, err)
	}
	return nil
}

// encodeRequest encodes the internal CredentialProviderRequest type into the v1alpha1 version in json
func (e *execPlugin) encodeRequest(request *credentialproviderapi.CredentialProviderRequest) ([]byte, error) {
	data, err := runtime.Encode(e.encoder, request)
	if err != nil {
		return nil, fmt.Errorf("error encoding request: %w", err)
	}

	return data, nil
}

// decodeResponse decodes data into the internal CredentialProviderResponse type
func (e *execPlugin) decodeResponse(data []byte) (*credentialproviderapi.CredentialProviderResponse, error) {
	obj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}

	if gvk.Kind != "CredentialProviderResponse" {
		return nil, fmt.Errorf("failed to decode CredentialProviderResponse, unexpected Kind: %q", gvk.Kind)
	}

	if gvk.Group != credentialproviderapi.GroupName {
		return nil, fmt.Errorf("failed to decode CredentialProviderResponse, unexpected Group: %s", gvk.Group)
	}

	if internalResponse, ok := obj.(*credentialproviderapi.CredentialProviderResponse); ok {
		return internalResponse, nil
	}

	return nil, fmt.Errorf("unable to convert %T to *CredentialProviderResponse", obj)
}

// parseRegistry extracts the registry hostname of an image (including port if specified).
func parseRegistry(image string) string {
	imageParts := strings.Split(image, "/")
	return imageParts[0]
}

// mergedEnvVars overlays system defined env vars with credential provider env vars,
// it gives priority to the credential provider vars allowing user to override system
// env vars
func mergeEnvVars(sysEnvVars, credProviderVars []string) []string {
	mergedEnvVars := sysEnvVars
	mergedEnvVars = append(mergedEnvVars, credProviderVars...)
	return mergedEnvVars
}

func generateServiceAccountCacheKey(params cacheKeyParams) (string, error) {
	builder := newCacheKeyBuilder().
		addString(params.namespace).
		addString(params.serviceAccountName).
		addString(string(params.serviceAccountUID)).
		addAnnotations(params.saAnnotations)

	if params.cacheType == kubeletconfig.TokenServiceAccountTokenCacheType {
		builder = builder.addString("token").addString(params.saTokenHash)
	}

	return builder.build()
}

func generateCacheKey(baseKey, serviceAccountCacheKey string) (string, error) {
	builder := newCacheKeyBuilder().
		addString(baseKey).
		addString(serviceAccountCacheKey)

	return builder.build()
}

func generateSingleFlightKey(image, saTokenHash string, saAnnotations map[string]string) (string, error) {
	builder := newCacheKeyBuilder().
		addString(image).
		addString(saTokenHash).
		addAnnotations(saAnnotations)

	return builder.build()
}

// getHashIfNotEmpty returns the sha256 hash of the data if it is not empty.
func getHashIfNotEmpty(data string) string {
	if len(data) > 0 {
		return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))
	}
	return ""
}
