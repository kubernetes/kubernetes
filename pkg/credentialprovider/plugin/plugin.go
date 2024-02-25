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
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	credentialproviderapi "k8s.io/kubelet/pkg/apis/credentialprovider"
	"k8s.io/kubelet/pkg/apis/credentialprovider/install"
	credentialproviderv1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1"
	credentialproviderv1alpha1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1"
	credentialproviderv1beta1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1beta1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigv1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/config/v1beta1"
	"k8s.io/utils/clock"
)

const (
	globalCacheKey     = "global"
	cachePurgeInterval = time.Minute * 15
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)

	apiVersions = map[string]schema.GroupVersion{
		credentialproviderv1alpha1.SchemeGroupVersion.String(): credentialproviderv1alpha1.SchemeGroupVersion,
		credentialproviderv1beta1.SchemeGroupVersion.String():  credentialproviderv1beta1.SchemeGroupVersion,
		credentialproviderv1.SchemeGroupVersion.String():       credentialproviderv1.SchemeGroupVersion,
	}
)

func init() {
	install.Install(scheme)
	kubeletconfig.AddToScheme(scheme)
	kubeletconfigv1alpha1.AddToScheme(scheme)
	kubeletconfigv1beta1.AddToScheme(scheme)
	kubeletconfigv1.AddToScheme(scheme)
}

// RegisterCredentialProviderPlugins is called from kubelet to register external credential provider
// plugins according to the CredentialProviderConfig config file.
func RegisterCredentialProviderPlugins(pluginConfigFile, pluginBinDir string) error {
	if _, err := os.Stat(pluginBinDir); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("plugin binary directory %s did not exist", pluginBinDir)
		}

		return fmt.Errorf("error inspecting binary directory %s: %w", pluginBinDir, err)
	}

	credentialProviderConfig, err := readCredentialProviderConfigFile(pluginConfigFile)
	if err != nil {
		return err
	}

	errs := validateCredentialProviderConfig(credentialProviderConfig)
	if len(errs) > 0 {
		return fmt.Errorf("failed to validate credential provider config: %v", errs.ToAggregate())
	}

	// Register metrics for credential providers
	registerMetrics()

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

		plugin, err := newPluginProvider(pluginBinDir, provider)
		if err != nil {
			return fmt.Errorf("error initializing plugin provider %s: %w", provider.Name, err)
		}

		credentialprovider.RegisterCredentialProvider(provider.Name, plugin)
	}

	return nil
}

// newPluginProvider returns a new pluginProvider based on the credential provider config.
func newPluginProvider(pluginBinDir string, provider kubeletconfig.CredentialProvider) (*pluginProvider, error) {
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

// Provide returns a credentialprovider.DockerConfig based on the credentials returned
// from cache or the exec plugin.
func (p *pluginProvider) Provide(image string) credentialprovider.DockerConfig {
	if !p.isImageAllowed(image) {
		return credentialprovider.DockerConfig{}
	}

	cachedConfig, found, err := p.getCachedCredentials(image)
	if err != nil {
		klog.Errorf("Failed to get cached docker config: %v", err)
		return credentialprovider.DockerConfig{}
	}

	if found {
		return cachedConfig
	}

	// ExecPlugin is wrapped in single flight to exec plugin once for concurrent same image request.
	// The caveat here is we don't know cacheKeyType yet, so if cacheKeyType is registry/global and credentials saved in cache
	// on per registry/global basis then exec will be called for all requests if requests are made concurrently.
	// foo.bar.registry
	// foo.bar.registry/image1
	// foo.bar.registry/image2
	res, err, _ := p.group.Do(image, func() (interface{}, error) {
		return p.plugin.ExecPlugin(context.Background(), image)
	})

	if err != nil {
		klog.Errorf("Failed getting credential from external registry credential provider: %v", err)
		return credentialprovider.DockerConfig{}
	}

	response, ok := res.(*credentialproviderapi.CredentialProviderResponse)
	if !ok {
		klog.Errorf("Invalid response type returned by external credential provider")
		return credentialprovider.DockerConfig{}
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
		return credentialprovider.DockerConfig{}
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
		return dockerConfig
	}

	var expiresAt time.Time
	// nil cache duration means use the default cache duration
	if response.CacheDuration == nil {
		if p.defaultCacheDuration == 0 {
			return dockerConfig
		}
		expiresAt = p.clock.Now().Add(p.defaultCacheDuration)
	} else {
		expiresAt = p.clock.Now().Add(response.CacheDuration.Duration)
	}

	cachedEntry := &cacheEntry{
		key:         cacheKey,
		credentials: dockerConfig,
		expiresAt:   expiresAt,
	}

	if err := p.cache.Add(cachedEntry); err != nil {
		klog.Errorf("Error adding auth entry to cache: %v", err)
	}

	return dockerConfig
}

// Enabled always returns true since registration of the plugin via kubelet implies it should be enabled.
func (p *pluginProvider) Enabled() bool {
	return true
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
func (p *pluginProvider) getCachedCredentials(image string) (credentialprovider.DockerConfig, bool, error) {
	p.Lock()
	if p.clock.Now().After(p.lastCachePurge.Add(cachePurgeInterval)) {
		// NewExpirationCache purges expired entries when List() is called
		// The expired entry in the cache is removed only when Get or List called on it.
		// List() is called on some interval to remove those expired entries on which Get is never called.
		_ = p.cache.List()
		p.lastCachePurge = p.clock.Now()
	}
	p.Unlock()

	obj, found, err := p.cache.GetByKey(image)
	if err != nil {
		return nil, false, err
	}

	if found {
		return obj.(*cacheEntry).credentials, true, nil
	}

	registry := parseRegistry(image)
	obj, found, err = p.cache.GetByKey(registry)
	if err != nil {
		return nil, false, err
	}

	if found {
		return obj.(*cacheEntry).credentials, true, nil
	}

	obj, found, err = p.cache.GetByKey(globalCacheKey)
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
	ExecPlugin(ctx context.Context, image string) (*credentialproviderapi.CredentialProviderResponse, error)
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
func (e *execPlugin) ExecPlugin(ctx context.Context, image string) (*credentialproviderapi.CredentialProviderResponse, error) {
	klog.V(5).Infof("Getting image %s credentials from external exec plugin %s", image, e.name)

	authRequest := &credentialproviderapi.CredentialProviderRequest{Image: image}
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
