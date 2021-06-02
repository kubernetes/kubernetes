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

package options

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/klog/v2"
)

type EtcdOptions struct {
	// The value of Paging on StorageConfig will be overridden by the
	// calculated feature gate value.
	StorageConfig                    storagebackend.Config
	EncryptionProviderConfigFilepath string

	EtcdServersOverrides []string

	// To enable protobuf as storage format, it is enough
	// to set it to "application/vnd.kubernetes.protobuf".
	DefaultStorageMediaType string
	DeleteCollectionWorkers int
	EnableGarbageCollection bool

	// Set EnableWatchCache to false to disable all watch caches
	EnableWatchCache bool
	// Set DefaultWatchCacheSize to zero to disable watch caches for those resources that have no explicit cache size set
	DefaultWatchCacheSize int
	// WatchCacheSizes represents override to a given resource
	WatchCacheSizes []string
}

var storageTypes = sets.NewString(
	storagebackend.StorageTypeETCD3,
)

func NewEtcdOptions(backendConfig *storagebackend.Config) *EtcdOptions {
	options := &EtcdOptions{
		StorageConfig:           *backendConfig,
		DefaultStorageMediaType: "application/json",
		DeleteCollectionWorkers: 1,
		EnableGarbageCollection: true,
		EnableWatchCache:        true,
		DefaultWatchCacheSize:   100,
	}
	options.StorageConfig.CountMetricPollPeriod = time.Minute
	return options
}

func (s *EtcdOptions) Validate() []error {
	if s == nil {
		return nil
	}

	allErrors := []error{}
	if len(s.StorageConfig.Transport.ServerList) == 0 {
		allErrors = append(allErrors, fmt.Errorf("--etcd-servers must be specified"))
	}

	if s.StorageConfig.Type != storagebackend.StorageTypeUnset && !storageTypes.Has(s.StorageConfig.Type) {
		allErrors = append(allErrors, fmt.Errorf("--storage-backend invalid, allowed values: %s. If not specified, it will default to 'etcd3'", strings.Join(storageTypes.List(), ", ")))
	}

	for _, override := range s.EtcdServersOverrides {
		tokens := strings.Split(override, "#")
		if len(tokens) != 2 {
			allErrors = append(allErrors, fmt.Errorf("--etcd-servers-overrides invalid, must be of format: group/resource#servers, where servers are URLs, semicolon separated"))
			continue
		}

		apiresource := strings.Split(tokens[0], "/")
		if len(apiresource) != 2 {
			allErrors = append(allErrors, fmt.Errorf("--etcd-servers-overrides invalid, must be of format: group/resource#servers, where servers are URLs, semicolon separated"))
			continue
		}

	}

	return allErrors
}

// AddEtcdFlags adds flags related to etcd storage for a specific APIServer to the specified FlagSet
func (s *EtcdOptions) AddFlags(fs *pflag.FlagSet) {
	if s == nil {
		return
	}

	fs.StringSliceVar(&s.EtcdServersOverrides, "etcd-servers-overrides", s.EtcdServersOverrides, ""+
		"Per-resource etcd servers overrides, comma separated. The individual override "+
		"format: group/resource#servers, where servers are URLs, semicolon separated. "+
		"Note that this applies only to resources compiled into this server binary. ")

	fs.StringVar(&s.DefaultStorageMediaType, "storage-media-type", s.DefaultStorageMediaType, ""+
		"The media type to use to store objects in storage. "+
		"Some resources or storage backends may only support a specific media type and will ignore this setting.")
	fs.IntVar(&s.DeleteCollectionWorkers, "delete-collection-workers", s.DeleteCollectionWorkers,
		"Number of workers spawned for DeleteCollection call. These are used to speed up namespace cleanup.")

	fs.BoolVar(&s.EnableGarbageCollection, "enable-garbage-collector", s.EnableGarbageCollection, ""+
		"Enables the generic garbage collector. MUST be synced with the corresponding flag "+
		"of the kube-controller-manager.")

	fs.BoolVar(&s.EnableWatchCache, "watch-cache", s.EnableWatchCache,
		"Enable watch caching in the apiserver")

	fs.IntVar(&s.DefaultWatchCacheSize, "default-watch-cache-size", s.DefaultWatchCacheSize,
		"Default watch cache size. If zero, watch cache will be disabled for resources that do not have a default watch size set.")

	fs.StringSliceVar(&s.WatchCacheSizes, "watch-cache-sizes", s.WatchCacheSizes, ""+
		"Watch cache size settings for some resources (pods, nodes, etc.), comma separated. "+
		"The individual setting format: resource[.group]#size, where resource is lowercase plural (no version), "+
		"group is omitted for resources of apiVersion v1 (the legacy core API) and included for others, "+
		"and size is a number. It takes effect when watch-cache is enabled. "+
		"Some resources (replicationcontrollers, endpoints, nodes, pods, services, apiservices.apiregistration.k8s.io) "+
		"have system defaults set by heuristics, others default to default-watch-cache-size")

	fs.StringVar(&s.StorageConfig.Type, "storage-backend", s.StorageConfig.Type,
		"The storage backend for persistence. Options: 'etcd3' (default).")

	dummyCacheSize := 0
	fs.IntVar(&dummyCacheSize, "deserialization-cache-size", 0, "Number of deserialized json objects to cache in memory.")
	fs.MarkDeprecated("deserialization-cache-size", "the deserialization cache was dropped in 1.13 with support for etcd2")

	fs.StringSliceVar(&s.StorageConfig.Transport.ServerList, "etcd-servers", s.StorageConfig.Transport.ServerList,
		"List of etcd servers to connect with (scheme://ip:port), comma separated.")

	fs.StringVar(&s.StorageConfig.Prefix, "etcd-prefix", s.StorageConfig.Prefix,
		"The prefix to prepend to all resource paths in etcd.")

	fs.StringVar(&s.StorageConfig.Transport.KeyFile, "etcd-keyfile", s.StorageConfig.Transport.KeyFile,
		"SSL key file used to secure etcd communication.")

	fs.StringVar(&s.StorageConfig.Transport.CertFile, "etcd-certfile", s.StorageConfig.Transport.CertFile,
		"SSL certification file used to secure etcd communication.")

	fs.StringVar(&s.StorageConfig.Transport.TrustedCAFile, "etcd-cafile", s.StorageConfig.Transport.TrustedCAFile,
		"SSL Certificate Authority file used to secure etcd communication.")

	fs.StringVar(&s.EncryptionProviderConfigFilepath, "experimental-encryption-provider-config", s.EncryptionProviderConfigFilepath,
		"The file containing configuration for encryption providers to be used for storing secrets in etcd")
	fs.MarkDeprecated("experimental-encryption-provider-config", "use --encryption-provider-config.")

	fs.StringVar(&s.EncryptionProviderConfigFilepath, "encryption-provider-config", s.EncryptionProviderConfigFilepath,
		"The file containing configuration for encryption providers to be used for storing secrets in etcd")

	fs.DurationVar(&s.StorageConfig.CompactionInterval, "etcd-compaction-interval", s.StorageConfig.CompactionInterval,
		"The interval of compaction requests. If 0, the compaction request from apiserver is disabled.")

	fs.DurationVar(&s.StorageConfig.CountMetricPollPeriod, "etcd-count-metric-poll-period", s.StorageConfig.CountMetricPollPeriod, ""+
		"Frequency of polling etcd for number of resources per type. 0 disables the metric collection.")

	fs.DurationVar(&s.StorageConfig.DBMetricPollInterval, "etcd-db-metric-poll-interval", s.StorageConfig.DBMetricPollInterval,
		"The interval of requests to poll etcd and update metric. 0 disables the metric collection")

	fs.DurationVar(&s.StorageConfig.HealthcheckTimeout, "etcd-healthcheck-timeout", s.StorageConfig.HealthcheckTimeout,
		"The timeout to use when checking etcd health.")

	fs.Int64Var(&s.StorageConfig.LeaseManagerConfig.ReuseDurationSeconds, "lease-reuse-duration-seconds", s.StorageConfig.LeaseManagerConfig.ReuseDurationSeconds,
		"The time in seconds that each lease is reused. A lower value could avoid large number of objects reusing the same lease. Notice that a too small value may cause performance problems at storage layer.")
}

func (s *EtcdOptions) ApplyTo(c *server.Config) error {
	if s == nil {
		return nil
	}
	if err := s.addEtcdHealthEndpoint(c); err != nil {
		return err
	}
	transformerOverrides := make(map[schema.GroupResource]value.Transformer)
	if len(s.EncryptionProviderConfigFilepath) > 0 {
		var err error
		transformerOverrides, err = encryptionconfig.GetTransformerOverrides(s.EncryptionProviderConfigFilepath)
		if err != nil {
			return err
		}
	}

	c.RESTOptionsGetter = &SimpleRestOptionsFactory{
		Options:              *s,
		TransformerOverrides: transformerOverrides,
	}
	return nil
}

func (s *EtcdOptions) ApplyWithStorageFactoryTo(factory serverstorage.StorageFactory, c *server.Config) error {
	if err := s.addEtcdHealthEndpoint(c); err != nil {
		return err
	}
	c.RESTOptionsGetter = &StorageFactoryRestOptionsFactory{Options: *s, StorageFactory: factory}
	return nil
}

func (s *EtcdOptions) addEtcdHealthEndpoint(c *server.Config) error {
	healthCheck, err := storagefactory.CreateHealthCheck(s.StorageConfig)
	if err != nil {
		return err
	}
	c.AddHealthChecks(healthz.NamedCheck("etcd", func(r *http.Request) error {
		return healthCheck()
	}))

	if s.EncryptionProviderConfigFilepath != "" {
		kmsPluginHealthzChecks, err := encryptionconfig.GetKMSPluginHealthzCheckers(s.EncryptionProviderConfigFilepath)
		if err != nil {
			return err
		}
		c.AddHealthChecks(kmsPluginHealthzChecks...)
	}

	return nil
}

type SimpleRestOptionsFactory struct {
	Options              EtcdOptions
	TransformerOverrides map[schema.GroupResource]value.Transformer
}

func (f *SimpleRestOptionsFactory) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	ret := generic.RESTOptions{
		StorageConfig:           &f.Options.StorageConfig,
		Decorator:               generic.UndecoratedStorage,
		EnableGarbageCollection: f.Options.EnableGarbageCollection,
		DeleteCollectionWorkers: f.Options.DeleteCollectionWorkers,
		ResourcePrefix:          resource.Group + "/" + resource.Resource,
		CountMetricPollPeriod:   f.Options.StorageConfig.CountMetricPollPeriod,
	}
	if f.TransformerOverrides != nil {
		if transformer, ok := f.TransformerOverrides[resource]; ok {
			ret.StorageConfig.Transformer = transformer
		}
	}
	if f.Options.EnableWatchCache {
		sizes, err := ParseWatchCacheSizes(f.Options.WatchCacheSizes)
		if err != nil {
			return generic.RESTOptions{}, err
		}
		size, ok := sizes[resource]
		if ok && size > 0 {
			klog.Warningf("Dropping watch-cache-size for %v - watchCache size is now dynamic", resource)
		}
		if ok && size <= 0 {
			ret.Decorator = generic.UndecoratedStorage
		} else {
			ret.Decorator = genericregistry.StorageWithCacher()
		}
	}
	return ret, nil
}

type StorageFactoryRestOptionsFactory struct {
	Options        EtcdOptions
	StorageFactory serverstorage.StorageFactory
}

func (f *StorageFactoryRestOptionsFactory) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	storageConfig, err := f.StorageFactory.NewConfig(resource)
	if err != nil {
		return generic.RESTOptions{}, fmt.Errorf("unable to find storage destination for %v, due to %v", resource, err.Error())
	}

	ret := generic.RESTOptions{
		StorageConfig:           storageConfig,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: f.Options.DeleteCollectionWorkers,
		EnableGarbageCollection: f.Options.EnableGarbageCollection,
		ResourcePrefix:          f.StorageFactory.ResourcePrefix(resource),
		CountMetricPollPeriod:   f.Options.StorageConfig.CountMetricPollPeriod,
	}
	if f.Options.EnableWatchCache {
		sizes, err := ParseWatchCacheSizes(f.Options.WatchCacheSizes)
		if err != nil {
			return generic.RESTOptions{}, err
		}
		size, ok := sizes[resource]
		if ok && size > 0 {
			klog.Warningf("Dropping watch-cache-size for %v - watchCache size is now dynamic", resource)
		}
		if ok && size <= 0 {
			ret.Decorator = generic.UndecoratedStorage
		} else {
			ret.Decorator = genericregistry.StorageWithCacher()
		}
	}

	return ret, nil
}

// ParseWatchCacheSizes turns a list of cache size values into a map of group resources
// to requested sizes.
func ParseWatchCacheSizes(cacheSizes []string) (map[schema.GroupResource]int, error) {
	watchCacheSizes := make(map[schema.GroupResource]int)
	for _, c := range cacheSizes {
		tokens := strings.Split(c, "#")
		if len(tokens) != 2 {
			return nil, fmt.Errorf("invalid value of watch cache size: %s", c)
		}

		size, err := strconv.Atoi(tokens[1])
		if err != nil {
			return nil, fmt.Errorf("invalid size of watch cache size: %s", c)
		}
		if size < 0 {
			return nil, fmt.Errorf("watch cache size cannot be negative: %s", c)
		}
		watchCacheSizes[schema.ParseGroupResource(tokens[0])] = size
	}
	return watchCacheSizes, nil
}

// WriteWatchCacheSizes turns a map of cache size values into a list of string specifications.
func WriteWatchCacheSizes(watchCacheSizes map[schema.GroupResource]int) ([]string, error) {
	var cacheSizes []string

	for resource, size := range watchCacheSizes {
		if size < 0 {
			return nil, fmt.Errorf("watch cache size cannot be negative for resource %s", resource)
		}
		cacheSizes = append(cacheSizes, fmt.Sprintf("%s#%d", resource.String(), size))
	}
	return cacheSizes, nil
}
