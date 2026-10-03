/*
Copyright The Kubernetes Authors.

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

package exec

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/tools/clientcmd/api"
)

// cacheKey is a stable hash of the configuration used to initialize an
// authenticator.
type cacheKey [sha256.Size]byte

// newCacheKey returns a stable cache key for an exec configuration and cluster.
// The configuration is hashed to avoid retaining potentially sensitive values
// such as arguments and environment variables in the cache key.
func newCacheKey(conf *api.ExecConfig, cluster *clientauthentication.Cluster) (cacheKey, error) {
	return newCacheKeyData(conf, cluster).key()
}

func newCacheKeyData(conf *api.ExecConfig, cluster *clientauthentication.Cluster) cacheKeyData {
	return cacheKeyData{
		ExecConfig: newExecConfigCacheKeyData(conf),
		Cluster:    newClusterCacheKeyData(cluster),
	}
}

type cacheKeyData struct {
	ExecConfig execConfigCacheKeyData `json:"execConfig"`
	Cluster    *clusterCacheKeyData   `json:"cluster,omitempty"`
}

// key returns the SHA-256 digest of the cache identity data.
//
// JSON encoding preserves values omitted by String and GoString and provides
// stable map ordering, while hashing avoids retaining potentially sensitive
// values in the cache key.
func (d cacheKeyData) key() (cacheKey, error) {
	data, err := json.Marshal(d)
	if err != nil {
		// Avoid returning the original marshal error, which may include values
		// from custom marshalers and therefore expose sensitive configuration.
		if typeErr, ok := errors.AsType[*json.UnsupportedTypeError](err); ok {
			return cacheKey{}, fmt.Errorf("cannot marshal unsupported type %s", typeErr.Type)
		}
		if _, ok := errors.AsType[*json.UnsupportedValueError](err); ok {
			return cacheKey{}, errors.New("cannot marshal unsupported value")
		}
		if marshalerErr, ok := errors.AsType[*json.MarshalerError](err); ok {
			return cacheKey{}, fmt.Errorf("cannot marshal type %s", marshalerErr.Type)
		}
		return cacheKey{}, errors.New("cannot marshal cache key")
	}

	return sha256.Sum256(data), nil
}

func newExecConfigCacheKeyData(conf *api.ExecConfig) execConfigCacheKeyData {
	return execConfigCacheKeyData{
		Command:                 conf.Command,
		Args:                    conf.Args,
		Env:                     conf.Env,
		APIVersion:              conf.APIVersion,
		InstallHint:             conf.InstallHint,
		ProvideClusterInfo:      conf.ProvideClusterInfo,
		Config:                  newObjectCacheKeyData(conf.Config),
		InteractiveMode:         conf.InteractiveMode,
		StdinUnavailable:        conf.StdinUnavailable,
		StdinUnavailableMessage: conf.StdinUnavailableMessage,
		PluginPolicy:            newPluginPolicyCacheKeyData(conf.PluginPolicy),
	}
}

// Fields from [api.ExecConfig] that affect authenticator identity.
type execConfigCacheKeyData struct {
	Command                 string                   `json:"command"`
	Args                    []string                 `json:"args"`
	Env                     []api.ExecEnvVar         `json:"env"`
	APIVersion              string                   `json:"apiVersion"`
	InstallHint             string                   `json:"installHint"`
	ProvideClusterInfo      bool                     `json:"provideClusterInfo"`
	Config                  *objectCacheKeyData      `json:"config,omitempty"`
	InteractiveMode         api.ExecInteractiveMode  `json:"interactiveMode"`
	StdinUnavailable        bool                     `json:"stdinUnavailable"`
	StdinUnavailableMessage string                   `json:"stdinUnavailableMessage"`
	PluginPolicy            pluginPolicyCacheKeyData `json:"pluginPolicy"`
}

func newPluginPolicyCacheKeyData(policy api.PluginPolicy) pluginPolicyCacheKeyData {
	allowlist := make([]allowlistEntryCacheKeyData, len(policy.Allowlist))
	for i, entry := range policy.Allowlist {
		allowlist[i] = allowlistEntryCacheKeyData{
			Command: entry.Command,
		}
	}

	return pluginPolicyCacheKeyData{
		PolicyType: policy.PolicyType,
		Allowlist:  allowlist,
	}
}

// pluginPolicyCacheKeyData is the representation of api.PluginPolicy used when
// computing an exec credential cache key. It includes fields that are
// intentionally excluded from JSON serialization on api.PluginPolicy.
type pluginPolicyCacheKeyData struct {
	PolicyType api.PolicyType
	Allowlist  []allowlistEntryCacheKeyData
}

// allowlistEntryCacheKeyData is the representation of api.AllowlistEntry used
// when computing an exec credential cache key. It includes fields that are
// intentionally excluded from JSON serialization on api.AllowlistEntry.
type allowlistEntryCacheKeyData struct {
	Command string
}

func newClusterCacheKeyData(cluster *clientauthentication.Cluster) *clusterCacheKeyData {
	if cluster == nil {
		return nil
	}

	return &clusterCacheKeyData{
		Server:                   cluster.Server,
		TLSServerName:            cluster.TLSServerName,
		InsecureSkipTLSVerify:    cluster.InsecureSkipTLSVerify,
		CertificateAuthorityData: cluster.CertificateAuthorityData,
		ProxyURL:                 cluster.ProxyURL,
		DisableCompression:       cluster.DisableCompression,
		Config:                   newObjectCacheKeyData(cluster.Config),
	}
}

// Fields from [clientauthentication.Cluster] that affect authenticator identity.
type clusterCacheKeyData struct {
	Server                   string `json:"server"`
	TLSServerName            string `json:"tlsServerName"`
	InsecureSkipTLSVerify    bool   `json:"insecureSkipTLSVerify"`
	CertificateAuthorityData []byte `json:"certificateAuthorityData"`
	ProxyURL                 string `json:"proxyURL"`
	DisableCompression       bool   `json:"disableCompression"`

	Config *objectCacheKeyData `json:"config,omitempty"`
}

func newObjectCacheKeyData(obj runtime.Object) *objectCacheKeyData {
	if obj == nil {
		return nil
	}
	return &objectCacheKeyData{Object: obj}
}

// objectCacheKeyData defines the observable runtime.Object state that
// participates in cache identity. JSON-visible state is included; unexported
// runtime state is intentionally ignored.
type objectCacheKeyData struct {
	Object runtime.Object `json:"object"`
}

func newCache() *cache {
	return &cache{m: make(map[cacheKey]*Authenticator)}
}

type cache struct {
	mu sync.Mutex
	m  map[cacheKey]*Authenticator
}

func (c *cache) get(s cacheKey) (*Authenticator, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	a, ok := c.m[s]
	return a, ok
}

// put inserts an authenticator into the cache. If an authenticator is already
// associated with the key, the first one is returned instead.
func (c *cache) put(s cacheKey, a *Authenticator) *Authenticator {
	c.mu.Lock()
	defer c.mu.Unlock()
	existing, ok := c.m[s]
	if ok {
		return existing
	}
	c.m[s] = a
	return a
}
