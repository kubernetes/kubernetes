// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Viper is an application configuration system.
// It believes that applications can be configured a variety of ways
// via flags, ENVIRONMENT variables, configuration files retrieved
// from the file system, or a remote key/value store.

// Each item takes precedence over the item below it:

// overrides
// flag
// env
// config
// key/value store
// default

package viper

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/magiconair/properties"
	"github.com/mitchellh/mapstructure"
	"github.com/spf13/afero"
	"github.com/spf13/cast"
	"github.com/spf13/pflag"
	"github.com/subosito/gotenv"
	"gopkg.in/ini.v1"

	"github.com/spf13/viper/internal/encoding"
	"github.com/spf13/viper/internal/encoding/hcl"
	"github.com/spf13/viper/internal/encoding/json"
	"github.com/spf13/viper/internal/encoding/toml"
	"github.com/spf13/viper/internal/encoding/yaml"
)

// ConfigMarshalError happens when failing to marshal the configuration.
type ConfigMarshalError struct {
	err error
}

// Error returns the formatted configuration error.
func (e ConfigMarshalError) Error() string {
	return fmt.Sprintf("While marshaling config: %s", e.err.Error())
}

var v *Viper

type RemoteResponse struct {
	Value []byte
	Error error
}

var (
	encoderRegistry = encoding.NewEncoderRegistry()
	decoderRegistry = encoding.NewDecoderRegistry()
)

func init() {
	v = New()

	{
		codec := yaml.Codec{}

		encoderRegistry.RegisterEncoder("yaml", codec)
		decoderRegistry.RegisterDecoder("yaml", codec)

		encoderRegistry.RegisterEncoder("yml", codec)
		decoderRegistry.RegisterDecoder("yml", codec)
	}

	{
		codec := json.Codec{}

		encoderRegistry.RegisterEncoder("json", codec)
		decoderRegistry.RegisterDecoder("json", codec)
	}

	{
		codec := toml.Codec{}

		encoderRegistry.RegisterEncoder("toml", codec)
		decoderRegistry.RegisterDecoder("toml", codec)
	}

	{
		codec := hcl.Codec{}

		encoderRegistry.RegisterEncoder("hcl", codec)
		decoderRegistry.RegisterDecoder("hcl", codec)

		encoderRegistry.RegisterEncoder("tfvars", codec)
		decoderRegistry.RegisterDecoder("tfvars", codec)
	}
}

type remoteConfigFactory interface {
	Get(rp RemoteProvider) (io.Reader, error)
	Watch(rp RemoteProvider) (io.Reader, error)
	WatchChannel(rp RemoteProvider) (<-chan *RemoteResponse, chan bool)
}

// RemoteConfig is optional, see the remote package
var RemoteConfig remoteConfigFactory

// UnsupportedConfigError denotes encountering an unsupported
// configuration filetype.
type UnsupportedConfigError string

// Error returns the formatted configuration error.
func (str UnsupportedConfigError) Error() string {
	return fmt.Sprintf("Unsupported Config Type %q", string(str))
}

// UnsupportedRemoteProviderError denotes encountering an unsupported remote
// provider. Currently only etcd and Consul are supported.
type UnsupportedRemoteProviderError string

// Error returns the formatted remote provider error.
func (str UnsupportedRemoteProviderError) Error() string {
	return fmt.Sprintf("Unsupported Remote Provider Type %q", string(str))
}

// RemoteConfigError denotes encountering an error while trying to
// pull the configuration from the remote provider.
type RemoteConfigError string

// Error returns the formatted remote provider error
func (rce RemoteConfigError) Error() string {
	return fmt.Sprintf("Remote Configurations Error: %s", string(rce))
}

// ConfigFileNotFoundError denotes failing to find configuration file.
type ConfigFileNotFoundError struct {
	name, locations string
}

// Error returns the formatted configuration error.
func (fnfe ConfigFileNotFoundError) Error() string {
	return fmt.Sprintf("Config File %q Not Found in %q", fnfe.name, fnfe.locations)
}

// ConfigFileAlreadyExistsError denotes failure to write new configuration file.
type ConfigFileAlreadyExistsError string

// Error returns the formatted error when configuration already exists.
func (faee ConfigFileAlreadyExistsError) Error() string {
	return fmt.Sprintf("Config File %q Already Exists", string(faee))
}

// A DecoderConfigOption can be passed to viper.Unmarshal to configure
// mapstructure.DecoderConfig options
type DecoderConfigOption func(*mapstructure.DecoderConfig)

// DecodeHook returns a DecoderConfigOption which overrides the default
// DecoderConfig.DecodeHook value, the default is:
//
//  mapstructure.ComposeDecodeHookFunc(
//		mapstructure.StringToTimeDurationHookFunc(),
//		mapstructure.StringToSliceHookFunc(","),
//	)
func DecodeHook(hook mapstructure.DecodeHookFunc) DecoderConfigOption {
	return func(c *mapstructure.DecoderConfig) {
		c.DecodeHook = hook
	}
}

// Viper is a prioritized configuration registry. It
// maintains a set of configuration sources, fetches
// values to populate those, and provides them according
// to the source's priority.
// The priority of the sources is the following:
// 1. overrides
// 2. flags
// 3. env. variables
// 4. config file
// 5. key/value store
// 6. defaults
//
// For example, if values from the following sources were loaded:
//
//  Defaults : {
//  	"secret": "",
//  	"user": "default",
//  	"endpoint": "https://localhost"
//  }
//  Config : {
//  	"user": "root"
//  	"secret": "defaultsecret"
//  }
//  Env : {
//  	"secret": "somesecretkey"
//  }
//
// The resulting config will have the following values:
//
//	{
//		"secret": "somesecretkey",
//		"user": "root",
//		"endpoint": "https://localhost"
//	}
//
// Note: Vipers are not safe for concurrent Get() and Set() operations.
type Viper struct {
	// Delimiter that separates a list of keys
	// used to access a nested value in one go
	keyDelim string

	// A set of paths to look for the config file in
	configPaths []string

	// The filesystem to read config from.
	fs afero.Fs

	// A set of remote providers to search for the configuration
	remoteProviders []*defaultRemoteProvider

	// Name of file to look for inside the path
	configName        string
	configFile        string
	configType        string
	configPermissions os.FileMode
	envPrefix         string

	// Specific commands for ini parsing
	iniLoadOptions ini.LoadOptions

	automaticEnvApplied bool
	envKeyReplacer      StringReplacer
	allowEmptyEnv       bool

	config         map[string]interface{}
	override       map[string]interface{}
	defaults       map[string]interface{}
	kvstore        map[string]interface{}
	pflags         map[string]FlagValue
	env            map[string][]string
	aliases        map[string]string
	typeByDefValue bool

	// Store read properties on the object so that we can write back in order with comments.
	// This will only be used if the configuration read is a properties file.
	properties *properties.Properties

	onConfigChange func(fsnotify.Event)

	logger Logger
}

// New returns an initialized Viper instance.
func New() *Viper {
	v := new(Viper)
	v.keyDelim = "."
	v.configName = "config"
	v.configPermissions = os.FileMode(0o644)
	v.fs = afero.NewOsFs()
	v.config = make(map[string]interface{})
	v.override = make(map[string]interface{})
	v.defaults = make(map[string]interface{})
	v.kvstore = make(map[string]interface{})
	v.pflags = make(map[string]FlagValue)
	v.env = make(map[string][]string)
	v.aliases = make(map[string]string)
	v.typeByDefValue = false
	v.logger = jwwLogger{}

	return v
}

// Option configures Viper using the functional options paradigm popularized by Rob Pike and Dave Cheney.
// If you're unfamiliar with this style,
// see https://commandcenter.blogspot.com/2014/01/self-referential-functions-and-design.html and
// https://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis.
type Option interface {
	apply(v *Viper)
}

type optionFunc func(v *Viper)

func (fn optionFunc) apply(v *Viper) {
	fn(v)
}

// KeyDelimiter sets the delimiter used for determining key parts.
// By default it's value is ".".
func KeyDelimiter(d string) Option {
	return optionFunc(func(v *Viper) {
		v.keyDelim = d
	})
}

// StringReplacer applies a set of replacements to a string.
type StringReplacer interface {
	// Replace returns a copy of s with all replacements performed.
	Replace(s string) string
}

// EnvKeyReplacer sets a replacer used for mapping environment variables to internal keys.
func EnvKeyReplacer(r StringReplacer) Option {
	return optionFunc(func(v *Viper) {
		v.envKeyReplacer = r
	})
}

// NewWithOptions creates a new Viper instance.
func NewWithOptions(opts ...Option) *Viper {
	v := New()

	for _, opt := range opts {
		opt.apply(v)
	}

	return v
}

// Reset is intended for testing, will reset all to default settings.
// In the public interface for the viper package so applications
// can use it in their testing as well.
func Reset() {
	v = New()
	SupportedExts = []string{"json", "toml", "yaml", "yml", "properties", "props", "prop", "hcl", "tfvars", "dotenv", "env", "ini"}
	SupportedRemoteProviders = []string{"etcd", "consul", "firestore"}
}

type defaultRemoteProvider struct {
	provider      string
	endpoint      string
	path          string
	secretKeyring string
}

func (rp defaultRemoteProvider) Provider() string {
	return rp.provider
}

func (rp defaultRemoteProvider) Endpoint() string {
	return rp.endpoint
}

func (rp defaultRemoteProvider) Path() string {
	return rp.path
}

func (rp defaultRemoteProvider) SecretKeyring() string {
	return rp.secretKeyring
}

// RemoteProvider stores the configuration necessary
// to connect to a remote key/value store.
// Optional secretKeyring to unencrypt encrypted values
// can be provided.
type RemoteProvider interface {
	Provider() string
	Endpoint() string
	Path() string
	SecretKeyring() string
}

// SupportedExts are universally supported extensions.
var SupportedExts = []string{"json", "toml", "yaml", "yml", "properties", "props", "prop", "hcl", "tfvars", "dotenv", "env", "ini"}

// SupportedRemoteProviders are universally supported remote providers.
var SupportedRemoteProviders = []string{"etcd", "consul", "firestore"}

func OnConfigChange(run func(in fsnotify.Event)) { v.OnConfigChange(run) }
func (v *Viper) OnConfigChange(run func(in fsnotify.Event)) {
	v.onConfigChange = run
}

func WatchConfig() { v.WatchConfig() }

func (v *Viper) WatchConfig() {
	initWG := sync.WaitGroup{}
	initWG.Add(1)
	go func() {
		watcher, err := newWatcher()
		if err != nil {
			log.Fatal(err)
		}
		defer watcher.Close()
		// we have to watch the entire directory to pick up renames/atomic saves in a cross-platform way
		filename, err := v.getConfigFile()
		if err != nil {
			log.Printf("error: %v\n", err)
			initWG.Done()
			return
		}

		configFile := filepath.Clean(filename)
		configDir, _ := filepath.Split(configFile)
		realConfigFile, _ := filepath.EvalSymlinks(filename)

		eventsWG := sync.WaitGroup{}
		eventsWG.Add(1)
		go func() {
			for {
				select {
				case event, ok := <-watcher.Events:
					if !ok { // 'Events' channel is closed
						eventsWG.Done()
						return
					}
					currentConfigFile, _ := filepath.EvalSymlinks(filename)
					// we only care about the config file with the following cases:
					// 1 - if the config file was modified or created
					// 2 - if the real path to the config file changed (eg: k8s ConfigMap replacement)
					const writeOrCreateMask = fsnotify.Write | fsnotify.Create
					if (filepath.Clean(event.Name) == configFile &&
						event.Op&writeOrCreateMask != 0) ||
						(currentConfigFile != "" && currentConfigFile != realConfigFile) {
						realConfigFile = currentConfigFile
						err := v.ReadInConfig()
						if err != nil {
							log.Printf("error reading config file: %v\n", err)
						}
						if v.onConfigChange != nil {
							v.onConfigChange(event)
						}
					} else if filepath.Clean(event.Name) == configFile &&
						event.Op&fsnotify.Remove&fsnotify.Remove != 0 {
						eventsWG.Done()
						return
					}

				case err, ok := <-watcher.Errors:
					if ok { // 'Errors' channel is not closed
						log.Printf("watcher error: %v\n", err)
					}
					eventsWG.Done()
					return
				}
			}
		}()
		watcher.Add(configDir)
		initWG.Done()   // done initializing the watch in this go routine, so the parent routine can move on...
		eventsWG.Wait() // now, wait for event loop to end in this go-routine...
	}()
	initWG.Wait() // make sure that the go routine above fully ended before returning
}

// SetConfigFile explicitly defines the path, name and extension of the config file.
// Viper will use this and not check any of the config paths.
func SetConfigFile(in string) { v.SetConfigFile(in) }

func (v *Viper) SetConfigFile(in string) {
	if in != "" {
		v.configFile = in
	}
}

// SetEnvPrefix defines a prefix that ENVIRONMENT variables will use.
// E.g. if your prefix is "spf", the env registry will look for env
// variables that start with "SPF_".
func SetEnvPrefix(in string) { v.SetEnvPrefix(in) }

func (v *Viper) SetEnvPrefix(in string) {
	if in != "" {
		v.envPrefix = in
	}
}

func (v *Viper) mergeWithEnvPrefix(in string) string {
	if v.envPrefix != "" {
		return strings.ToUpper(v.envPrefix + "_" + in)
	}

	return strings.ToUpper(in)
}

// AllowEmptyEnv tells Viper to consider set,
// but empty environment variables as valid values instead of falling back.
// For backward compatibility reasons this is false by default.
func AllowEmptyEnv(allowEmptyEnv bool) { v.AllowEmptyEnv(allowEmptyEnv) }

func (v *Viper) AllowEmptyEnv(allowEmptyEnv bool) {
	v.allowEmptyEnv = allowEmptyEnv
}

// TODO: should getEnv logic be moved into find(). Can generalize the use of
// rewriting keys many things, Ex: Get('someKey') -> some_key
// (camel case to snake case for JSON keys perhaps)

// getEnv is a wrapper around os.Getenv which replaces characters in the original
// key. This allows env vars which have different keys than the config object
// keys.
func (v *Viper) getEnv(key string) (string, bool) {
	if v.envKeyReplacer != nil {
		key = v.envKeyReplacer.Replace(key)
	}

	val, ok := os.LookupEnv(key)

	return val, ok && (v.allowEmptyEnv || val != "")
}

// ConfigFileUsed returns the file used to populate the config registry.
func ConfigFileUsed() string            { return v.ConfigFileUsed() }
func (v *Viper) ConfigFileUsed() string { return v.configFile }

// AddConfigPath adds a path for Viper to search for the config file in.
// Can be called multiple times to define multiple search paths.
func AddConfigPath(in string) { v.AddConfigPath(in) }

func (v *Viper) AddConfigPath(in string) {
	if in != "" {
		absin := absPathify(v.logger, in)

		v.logger.Info("adding path to search paths", "path", absin)
		if !stringInSlice(absin, v.configPaths) {
			v.configPaths = append(v.configPaths, absin)
		}
	}
}

// AddRemoteProvider adds a remote configuration source.
// Remote Providers are searched in the order they are added.
// provider is a string value: "etcd", "consul" or "firestore" are currently supported.
// endpoint is the url.  etcd requires http://ip:port  consul requires ip:port
// path is the path in the k/v store to retrieve configuration
// To retrieve a config file called myapp.json from /configs/myapp.json
// you should set path to /configs and set config name (SetConfigName()) to
// "myapp"
func AddRemoteProvider(provider, endpoint, path string) error {
	return v.AddRemoteProvider(provider, endpoint, path)
}

func (v *Viper) AddRemoteProvider(provider, endpoint, path string) error {
	if !stringInSlice(provider, SupportedRemoteProviders) {
		return UnsupportedRemoteProviderError(provider)
	}
	if provider != "" && endpoint != "" {
		v.logger.Info("adding remote provider", "provider", provider, "endpoint", endpoint)

		rp := &defaultRemoteProvider{
			endpoint: endpoint,
			provider: provider,
			path:     path,
		}
		if !v.providerPathExists(rp) {
			v.remoteProviders = append(v.remoteProviders, rp)
		}
	}
	return nil
}

// AddSecureRemoteProvider adds a remote configuration source.
// Secure Remote Providers are searched in the order they are added.
// provider is a string value: "etcd", "consul" or "firestore" are currently supported.
// endpoint is the url.  etcd requires http://ip:port  consul requires ip:port
// secretkeyring is the filepath to your openpgp secret keyring.  e.g. /etc/secrets/myring.gpg
// path is the path in the k/v store to retrieve configuration
// To retrieve a config file called myapp.json from /configs/myapp.json
// you should set path to /configs and set config name (SetConfigName()) to
// "myapp"
// Secure Remote Providers are implemented with github.com/bketelsen/crypt
func AddSecureRemoteProvider(provider, endpoint, path, secretkeyring string) error {
	return v.AddSecureRemoteProvider(provider, endpoint, path, secretkeyring)
}

func (v *Viper) AddSecureRemoteProvider(provider, endpoint, path, secretkeyring string) error {
	if !stringInSlice(provider, SupportedRemoteProviders) {
		return UnsupportedRemoteProviderError(provider)
	}
	if provider != "" && endpoint != "" {
		v.logger.Info("adding remote provider", "provider", provider, "endpoint", endpoint)

		rp := &defaultRemoteProvider{
			endpoint:      endpoint,
			provider:      provider,
			path:          path,
			secretKeyring: secretkeyring,
		}
		if !v.providerPathExists(rp) {
			v.remoteProviders = append(v.remoteProviders, rp)
		}
	}
	return nil
}

func (v *Viper) providerPathExists(p *defaultRemoteProvider) bool {
	for _, y := range v.remoteProviders {
		if reflect.DeepEqual(y, p) {
			return true
		}
	}
	return false
}

// searchMap recursively searches for a value for path in source map.
// Returns nil if not found.
// Note: This assumes that the path entries and map keys are lower cased.
func (v *Viper) searchMap(source map[string]interface{}, path []string) interface{} {
	if len(path) == 0 {
		return source
	}

	next, ok := source[path[0]]
	if ok {
		// Fast path
		if len(path) == 1 {
			return next
		}

		// Nested case
		switch next.(type) {
		case map[interface{}]interface{}:
			return v.searchMap(cast.ToStringMap(next), path[1:])
		case map[string]interface{}:
			// Type assertion is safe here since it is only reached
			// if the type of `next` is the same as the type being asserted
			return v.searchMap(next.(map[string]interface{}), path[1:])
		default:
			// got a value but nested key expected, return "nil" for not found
			return nil
		}
	}
	return nil
}

// searchIndexableWithPathPrefixes recursively searches for a value for path in source map/slice.
//
// While searchMap() considers each path element as a single map key or slice index, this
// function searches for, and prioritizes, merged path elements.
// e.g., if in the source, "foo" is defined with a sub-key "bar", and "foo.bar"
// is also defined, this latter value is returned for path ["foo", "bar"].
//
// This should be useful only at config level (other maps may not contain dots
// in their keys).
//
// Note: This assumes that the path entries and map keys are lower cased.
func (v *Viper) searchIndexableWithPathPrefixes(source interface{}, path []string) interface{} {
	if len(path) == 0 {
		return source
	}

	// search for path prefixes, starting from the longest one
	for i := len(path); i > 0; i-- {
		prefixKey := strings.ToLower(strings.Join(path[0:i], v.keyDelim))

		var val interface{}
		switch sourceIndexable := source.(type) {
		case []interface{}:
			val = v.searchSliceWithPathPrefixes(sourceIndexable, prefixKey, i, path)
		case map[string]interface{}:
			val = v.searchMapWithPathPrefixes(sourceIndexable, prefixKey, i, path)
		}
		if val != nil {
			return val
		}
	}

	// not found
	return nil
}

// searchSliceWithPathPrefixes searches for a value for path in sourceSlice
//
// This function is part of the searchIndexableWithPathPrefixes recurring search and
// should not be called directly from functions other than searchIndexableWithPathPrefixes.
func (v *Viper) searchSliceWithPathPrefixes(
	sourceSlice []interface{},
	prefixKey string,
	pathIndex int,
	path []string,
) interface{} {
	// if the prefixKey is not a number or it is out of bounds of the slice
	index, err := strconv.Atoi(prefixKey)
	if err != nil || len(sourceSlice) <= index {
		return nil
	}

	next := sourceSlice[index]

	// Fast path
	if pathIndex == len(path) {
		return next
	}

	switch n := next.(type) {
	case map[interface{}]interface{}:
		return v.searchIndexableWithPathPrefixes(cast.ToStringMap(n), path[pathIndex:])
	case map[string]interface{}, []interface{}:
		return v.searchIndexableWithPathPrefixes(n, path[pathIndex:])
	default:
		// got a value but nested key expected, do nothing and look for next prefix
	}

	// not found
	return nil
}

// searchMapWithPathPrefixes searches for a value for path in sourceMap
//
// This function is part of the searchIndexableWithPathPrefixes recurring search and
// should not be called directly from functions other than searchIndexableWithPathPrefixes.
func (v *Viper) searchMapWithPathPrefixes(
	sourceMap map[string]interface{},
	prefixKey string,
	pathIndex int,
	path []string,
) interface{} {
	next, ok := sourceMap[prefixKey]
	if !ok {
		return nil
	}

	// Fast path
	if pathIndex == len(path) {
		return next
	}

	// Nested case
	switch n := next.(type) {
	case map[interface{}]interface{}:
		return v.searchIndexableWithPathPrefixes(cast.ToStringMap(n), path[pathIndex:])
	case map[string]interface{}, []interface{}:
		return v.searchIndexableWithPathPrefixes(n, path[pathIndex:])
	default:
		// got a value but nested key expected, do nothing and look for next prefix
	}

	// not found
	return nil
}

// isPathShadowedInDeepMap makes sure the given path is not shadowed somewhere
// on its path in the map.
// e.g., if "foo.bar" has a value in the given map, it “shadows”
//       "foo.bar.baz" in a lower-priority map
func (v *Viper) isPathShadowedInDeepMap(path []string, m map[string]interface{}) string {
	var parentVal interface{}
	for i := 1; i < len(path); i++ {
		parentVal = v.searchMap(m, path[0:i])
		if parentVal == nil {
			// not found, no need to add more path elements
			return ""
		}
		switch parentVal.(type) {
		case map[interface{}]interface{}:
			continue
		case map[string]interface{}:
			continue
		default:
			// parentVal is a regular value which shadows "path"
			return strings.Join(path[0:i], v.keyDelim)
		}
	}
	return ""
}

// isPathShadowedInFlatMap makes sure the given path is not shadowed somewhere
// in a sub-path of the map.
// e.g., if "foo.bar" has a value in the given map, it “shadows”
//       "foo.bar.baz" in a lower-priority map
func (v *Viper) isPathShadowedInFlatMap(path []string, mi interface{}) string {
	// unify input map
	var m map[string]interface{}
	switch mi.(type) {
	case map[string]string, map[string]FlagValue:
		m = cast.ToStringMap(mi)
	default:
		return ""
	}

	// scan paths
	var parentKey string
	for i := 1; i < len(path); i++ {
		parentKey = strings.Join(path[0:i], v.keyDelim)
		if _, ok := m[parentKey]; ok {
			return parentKey
		}
	}
	return ""
}

// isPathShadowedInAutoEnv makes sure the given path is not shadowed somewhere
// in the environment, when automatic env is on.
// e.g., if "foo.bar" has a value in the environment, it “shadows”
//       "foo.bar.baz" in a lower-priority map
func (v *Viper) isPathShadowedInAutoEnv(path []string) string {
	var parentKey string
	for i := 1; i < len(path); i++ {
		parentKey = strings.Join(path[0:i], v.keyDelim)
		if _, ok := v.getEnv(v.mergeWithEnvPrefix(parentKey)); ok {
			return parentKey
		}
	}
	return ""
}

// SetTypeByDefaultValue enables or disables the inference of a key value's
// type when the Get function is used based upon a key's default value as
// opposed to the value returned based on the normal fetch logic.
//
// For example, if a key has a default value of []string{} and the same key
// is set via an environment variable to "a b c", a call to the Get function
// would return a string slice for the key if the key's type is inferred by
// the default value and the Get function would return:
//
//   []string {"a", "b", "c"}
//
// Otherwise the Get function would return:
//
//   "a b c"
func SetTypeByDefaultValue(enable bool) { v.SetTypeByDefaultValue(enable) }

func (v *Viper) SetTypeByDefaultValue(enable bool) {
	v.typeByDefValue = enable
}

// GetViper gets the global Viper instance.
func GetViper() *Viper {
	return v
}

// Get can retrieve any value given the key to use.
// Get is case-insensitive for a key.
// Get has the behavior of returning the value associated with the first
// place from where it is set. Viper will check in the following order:
// override, flag, env, config file, key/value store, default
//
// Get returns an interface. For a specific value use one of the Get____ methods.
func Get(key string) interface{} { return v.Get(key) }

func (v *Viper) Get(key string) interface{} {
	lcaseKey := strings.ToLower(key)
	val := v.find(lcaseKey, true)
	if val == nil {
		return nil
	}

	if v.typeByDefValue {
		// TODO(bep) this branch isn't covered by a single test.
		valType := val
		path := strings.Split(lcaseKey, v.keyDelim)
		defVal := v.searchMap(v.defaults, path)
		if defVal != nil {
			valType = defVal
		}

		switch valType.(type) {
		case bool:
			return cast.ToBool(val)
		case string:
			return cast.ToString(val)
		case int32, int16, int8, int:
			return cast.ToInt(val)
		case uint:
			return cast.ToUint(val)
		case uint32:
			return cast.ToUint32(val)
		case uint64:
			return cast.ToUint64(val)
		case int64:
			return cast.ToInt64(val)
		case float64, float32:
			return cast.ToFloat64(val)
		case time.Time:
			return cast.ToTime(val)
		case time.Duration:
			return cast.ToDuration(val)
		case []string:
			return cast.ToStringSlice(val)
		case []int:
			return cast.ToIntSlice(val)
		}
	}

	return val
}

// Sub returns new Viper instance representing a sub tree of this instance.
// Sub is case-insensitive for a key.
func Sub(key string) *Viper { return v.Sub(key) }

func (v *Viper) Sub(key string) *Viper {
	subv := New()
	data := v.Get(key)
	if data == nil {
		return nil
	}

	if reflect.TypeOf(data).Kind() == reflect.Map {
		subv.config = cast.ToStringMap(data)
		return subv
	}
	return nil
}

// GetString returns the value associated with the key as a string.
func GetString(key string) string { return v.GetString(key) }

func (v *Viper) GetString(key string) string {
	return cast.ToString(v.Get(key))
}

// GetBool returns the value associated with the key as a boolean.
func GetBool(key string) bool { return v.GetBool(key) }

func (v *Viper) GetBool(key string) bool {
	return cast.ToBool(v.Get(key))
}

// GetInt returns the value associated with the key as an integer.
func GetInt(key string) int { return v.GetInt(key) }

func (v *Viper) GetInt(key string) int {
	return cast.ToInt(v.Get(key))
}

// GetInt32 returns the value associated with the key as an integer.
func GetInt32(key string) int32 { return v.GetInt32(key) }

func (v *Viper) GetInt32(key string) int32 {
	return cast.ToInt32(v.Get(key))
}

// GetInt64 returns the value associated with the key as an integer.
func GetInt64(key string) int64 { return v.GetInt64(key) }

func (v *Viper) GetInt64(key string) int64 {
	return cast.ToInt64(v.Get(key))
}

// GetUint returns the value associated with the key as an unsigned integer.
func GetUint(key string) uint { return v.GetUint(key) }

func (v *Viper) GetUint(key string) uint {
	return cast.ToUint(v.Get(key))
}

// GetUint32 returns the value associated with the key as an unsigned integer.
func GetUint32(key string) uint32 { return v.GetUint32(key) }

func (v *Viper) GetUint32(key string) uint32 {
	return cast.ToUint32(v.Get(key))
}

// GetUint64 returns the value associated with the key as an unsigned integer.
func GetUint64(key string) uint64 { return v.GetUint64(key) }

func (v *Viper) GetUint64(key string) uint64 {
	return cast.ToUint64(v.Get(key))
}

// GetFloat64 returns the value associated with the key as a float64.
func GetFloat64(key string) float64 { return v.GetFloat64(key) }

func (v *Viper) GetFloat64(key string) float64 {
	return cast.ToFloat64(v.Get(key))
}

// GetTime returns the value associated with the key as time.
func GetTime(key string) time.Time { return v.GetTime(key) }

func (v *Viper) GetTime(key string) time.Time {
	return cast.ToTime(v.Get(key))
}

// GetDuration returns the value associated with the key as a duration.
func GetDuration(key string) time.Duration { return v.GetDuration(key) }

func (v *Viper) GetDuration(key string) time.Duration {
	return cast.ToDuration(v.Get(key))
}

// GetIntSlice returns the value associated with the key as a slice of int values.
func GetIntSlice(key string) []int { return v.GetIntSlice(key) }

func (v *Viper) GetIntSlice(key string) []int {
	return cast.ToIntSlice(v.Get(key))
}

// GetStringSlice returns the value associated with the key as a slice of strings.
func GetStringSlice(key string) []string { return v.GetStringSlice(key) }

func (v *Viper) GetStringSlice(key string) []string {
	return cast.ToStringSlice(v.Get(key))
}

// GetStringMap returns the value associated with the key as a map of interfaces.
func GetStringMap(key string) map[string]interface{} { return v.GetStringMap(key) }

func (v *Viper) GetStringMap(key string) map[string]interface{} {
	return cast.ToStringMap(v.Get(key))
}

// GetStringMapString returns the value associated with the key as a map of strings.
func GetStringMapString(key string) map[string]string { return v.GetStringMapString(key) }

func (v *Viper) GetStringMapString(key string) map[string]string {
	return cast.ToStringMapString(v.Get(key))
}

// GetStringMapStringSlice returns the value associated with the key as a map to a slice of strings.
func GetStringMapStringSlice(key string) map[string][]string { return v.GetStringMapStringSlice(key) }

func (v *Viper) GetStringMapStringSlice(key string) map[string][]string {
	return cast.ToStringMapStringSlice(v.Get(key))
}

// GetSizeInBytes returns the size of the value associated with the given key
// in bytes.
func GetSizeInBytes(key string) uint { return v.GetSizeInBytes(key) }

func (v *Viper) GetSizeInBytes(key string) uint {
	sizeStr := cast.ToString(v.Get(key))
	return parseSizeInBytes(sizeStr)
}

// UnmarshalKey takes a single key and unmarshals it into a Struct.
func UnmarshalKey(key string, rawVal interface{}, opts ...DecoderConfigOption) error {
	return v.UnmarshalKey(key, rawVal, opts...)
}

func (v *Viper) UnmarshalKey(key string, rawVal interface{}, opts ...DecoderConfigOption) error {
	return decode(v.Get(key), defaultDecoderConfig(rawVal, opts...))
}

// Unmarshal unmarshals the config into a Struct. Make sure that the tags
// on the fields of the structure are properly set.
func Unmarshal(rawVal interface{}, opts ...DecoderConfigOption) error {
	return v.Unmarshal(rawVal, opts...)
}

func (v *Viper) Unmarshal(rawVal interface{}, opts ...DecoderConfigOption) error {
	return decode(v.AllSettings(), defaultDecoderConfig(rawVal, opts...))
}

// defaultDecoderConfig returns default mapsstructure.DecoderConfig with suppot
// of time.Duration values & string slices
func defaultDecoderConfig(output interface{}, opts ...DecoderConfigOption) *mapstructure.DecoderConfig {
	c := &mapstructure.DecoderConfig{
		Metadata:         nil,
		Result:           output,
		WeaklyTypedInput: true,
		DecodeHook: mapstructure.ComposeDecodeHookFunc(
			mapstructure.StringToTimeDurationHookFunc(),
			mapstructure.StringToSliceHookFunc(","),
		),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// A wrapper around mapstructure.Decode that mimics the WeakDecode functionality
func decode(input interface{}, config *mapstructure.DecoderConfig) error {
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return err
	}
	return decoder.Decode(input)
}

// UnmarshalExact unmarshals the config into a Struct, erroring if a field is nonexistent
// in the destination struct.
func UnmarshalExact(rawVal interface{}, opts ...DecoderConfigOption) error {
	return v.UnmarshalExact(rawVal, opts...)
}

func (v *Viper) UnmarshalExact(rawVal interface{}, opts ...DecoderConfigOption) error {
	config := defaultDecoderConfig(rawVal, opts...)
	config.ErrorUnused = true

	return decode(v.AllSettings(), config)
}

// BindPFlags binds a full flag set to the configuration, using each flag's long
// name as the config key.
func BindPFlags(flags *pflag.FlagSet) error { return v.BindPFlags(flags) }

func (v *Viper) BindPFlags(flags *pflag.FlagSet) error {
	return v.BindFlagValues(pflagValueSet{flags})
}

// BindPFlag binds a specific key to a pflag (as used by cobra).
// Example (where serverCmd is a Cobra instance):
//
//	 serverCmd.Flags().Int("port", 1138, "Port to run Application server on")
//	 Viper.BindPFlag("port", serverCmd.Flags().Lookup("port"))
//
func BindPFlag(key string, flag *pflag.Flag) error { return v.BindPFlag(key, flag) }

func (v *Viper) BindPFlag(key string, flag *pflag.Flag) error {
	if flag == nil {
		return fmt.Errorf("flag for %q is nil", key)
	}
	return v.BindFlagValue(key, pflagValue{flag})
}

// BindFlagValues binds a full FlagValue set to the configuration, using each flag's long
// name as the config key.
func BindFlagValues(flags FlagValueSet) error { return v.BindFlagValues(flags) }

func (v *Viper) BindFlagValues(flags FlagValueSet) (err error) {
	flags.VisitAll(func(flag FlagValue) {
		if err = v.BindFlagValue(flag.Name(), flag); err != nil {
			return
		}
	})
	return nil
}

// BindFlagValue binds a specific key to a FlagValue.
func BindFlagValue(key string, flag FlagValue) error { return v.BindFlagValue(key, flag) }

func (v *Viper) BindFlagValue(key string, flag FlagValue) error {
	if flag == nil {
		return fmt.Errorf("flag for %q is nil", key)
	}
	v.pflags[strings.ToLower(key)] = flag
	return nil
}

// BindEnv binds a Viper key to a ENV variable.
// ENV variables are case sensitive.
// If only a key is provided, it will use the env key matching the key, uppercased.
// If more arguments are provided, they will represent the env variable names that
// should bind to this key and will be taken in the specified order.
// EnvPrefix will be used when set when env name is not provided.
func BindEnv(input ...string) error { return v.BindEnv(input...) }

func (v *Viper) BindEnv(input ...string) error {
	if len(input) == 0 {
		return fmt.Errorf("missing key to bind to")
	}

	key := strings.ToLower(input[0])

	if len(input) == 1 {
		v.env[key] = append(v.env[key], v.mergeWithEnvPrefix(key))
	} else {
		v.env[key] = append(v.env[key], input[1:]...)
	}

	return nil
}

// Given a key, find the value.
//
// Viper will check to see if an alias exists first.
// Viper will then check in the following order:
// flag, env, config file, key/value store.
// Lastly, if no value was found and flagDefault is true, and if the key
// corresponds to a flag, the flag's default value is returned.
//
// Note: this assumes a lower-cased key given.
func (v *Viper) find(lcaseKey string, flagDefault bool) interface{} {
	var (
		val    interface{}
		exists bool
		path   = strings.Split(lcaseKey, v.keyDelim)
		nested = len(path) > 1
	)

	// compute the path through the nested maps to the nested value
	if nested && v.isPathShadowedInDeepMap(path, castMapStringToMapInterface(v.aliases)) != "" {
		return nil
	}

	// if the requested key is an alias, then return the proper key
	lcaseKey = v.realKey(lcaseKey)
	path = strings.Split(lcaseKey, v.keyDelim)
	nested = len(path) > 1

	// Set() override first
	val = v.searchMap(v.override, path)
	if val != nil {
		return val
	}
	if nested && v.isPathShadowedInDeepMap(path, v.override) != "" {
		return nil
	}

	// PFlag override next
	flag, exists := v.pflags[lcaseKey]
	if exists && flag.HasChanged() {
		switch flag.ValueType() {
		case "int", "int8", "int16", "int32", "int64":
			return cast.ToInt(flag.ValueString())
		case "bool":
			return cast.ToBool(flag.ValueString())
		case "stringSlice", "stringArray":
			s := strings.TrimPrefix(flag.ValueString(), "[")
			s = strings.TrimSuffix(s, "]")
			res, _ := readAsCSV(s)
			return res
		case "intSlice":
			s := strings.TrimPrefix(flag.ValueString(), "[")
			s = strings.TrimSuffix(s, "]")
			res, _ := readAsCSV(s)
			return cast.ToIntSlice(res)
		case "stringToString":
			return stringToStringConv(flag.ValueString())
		default:
			return flag.ValueString()
		}
	}
	if nested && v.isPathShadowedInFlatMap(path, v.pflags) != "" {
		return nil
	}

	// Env override next
	if v.automaticEnvApplied {
		// even if it hasn't been registered, if automaticEnv is used,
		// check any Get request
		if val, ok := v.getEnv(v.mergeWithEnvPrefix(lcaseKey)); ok {
			return val
		}
		if nested && v.isPathShadowedInAutoEnv(path) != "" {
			return nil
		}
	}
	envkeys, exists := v.env[lcaseKey]
	if exists {
		for _, envkey := range envkeys {
			if val, ok := v.getEnv(envkey); ok {
				return val
			}
		}
	}
	if nested && v.isPathShadowedInFlatMap(path, v.env) != "" {
		return nil
	}

	// Config file next
	val = v.searchIndexableWithPathPrefixes(v.config, path)
	if val != nil {
		return val
	}
	if nested && v.isPathShadowedInDeepMap(path, v.config) != "" {
		return nil
	}

	// K/V store next
	val = v.searchMap(v.kvstore, path)
	if val != nil {
		return val
	}
	if nested && v.isPathShadowedInDeepMap(path, v.kvstore) != "" {
		return nil
	}

	// Default next
	val = v.searchMap(v.defaults, path)
	if val != nil {
		return val
	}
	if nested && v.isPathShadowedInDeepMap(path, v.defaults) != "" {
		return nil
	}

	if flagDefault {
		// last chance: if no value is found and a flag does exist for the key,
		// get the flag's default value even if the flag's value has not been set.
		if flag, exists := v.pflags[lcaseKey]; exists {
			switch flag.ValueType() {
			case "int", "int8", "int16", "int32", "int64":
				return cast.ToInt(flag.ValueString())
			case "bool":
				return cast.ToBool(flag.ValueString())
			case "stringSlice", "stringArray":
				s := strings.TrimPrefix(flag.ValueString(), "[")
				s = strings.TrimSuffix(s, "]")
				res, _ := readAsCSV(s)
				return res
			case "intSlice":
				s := strings.TrimPrefix(flag.ValueString(), "[")
				s = strings.TrimSuffix(s, "]")
				res, _ := readAsCSV(s)
				return cast.ToIntSlice(res)
			case "stringToString":
				return stringToStringConv(flag.ValueString())
			default:
				return flag.ValueString()
			}
		}
		// last item, no need to check shadowing
	}

	return nil
}

func readAsCSV(val string) ([]string, error) {
	if val == "" {
		return []string{}, nil
	}
	stringReader := strings.NewReader(val)
	csvReader := csv.NewReader(stringReader)
	return csvReader.Read()
}

// mostly copied from pflag's implementation of this operation here https://github.com/spf13/pflag/blob/master/string_to_string.go#L79
// alterations are: errors are swallowed, map[string]interface{} is returned in order to enable cast.ToStringMap
func stringToStringConv(val string) interface{} {
	val = strings.Trim(val, "[]")
	// An empty string would cause an empty map
	if len(val) == 0 {
		return map[string]interface{}{}
	}
	r := csv.NewReader(strings.NewReader(val))
	ss, err := r.Read()
	if err != nil {
		return nil
	}
	out := make(map[string]interface{}, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return nil
		}
		out[kv[0]] = kv[1]
	}
	return out
}

// IsSet checks to see if the key has been set in any of the data locations.
// IsSet is case-insensitive for a key.
func IsSet(key string) bool { return v.IsSet(key) }

func (v *Viper) IsSet(key string) bool {
	lcaseKey := strings.ToLower(key)
	val := v.find(lcaseKey, false)
	return val != nil
}

// AutomaticEnv makes Viper check if environment variables match any of the existing keys
// (config, default or flags). If matching env vars are found, they are loaded into Viper.
func AutomaticEnv() { v.AutomaticEnv() }

func (v *Viper) AutomaticEnv() {
	v.automaticEnvApplied = true
}

// SetEnvKeyReplacer sets the strings.Replacer on the viper object
// Useful for mapping an environmental variable to a key that does
// not match it.
func SetEnvKeyReplacer(r *strings.Replacer) { v.SetEnvKeyReplacer(r) }

func (v *Viper) SetEnvKeyReplacer(r *strings.Replacer) {
	v.envKeyReplacer = r
}

// RegisterAlias creates an alias that provides another accessor for the same key.
// This enables one to change a name without breaking the application.
func RegisterAlias(alias string, key string) { v.RegisterAlias(alias, key) }

func (v *Viper) RegisterAlias(alias string, key string) {
	v.registerAlias(alias, strings.ToLower(key))
}

func (v *Viper) registerAlias(alias string, key string) {
	alias = strings.ToLower(alias)
	if alias != key && alias != v.realKey(key) {
		_, exists := v.aliases[alias]

		if !exists {
			// if we alias something that exists in one of the maps to another
			// name, we'll never be able to get that value using the original
			// name, so move the config value to the new realkey.
			if val, ok := v.config[alias]; ok {
				delete(v.config, alias)
				v.config[key] = val
			}
			if val, ok := v.kvstore[alias]; ok {
				delete(v.kvstore, alias)
				v.kvstore[key] = val
			}
			if val, ok := v.defaults[alias]; ok {
				delete(v.defaults, alias)
				v.defaults[key] = val
			}
			if val, ok := v.override[alias]; ok {
				delete(v.override, alias)
				v.override[key] = val
			}
			v.aliases[alias] = key
		}
	} else {
		v.logger.Warn("creating circular reference alias", "alias", alias, "key", key, "real_key", v.realKey(key))
	}
}

func (v *Viper) realKey(key string) string {
	newkey, exists := v.aliases[key]
	if exists {
		v.logger.Debug("key is an alias", "alias", key, "to", newkey)

		return v.realKey(newkey)
	}
	return key
}

// InConfig checks to see if the given key (or an alias) is in the config file.
func InConfig(key string) bool { return v.InConfig(key) }

func (v *Viper) InConfig(key string) bool {
	lcaseKey := strings.ToLower(key)

	// if the requested key is an alias, then return the proper key
	lcaseKey = v.realKey(lcaseKey)
	path := strings.Split(lcaseKey, v.keyDelim)

	return v.searchIndexableWithPathPrefixes(v.config, path) != nil
}

// SetDefault sets the default value for this key.
// SetDefault is case-insensitive for a key.
// Default only used when no value is provided by the user via flag, config or ENV.
func SetDefault(key string, value interface{}) { v.SetDefault(key, value) }

func (v *Viper) SetDefault(key string, value interface{}) {
	// If alias passed in, then set the proper default
	key = v.realKey(strings.ToLower(key))
	value = toCaseInsensitiveValue(value)

	path := strings.Split(key, v.keyDelim)
	lastKey := strings.ToLower(path[len(path)-1])
	deepestMap := deepSearch(v.defaults, path[0:len(path)-1])

	// set innermost value
	deepestMap[lastKey] = value
}

// Set sets the value for the key in the override register.
// Set is case-insensitive for a key.
// Will be used instead of values obtained via
// flags, config file, ENV, default, or key/value store.
func Set(key string, value interface{}) { v.Set(key, value) }

func (v *Viper) Set(key string, value interface{}) {
	// If alias passed in, then set the proper override
	key = v.realKey(strings.ToLower(key))
	value = toCaseInsensitiveValue(value)

	path := strings.Split(key, v.keyDelim)
	lastKey := strings.ToLower(path[len(path)-1])
	deepestMap := deepSearch(v.override, path[0:len(path)-1])

	// set innermost value
	deepestMap[lastKey] = value
}

// ReadInConfig will discover and load the configuration file from disk
// and key/value stores, searching in one of the defined paths.
func ReadInConfig() error { return v.ReadInConfig() }

func (v *Viper) ReadInConfig() error {
	v.logger.Info("attempting to read in config file")
	filename, err := v.getConfigFile()
	if err != nil {
		return err
	}

	if !stringInSlice(v.getConfigType(), SupportedExts) {
		return UnsupportedConfigError(v.getConfigType())
	}

	v.logger.Debug("reading file", "file", filename)
	file, err := afero.ReadFile(v.fs, filename)
	if err != nil {
		return err
	}

	config := make(map[string]interface{})

	err = v.unmarshalReader(bytes.NewReader(file), config)
	if err != nil {
		return err
	}

	v.config = config
	return nil
}

// MergeInConfig merges a new configuration with an existing config.
func MergeInConfig() error { return v.MergeInConfig() }

func (v *Viper) MergeInConfig() error {
	v.logger.Info("attempting to merge in config file")
	filename, err := v.getConfigFile()
	if err != nil {
		return err
	}

	if !stringInSlice(v.getConfigType(), SupportedExts) {
		return UnsupportedConfigError(v.getConfigType())
	}

	file, err := afero.ReadFile(v.fs, filename)
	if err != nil {
		return err
	}

	return v.MergeConfig(bytes.NewReader(file))
}

// ReadConfig will read a configuration file, setting existing keys to nil if the
// key does not exist in the file.
func ReadConfig(in io.Reader) error { return v.ReadConfig(in) }

func (v *Viper) ReadConfig(in io.Reader) error {
	v.config = make(map[string]interface{})
	return v.unmarshalReader(in, v.config)
}

// MergeConfig merges a new configuration with an existing config.
func MergeConfig(in io.Reader) error { return v.MergeConfig(in) }

func (v *Viper) MergeConfig(in io.Reader) error {
	cfg := make(map[string]interface{})
	if err := v.unmarshalReader(in, cfg); err != nil {
		return err
	}
	return v.MergeConfigMap(cfg)
}

// MergeConfigMap merges the configuration from the map given with an existing config.
// Note that the map given may be modified.
func MergeConfigMap(cfg map[string]interface{}) error { return v.MergeConfigMap(cfg) }

func (v *Viper) MergeConfigMap(cfg map[string]interface{}) error {
	if v.config == nil {
		v.config = make(map[string]interface{})
	}
	insensitiviseMap(cfg)
	mergeMaps(cfg, v.config, nil)
	return nil
}

// WriteConfig writes the current configuration to a file.
func WriteConfig() error { return v.WriteConfig() }

func (v *Viper) WriteConfig() error {
	filename, err := v.getConfigFile()
	if err != nil {
		return err
	}
	return v.writeConfig(filename, true)
}

// SafeWriteConfig writes current configuration to file only if the file does not exist.
func SafeWriteConfig() error { return v.SafeWriteConfig() }

func (v *Viper) SafeWriteConfig() error {
	if len(v.configPaths) < 1 {
		return errors.New("missing configuration for 'configPath'")
	}
	return v.SafeWriteConfigAs(filepath.Join(v.configPaths[0], v.configName+"."+v.configType))
}

// WriteConfigAs writes current configuration to a given filename.
func WriteConfigAs(filename string) error { return v.WriteConfigAs(filename) }

func (v *Viper) WriteConfigAs(filename string) error {
	return v.writeConfig(filename, true)
}

// SafeWriteConfigAs writes current configuration to a given filename if it does not exist.
func SafeWriteConfigAs(filename string) error { return v.SafeWriteConfigAs(filename) }

func (v *Viper) SafeWriteConfigAs(filename string) error {
	alreadyExists, err := afero.Exists(v.fs, filename)
	if alreadyExists && err == nil {
		return ConfigFileAlreadyExistsError(filename)
	}
	return v.writeConfig(filename, false)
}

func (v *Viper) writeConfig(filename string, force bool) error {
	v.logger.Info("attempting to write configuration to file")

	var configType string

	ext := filepath.Ext(filename)
	if ext != "" && ext != filepath.Base(filename) {
		configType = ext[1:]
	} else {
		configType = v.configType
	}
	if configType == "" {
		return fmt.Errorf("config type could not be determined for %s", filename)
	}

	if !stringInSlice(configType, SupportedExts) {
		return UnsupportedConfigError(configType)
	}
	if v.config == nil {
		v.config = make(map[string]interface{})
	}
	flags := os.O_CREATE | os.O_TRUNC | os.O_WRONLY
	if !force {
		flags |= os.O_EXCL
	}
	f, err := v.fs.OpenFile(filename, flags, v.configPermissions)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := v.marshalWriter(f, configType); err != nil {
		return err
	}

	return f.Sync()
}

// Unmarshal a Reader into a map.
// Should probably be an unexported function.
func unmarshalReader(in io.Reader, c map[string]interface{}) error {
	return v.unmarshalReader(in, c)
}

func (v *Viper) unmarshalReader(in io.Reader, c map[string]interface{}) error {
	buf := new(bytes.Buffer)
	buf.ReadFrom(in)

	switch format := strings.ToLower(v.getConfigType()); format {
	case "yaml", "yml", "json", "toml", "hcl", "tfvars":
		err := decoderRegistry.Decode(format, buf.Bytes(), &c)
		if err != nil {
			return ConfigParseError{err}
		}

	case "dotenv", "env":
		env, err := gotenv.StrictParse(buf)
		if err != nil {
			return ConfigParseError{err}
		}
		for k, v := range env {
			c[k] = v
		}

	case "properties", "props", "prop":
		v.properties = properties.NewProperties()
		var err error
		if v.properties, err = properties.Load(buf.Bytes(), properties.UTF8); err != nil {
			return ConfigParseError{err}
		}
		for _, key := range v.properties.Keys() {
			value, _ := v.properties.Get(key)
			// recursively build nested maps
			path := strings.Split(key, ".")
			lastKey := strings.ToLower(path[len(path)-1])
			deepestMap := deepSearch(c, path[0:len(path)-1])
			// set innermost value
			deepestMap[lastKey] = value
		}

	case "ini":
		cfg := ini.Empty(v.iniLoadOptions)
		err := cfg.Append(buf.Bytes())
		if err != nil {
			return ConfigParseError{err}
		}
		sections := cfg.Sections()
		for i := 0; i < len(sections); i++ {
			section := sections[i]
			keys := section.Keys()
			for j := 0; j < len(keys); j++ {
				key := keys[j]
				value := cfg.Section(section.Name()).Key(key.Name()).String()
				c[section.Name()+"."+key.Name()] = value
			}
		}
	}

	insensitiviseMap(c)
	return nil
}

// Marshal a map into Writer.
func (v *Viper) marshalWriter(f afero.File, configType string) error {
	c := v.AllSettings()
	switch configType {
	case "yaml", "yml", "json", "toml", "hcl", "tfvars":
		b, err := encoderRegistry.Encode(configType, c)
		if err != nil {
			return ConfigMarshalError{err}
		}

		_, err = f.WriteString(string(b))
		if err != nil {
			return ConfigMarshalError{err}
		}

	case "prop", "props", "properties":
		if v.properties == nil {
			v.properties = properties.NewProperties()
		}
		p := v.properties
		for _, key := range v.AllKeys() {
			_, _, err := p.Set(key, v.GetString(key))
			if err != nil {
				return ConfigMarshalError{err}
			}
		}
		_, err := p.WriteComment(f, "#", properties.UTF8)
		if err != nil {
			return ConfigMarshalError{err}
		}

	case "dotenv", "env":
		lines := []string{}
		for _, key := range v.AllKeys() {
			envName := strings.ToUpper(strings.Replace(key, ".", "_", -1))
			val := v.Get(key)
			lines = append(lines, fmt.Sprintf("%v=%v", envName, val))
		}
		s := strings.Join(lines, "\n")
		if _, err := f.WriteString(s); err != nil {
			return ConfigMarshalError{err}
		}

	case "ini":
		keys := v.AllKeys()
		cfg := ini.Empty()
		ini.PrettyFormat = false
		for i := 0; i < len(keys); i++ {
			key := keys[i]
			lastSep := strings.LastIndex(key, ".")
			sectionName := key[:(lastSep)]
			keyName := key[(lastSep + 1):]
			if sectionName == "default" {
				sectionName = ""
			}
			cfg.Section(sectionName).Key(keyName).SetValue(v.GetString(key))
		}
		cfg.WriteTo(f)
	}
	return nil
}

func keyExists(k string, m map[string]interface{}) string {
	lk := strings.ToLower(k)
	for mk := range m {
		lmk := strings.ToLower(mk)
		if lmk == lk {
			return mk
		}
	}
	return ""
}

func castToMapStringInterface(
	src map[interface{}]interface{}) map[string]interface{} {
	tgt := map[string]interface{}{}
	for k, v := range src {
		tgt[fmt.Sprintf("%v", k)] = v
	}
	return tgt
}

func castMapStringSliceToMapInterface(src map[string][]string) map[string]interface{} {
	tgt := map[string]interface{}{}
	for k, v := range src {
		tgt[k] = v
	}
	return tgt
}

func castMapStringToMapInterface(src map[string]string) map[string]interface{} {
	tgt := map[string]interface{}{}
	for k, v := range src {
		tgt[k] = v
	}
	return tgt
}

func castMapFlagToMapInterface(src map[string]FlagValue) map[string]interface{} {
	tgt := map[string]interface{}{}
	for k, v := range src {
		tgt[k] = v
	}
	return tgt
}

// mergeMaps merges two maps. The `itgt` parameter is for handling go-yaml's
// insistence on parsing nested structures as `map[interface{}]interface{}`
// instead of using a `string` as the key for nest structures beyond one level
// deep. Both map types are supported as there is a go-yaml fork that uses
// `map[string]interface{}` instead.
func mergeMaps(
	src, tgt map[string]interface{}, itgt map[interface{}]interface{}) {
	for sk, sv := range src {
		tk := keyExists(sk, tgt)
		if tk == "" {
			v.logger.Trace("", "tk", "\"\"", fmt.Sprintf("tgt[%s]", sk), sv)
			tgt[sk] = sv
			if itgt != nil {
				itgt[sk] = sv
			}
			continue
		}

		tv, ok := tgt[tk]
		if !ok {
			v.logger.Trace("", fmt.Sprintf("ok[%s]", tk), false, fmt.Sprintf("tgt[%s]", sk), sv)
			tgt[sk] = sv
			if itgt != nil {
				itgt[sk] = sv
			}
			continue
		}

		svType := reflect.TypeOf(sv)
		tvType := reflect.TypeOf(tv)
		if tvType != nil && svType != tvType { // Allow for the target to be nil
			v.logger.Error(
				"svType != tvType",
				"key", sk,
				"st", svType,
				"tt", tvType,
				"sv", sv,
				"tv", tv,
			)
			continue
		}

		v.logger.Trace(
			"processing",
			"key", sk,
			"st", svType,
			"tt", tvType,
			"sv", sv,
			"tv", tv,
		)

		switch ttv := tv.(type) {
		case map[interface{}]interface{}:
			v.logger.Trace("merging maps (must convert)")
			tsv := sv.(map[interface{}]interface{})
			ssv := castToMapStringInterface(tsv)
			stv := castToMapStringInterface(ttv)
			mergeMaps(ssv, stv, ttv)
		case map[string]interface{}:
			v.logger.Trace("merging maps")
			mergeMaps(sv.(map[string]interface{}), ttv, nil)
		default:
			v.logger.Trace("setting value")
			tgt[tk] = sv
			if itgt != nil {
				itgt[tk] = sv
			}
		}
	}
}

// ReadRemoteConfig attempts to get configuration from a remote source
// and read it in the remote configuration registry.
func ReadRemoteConfig() error { return v.ReadRemoteConfig() }

func (v *Viper) ReadRemoteConfig() error {
	return v.getKeyValueConfig()
}

func WatchRemoteConfig() error { return v.WatchRemoteConfig() }
func (v *Viper) WatchRemoteConfig() error {
	return v.watchKeyValueConfig()
}

func (v *Viper) WatchRemoteConfigOnChannel() error {
	return v.watchKeyValueConfigOnChannel()
}

// Retrieve the first found remote configuration.
func (v *Viper) getKeyValueConfig() error {
	if RemoteConfig == nil {
		return RemoteConfigError("Enable the remote features by doing a blank import of the viper/remote package: '_ github.com/spf13/viper/remote'")
	}

	for _, rp := range v.remoteProviders {
		val, err := v.getRemoteConfig(rp)
		if err != nil {
			v.logger.Error(fmt.Errorf("get remote config: %w", err).Error())

			continue
		}

		v.kvstore = val

		return nil
	}
	return RemoteConfigError("No Files Found")
}

func (v *Viper) getRemoteConfig(provider RemoteProvider) (map[string]interface{}, error) {
	reader, err := RemoteConfig.Get(provider)
	if err != nil {
		return nil, err
	}
	err = v.unmarshalReader(reader, v.kvstore)
	return v.kvstore, err
}

// Retrieve the first found remote configuration.
func (v *Viper) watchKeyValueConfigOnChannel() error {
	for _, rp := range v.remoteProviders {
		respc, _ := RemoteConfig.WatchChannel(rp)
		// Todo: Add quit channel
		go func(rc <-chan *RemoteResponse) {
			for {
				b := <-rc
				reader := bytes.NewReader(b.Value)
				v.unmarshalReader(reader, v.kvstore)
			}
		}(respc)
		return nil
	}
	return RemoteConfigError("No Files Found")
}

// Retrieve the first found remote configuration.
func (v *Viper) watchKeyValueConfig() error {
	for _, rp := range v.remoteProviders {
		val, err := v.watchRemoteConfig(rp)
		if err != nil {
			continue
		}
		v.kvstore = val
		return nil
	}
	return RemoteConfigError("No Files Found")
}

func (v *Viper) watchRemoteConfig(provider RemoteProvider) (map[string]interface{}, error) {
	reader, err := RemoteConfig.Watch(provider)
	if err != nil {
		return nil, err
	}
	err = v.unmarshalReader(reader, v.kvstore)
	return v.kvstore, err
}

// AllKeys returns all keys holding a value, regardless of where they are set.
// Nested keys are returned with a v.keyDelim separator
func AllKeys() []string { return v.AllKeys() }

func (v *Viper) AllKeys() []string {
	m := map[string]bool{}
	// add all paths, by order of descending priority to ensure correct shadowing
	m = v.flattenAndMergeMap(m, castMapStringToMapInterface(v.aliases), "")
	m = v.flattenAndMergeMap(m, v.override, "")
	m = v.mergeFlatMap(m, castMapFlagToMapInterface(v.pflags))
	m = v.mergeFlatMap(m, castMapStringSliceToMapInterface(v.env))
	m = v.flattenAndMergeMap(m, v.config, "")
	m = v.flattenAndMergeMap(m, v.kvstore, "")
	m = v.flattenAndMergeMap(m, v.defaults, "")

	// convert set of paths to list
	a := make([]string, 0, len(m))
	for x := range m {
		a = append(a, x)
	}
	return a
}

// flattenAndMergeMap recursively flattens the given map into a map[string]bool
// of key paths (used as a set, easier to manipulate than a []string):
// - each path is merged into a single key string, delimited with v.keyDelim
// - if a path is shadowed by an earlier value in the initial shadow map,
//   it is skipped.
// The resulting set of paths is merged to the given shadow set at the same time.
func (v *Viper) flattenAndMergeMap(shadow map[string]bool, m map[string]interface{}, prefix string) map[string]bool {
	if shadow != nil && prefix != "" && shadow[prefix] {
		// prefix is shadowed => nothing more to flatten
		return shadow
	}
	if shadow == nil {
		shadow = make(map[string]bool)
	}

	var m2 map[string]interface{}
	if prefix != "" {
		prefix += v.keyDelim
	}
	for k, val := range m {
		fullKey := prefix + k
		switch val.(type) {
		case map[string]interface{}:
			m2 = val.(map[string]interface{})
		case map[interface{}]interface{}:
			m2 = cast.ToStringMap(val)
		default:
			// immediate value
			shadow[strings.ToLower(fullKey)] = true
			continue
		}
		// recursively merge to shadow map
		shadow = v.flattenAndMergeMap(shadow, m2, fullKey)
	}
	return shadow
}

// mergeFlatMap merges the given maps, excluding values of the second map
// shadowed by values from the first map.
func (v *Viper) mergeFlatMap(shadow map[string]bool, m map[string]interface{}) map[string]bool {
	// scan keys
outer:
	for k := range m {
		path := strings.Split(k, v.keyDelim)
		// scan intermediate paths
		var parentKey string
		for i := 1; i < len(path); i++ {
			parentKey = strings.Join(path[0:i], v.keyDelim)
			if shadow[parentKey] {
				// path is shadowed, continue
				continue outer
			}
		}
		// add key
		shadow[strings.ToLower(k)] = true
	}
	return shadow
}

// AllSettings merges all settings and returns them as a map[string]interface{}.
func AllSettings() map[string]interface{} { return v.AllSettings() }

func (v *Viper) AllSettings() map[string]interface{} {
	m := map[string]interface{}{}
	// start from the list of keys, and construct the map one value at a time
	for _, k := range v.AllKeys() {
		value := v.Get(k)
		if value == nil {
			// should not happen, since AllKeys() returns only keys holding a value,
			// check just in case anything changes
			continue
		}
		path := strings.Split(k, v.keyDelim)
		lastKey := strings.ToLower(path[len(path)-1])
		deepestMap := deepSearch(m, path[0:len(path)-1])
		// set innermost value
		deepestMap[lastKey] = value
	}
	return m
}

// SetFs sets the filesystem to use to read configuration.
func SetFs(fs afero.Fs) { v.SetFs(fs) }

func (v *Viper) SetFs(fs afero.Fs) {
	v.fs = fs
}

// SetConfigName sets name for the config file.
// Does not include extension.
func SetConfigName(in string) { v.SetConfigName(in) }

func (v *Viper) SetConfigName(in string) {
	if in != "" {
		v.configName = in
		v.configFile = ""
	}
}

// SetConfigType sets the type of the configuration returned by the
// remote source, e.g. "json".
func SetConfigType(in string) { v.SetConfigType(in) }

func (v *Viper) SetConfigType(in string) {
	if in != "" {
		v.configType = in
	}
}

// SetConfigPermissions sets the permissions for the config file.
func SetConfigPermissions(perm os.FileMode) { v.SetConfigPermissions(perm) }

func (v *Viper) SetConfigPermissions(perm os.FileMode) {
	v.configPermissions = perm.Perm()
}

// IniLoadOptions sets the load options for ini parsing.
func IniLoadOptions(in ini.LoadOptions) Option {
	return optionFunc(func(v *Viper) {
		v.iniLoadOptions = in
	})
}

func (v *Viper) getConfigType() string {
	if v.configType != "" {
		return v.configType
	}

	cf, err := v.getConfigFile()
	if err != nil {
		return ""
	}

	ext := filepath.Ext(cf)

	if len(ext) > 1 {
		return ext[1:]
	}

	return ""
}

func (v *Viper) getConfigFile() (string, error) {
	if v.configFile == "" {
		cf, err := v.findConfigFile()
		if err != nil {
			return "", err
		}
		v.configFile = cf
	}
	return v.configFile, nil
}

// Debug prints all configuration registries for debugging
// purposes.
func Debug() { v.Debug() }

func (v *Viper) Debug() {
	fmt.Printf("Aliases:\n%#v\n", v.aliases)
	fmt.Printf("Override:\n%#v\n", v.override)
	fmt.Printf("PFlags:\n%#v\n", v.pflags)
	fmt.Printf("Env:\n%#v\n", v.env)
	fmt.Printf("Key/Value Store:\n%#v\n", v.kvstore)
	fmt.Printf("Config:\n%#v\n", v.config)
	fmt.Printf("Defaults:\n%#v\n", v.defaults)
}
