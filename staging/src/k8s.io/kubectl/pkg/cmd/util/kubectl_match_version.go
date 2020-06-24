/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"sync"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubectl/pkg/scheme"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/component-base/version"
)

const (
	flagMatchBinaryVersion                        = "match-server-version"
	suppressVersionSkewWarningEnvironmentVariable = "KUBECTL_SUPPRESS_VERSION_SKEW_WARNING"
)

// MatchVersionFlags is for setting the "match server version" function.
type MatchVersionFlags struct {
	Delegate genericclioptions.RESTClientGetter

	RequireMatchedServerVersion bool
	checkServerVersion          sync.Once
	matchesServerVersionErr     error

	clientVersion            apimachineryversion.Info
	versionSkewWarningWriter io.Writer
}

var _ genericclioptions.RESTClientGetter = &MatchVersionFlags{}

var leadingDigitsRegexp = regexp.MustCompile(`^(\d+)`)

func (f *MatchVersionFlags) checkMatchingServerVersion() error {
	f.checkServerVersion.Do(func() {
		if !f.RequireMatchedServerVersion {
			f.warnIfUnsupportedVersionSkew()
			return
		}
		discoveryClient, err := f.Delegate.ToDiscoveryClient()
		if err != nil {
			f.matchesServerVersionErr = err
			return
		}
		f.matchesServerVersionErr = discovery.MatchesServerVersion(f.clientVersion, discoveryClient)
	})

	return f.matchesServerVersionErr
}

func (f *MatchVersionFlags) warnIfUnsupportedVersionSkew() {
	if os.Getenv(suppressVersionSkewWarningEnvironmentVariable) == "true" {
		return
	}

	writer := f.versionSkewWarningWriter
	if writer == nil {
		writer = os.Stderr
	}

	discoveryClient, err := f.Delegate.ToDiscoveryClient()
	if err != nil {
		fmt.Fprintf(writer, "WARNING: version skew could not be checked because an error occurred while getting the discovery client: %v\n", err)
		return
	}
	serverVersion, err := discoveryClient.ServerVersion()
	if err != nil {
		fmt.Fprintf(writer, "WARNING: version skew could not be checked because an error occurred while getting the server version: %v\n", err)
		return
	}
	serverMajorVersion, serverMinorVersion, err := parseMajorMinorVersion(*serverVersion)
	if err != nil {
		fmt.Fprintf(writer, "WARNING: version skew could not be checked because an error occurred while parsing the server version: %v\n", err)
		return
	}

	clientMajorVersion, clientMinorVersion, err := parseMajorMinorVersion(f.clientVersion)
	if err != nil {
		fmt.Fprintf(writer, "WARNING: version skew could not be checked because an error occurred while parsing the client version: %v\n", err)
		return
	}

	const supportedSkew = 1
	if clientMajorVersion != serverMajorVersion || clientMinorVersion < serverMinorVersion-supportedSkew || clientMinorVersion > serverMinorVersion+supportedSkew {
		fmt.Fprintf(writer, "WARNING: version difference between client (%s.%s) and server (%s.%s) exceeds the supported minor version skew of +/-%d\n",
			f.clientVersion.Major, f.clientVersion.Minor, serverVersion.Major, serverVersion.Minor, supportedSkew)
	}
}

func parseMajorMinorVersion(v apimachineryversion.Info) (int, int, error) {
	major, err := strconv.Atoi(v.Major)
	if err != nil {
		return 0, 0, err
	}
	// Regex used to extract only leading digits because minor version can optionally include a plus sign (ex: 1.19+)
	minor, err := strconv.Atoi(leadingDigitsRegexp.FindString(v.Minor))
	if err != nil {
		return 0, 0, err
	}
	return major, minor, nil
}

// ToRESTConfig implements RESTClientGetter.
// Returns a REST client configuration based on a provided path
// to a .kubeconfig file, loading rules, and config flag overrides.
// Expects the AddFlags method to have been called.
func (f *MatchVersionFlags) ToRESTConfig() (*rest.Config, error) {
	if err := f.checkMatchingServerVersion(); err != nil {
		return nil, err
	}
	clientConfig, err := f.Delegate.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	// TODO we should not have to do this.  It smacks of something going wrong.
	setKubernetesDefaults(clientConfig)
	return clientConfig, nil
}

func (f *MatchVersionFlags) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	return f.Delegate.ToRawKubeConfigLoader()
}

func (f *MatchVersionFlags) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	if err := f.checkMatchingServerVersion(); err != nil {
		return nil, err
	}
	return f.Delegate.ToDiscoveryClient()
}

// ToRESTMapper returns a mapper.
func (f *MatchVersionFlags) ToRESTMapper() (meta.RESTMapper, error) {
	if err := f.checkMatchingServerVersion(); err != nil {
		return nil, err
	}
	return f.Delegate.ToRESTMapper()
}

func (f *MatchVersionFlags) AddFlags(flags *pflag.FlagSet) {
	flags.BoolVar(&f.RequireMatchedServerVersion, flagMatchBinaryVersion, f.RequireMatchedServerVersion, "Require server version to match client version")
}

func NewMatchVersionFlags(delegate genericclioptions.RESTClientGetter) *MatchVersionFlags {
	return &MatchVersionFlags{
		Delegate:                 delegate,
		clientVersion:            version.Get(),
		versionSkewWarningWriter: os.Stderr,
	}
}

// setKubernetesDefaults sets default values on the provided client config for accessing the
// Kubernetes API or returns an error if any of the defaults are impossible or invalid.
// TODO this isn't what we want.  Each clientset should be setting defaults as it sees fit.
func setKubernetesDefaults(config *rest.Config) error {
	// TODO remove this hack.  This is allowing the GetOptions to be serialized.
	config.GroupVersion = &schema.GroupVersion{Group: "", Version: "v1"}

	if config.APIPath == "" {
		config.APIPath = "/api"
	}
	if config.NegotiatedSerializer == nil {
		// This codec factory ensures the resources are not converted. Therefore, resources
		// will not be round-tripped through internal versions. Defaulting does not happen
		// on the client.
		config.NegotiatedSerializer = scheme.Codecs.WithoutConversion()
	}
	return rest.SetKubernetesDefaults(config)
}
