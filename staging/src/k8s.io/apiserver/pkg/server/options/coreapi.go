/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/util/feature"
	clientgoinformers "k8s.io/client-go/informers"
	clientgoclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	tracing "k8s.io/component-base/tracing"
)

// CoreAPIOptions contains options to configure the connection to a core API Kubernetes apiserver.
type CoreAPIOptions struct {
	// CoreAPIKubeconfigPath is a filename for a kubeconfig file to contact the core API server with.
	// If it is not set, the in cluster config is used.
	CoreAPIKubeconfigPath string
}

func NewCoreAPIOptions() *CoreAPIOptions {
	return &CoreAPIOptions{}
}

func (o *CoreAPIOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.CoreAPIKubeconfigPath, "kubeconfig", o.CoreAPIKubeconfigPath,
		"kubeconfig file pointing at the 'core' kubernetes server.")
}

func (o *CoreAPIOptions) ApplyTo(config *server.RecommendedConfig) error {
	if o == nil {
		return nil
	}

	// create shared informer for Kubernetes APIs
	var kubeconfig *rest.Config
	var err error
	if len(o.CoreAPIKubeconfigPath) > 0 {
		loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: o.CoreAPIKubeconfigPath}
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})
		kubeconfig, err = loader.ClientConfig()
		if err != nil {
			return fmt.Errorf("failed to load kubeconfig at %q: %v", o.CoreAPIKubeconfigPath, err)
		}
	} else {
		kubeconfig, err = rest.InClusterConfig()
		if err != nil {
			return err
		}
	}
	if feature.DefaultFeatureGate.Enabled(features.APIServerTracing) {
		kubeconfig.Wrap(tracing.WrapperFor(config.TracerProvider))
	}
	clientgoExternalClient, err := clientgoclientset.NewForConfig(kubeconfig)
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes clientset: %v", err)
	}
	config.ClientConfig = kubeconfig
	config.SharedInformerFactory = clientgoinformers.NewSharedInformerFactory(clientgoExternalClient, 10*time.Minute)

	return nil
}

func (o *CoreAPIOptions) Validate() []error {
	return nil
}
