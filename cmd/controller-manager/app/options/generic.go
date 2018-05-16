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

package options

import (
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
)

// GenericComponentConfigOptions holds the options which are generic.
type GenericComponentConfigOptions struct {
	MinResyncPeriod         metav1.Duration
	ContentType             string
	KubeAPIQPS              float32
	KubeAPIBurst            int32
	ControllerStartInterval metav1.Duration
	LeaderElection          componentconfig.LeaderElectionConfiguration
}

// NewGenericComponentConfigOptions returns generic configuration default values for both
// the kube-controller-manager and the cloud-contoller-manager. Any common changes should
// be made here. Any individual changes should be made in that controller.
func NewGenericComponentConfigOptions(cfg componentconfig.GenericComponentConfiguration) *GenericComponentConfigOptions {
	o := &GenericComponentConfigOptions{
		MinResyncPeriod:         cfg.MinResyncPeriod,
		ContentType:             cfg.ContentType,
		KubeAPIQPS:              cfg.KubeAPIQPS,
		KubeAPIBurst:            cfg.KubeAPIBurst,
		ControllerStartInterval: cfg.ControllerStartInterval,
		LeaderElection:          cfg.LeaderElection,
	}

	return o
}

// AddFlags adds flags related to generic for controller manager to the specified FlagSet.
func (o *GenericComponentConfigOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.MinResyncPeriod.Duration, "min-resync-period", o.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod.")
	fs.StringVar(&o.ContentType, "kube-api-content-type", o.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.KubeAPIQPS, "kube-api-qps", o.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver.")
	fs.Int32Var(&o.KubeAPIBurst, "kube-api-burst", o.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver.")
	fs.DurationVar(&o.ControllerStartInterval.Duration, "controller-start-interval", o.ControllerStartInterval.Duration, "Interval between starting controller managers.")

	leaderelectionconfig.BindFlags(&o.LeaderElection, fs)
}

// ApplyTo fills up generic config with options.
func (o *GenericComponentConfigOptions) ApplyTo(cfg *componentconfig.GenericComponentConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.MinResyncPeriod = o.MinResyncPeriod
	cfg.ContentType = o.ContentType
	cfg.KubeAPIQPS = o.KubeAPIQPS
	cfg.KubeAPIBurst = o.KubeAPIBurst
	cfg.ControllerStartInterval = o.ControllerStartInterval
	cfg.LeaderElection = o.LeaderElection

	return nil
}

// Validate checks validation of GenericOptions.
func (o *GenericComponentConfigOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
