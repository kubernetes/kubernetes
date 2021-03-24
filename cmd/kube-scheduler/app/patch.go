package app

import (
	"time"

	"k8s.io/klog/v2"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"

	libgorestclient "github.com/openshift/library-go/pkg/config/client"
	"github.com/openshift/library-go/pkg/monitor/health"
)

func setUpPreferredHostForOpenShift(logger klog.Logger, kubeSchedulerOptions *options.Options) error {
	if !kubeSchedulerOptions.OpenShiftContext.UnsupportedKubeAPIOverPreferredHost {
		return nil
	}

	master := kubeSchedulerOptions.Master
	var kubeConfig string

	// We cannot load component config anymore as the options are not being initialized.
	// if there was no kubeconfig specified we won't be able to get cluster info.
	// in that case try to load the configuration and read kubeconfig directly from it if it was provided.
	if len(kubeSchedulerOptions.ConfigFile) > 0 {
		cfg, err := options.LoadKubeSchedulerConfiguration(logger, kubeSchedulerOptions.ConfigFile)
		if err != nil {
			return err
		}
		kubeConfig = cfg.ClientConnection.Kubeconfig
	}

	config, err := clientcmd.BuildConfigFromFlags(master, kubeConfig)
	if err != nil {
		return err
	}
	libgorestclient.DefaultServerName(config)

	targetProvider := health.StaticTargetProvider{"localhost:6443"}
	kubeSchedulerOptions.OpenShiftContext.PreferredHostHealthMonitor, err = health.New(targetProvider, createRestConfigForHealthMonitor(config))
	if err != nil {
		return err
	}
	kubeSchedulerOptions.OpenShiftContext.PreferredHostHealthMonitor.
		WithHealthyProbesThreshold(3).
		WithUnHealthyProbesThreshold(5).
		WithProbeInterval(5 * time.Second).
		WithProbeResponseTimeout(2 * time.Second).
		WithMetrics(health.Register(legacyregistry.MustRegister))

	kubeSchedulerOptions.OpenShiftContext.PreferredHostRoundTripperWrapperFn = libgorestclient.NewPreferredHostRoundTripper(func() string {
		healthyTargets, _ := kubeSchedulerOptions.OpenShiftContext.PreferredHostHealthMonitor.Targets()
		if len(healthyTargets) == 1 {
			return healthyTargets[0]
		}
		return ""
	})

	kubeSchedulerOptions.Authentication.WithCustomRoundTripper(kubeSchedulerOptions.OpenShiftContext.PreferredHostRoundTripperWrapperFn)
	kubeSchedulerOptions.Authorization.WithCustomRoundTripper(kubeSchedulerOptions.OpenShiftContext.PreferredHostRoundTripperWrapperFn)
	return nil
}

func createRestConfigForHealthMonitor(restConfig *rest.Config) *rest.Config {
	restConfigCopy := *restConfig
	rest.AddUserAgent(&restConfigCopy, "kube-scheduler-health-monitor")

	return &restConfigCopy
}
