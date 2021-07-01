/*
Copyright 2021 The Kubernetes Authors.

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

package main

import (
	"os"
	"path/filepath"

	flag "github.com/spf13/pflag"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/klogr"
	"k8s.io/pod-security-admission/cmd/podsecuritywebhook/app"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

func main() {
	var configFile string
	options := ctrl.Options{
		Port:    webhook.DefaultPort,
		CertDir: filepath.Join(os.TempDir(), "k8s-webhook-server", "serving-certs"),
	}

	flag.IntVar(&options.Port, "port", options.Port, "The port that the webhook server serves at.")
	flag.StringVar(&options.CertDir, "cert-dir", options.CertDir, "The directory that contains the server key and certificate. "+
		"The server key and certificate must be named tls.key and tls.crt, respectively.")
	flag.StringVar(&configFile, "config", configFile, "The path to the PodSecurity configuration file.")
	flag.Parse()

	ctrl.SetLogger(klogr.New())

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), options)
	if err != nil {
		klog.ErrorS(err, "failed to initialize manager")
		os.Exit(1)
	}

	validator, err := app.NewValidator(mgr.GetClient(), mgr.GetCache(), configFile)
	if err != nil {
		klog.ErrorS(err, "failed to initialize validator")
		os.Exit(1)
	}

	mgr.GetWebhookServer().Register("/", &webhook.Admission{Handler: validator})

	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		klog.ErrorS(err, "problem starting webhook")
		os.Exit(1)
	}
}
