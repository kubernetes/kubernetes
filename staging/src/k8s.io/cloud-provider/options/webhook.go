/*
Copyright 2022 The Kubernetes Authors.

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
	"bytes"
	"context"
	"encoding/pem"
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/cloud-provider/config"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

const (
	CloudControllerManagerWebhookPort = 10260
)

type WebhookOptions struct {
	// Webhooks is the list of webhook names that should be enabled or disabled
	Webhooks []string
	// ValidatingWebhookConfigFilePath is the file containing the validating webhook configuration details
	ValidatingWebhookConfigFilePath string
	// ValidatingWebhookConfiguration is the decoded data from the file to be used during validation
	ValidatingWebhookConfiguration *admissionregistrationv1.ValidatingWebhookConfiguration
	// MutatingWebhookConfigFilePath is the file containing the mutating webhook configuration details
	MutatingWebhookConfigFilePath string
	// MutatingWebhookConfiguration is the decoded data from the file to be used during validation
	MutatingWebhookConfiguration *admissionregistrationv1.MutatingWebhookConfiguration
}

func NewWebhookOptions() *WebhookOptions {
	o := &WebhookOptions{}
	return o
}

func (o *WebhookOptions) AddFlags(fs *pflag.FlagSet, allWebhooks, disabledByDefaultWebhooks []string) {
	fs.StringSliceVar(&o.Webhooks, "webhooks", o.Webhooks, fmt.Sprintf(""+
		"A list of webhooks to enable. '*' enables all on-by-default webhooks, 'foo' enables the webhook "+
		"named 'foo', '-foo' disables the webhook named 'foo'.\nAll webhooks: %s\nDisabled-by-default webhooks: %s",
		strings.Join(allWebhooks, ", "), strings.Join(disabledByDefaultWebhooks, ", ")))
	fs.StringVar(&o.ValidatingWebhookConfigFilePath, "validating-webhook-config-file", o.ValidatingWebhookConfigFilePath,
		"Path to a kubeconfig formatted file that defines the validating webhook configuration.")
	fs.StringVar(&o.MutatingWebhookConfigFilePath, "mutating-webhook-config-file", o.MutatingWebhookConfigFilePath,
		"Path to a kubeconfig formatted file that defines the mutating webhook configuration.")
}

func (o *WebhookOptions) Validate(validatingWebhooks, mutatingWebhooks, disabledByDefaultWebhooks []string) []error {
	allErrors := []error{}

	validatingWebhooksSet := sets.NewString(validatingWebhooks...)
	mutatingWebhookSet := sets.NewString(mutatingWebhooks...)
	toValidate := sets.NewString(o.Webhooks...)
	toValidate.Insert(disabledByDefaultWebhooks...)
	for _, webhook := range toValidate.List() {
		if webhook == "*" {
			continue
		}
		webhook = strings.TrimPrefix(webhook, "-")
		if !validatingWebhooksSet.Has(webhook) && !mutatingWebhookSet.Has(webhook) {
			allErrors = append(allErrors, fmt.Errorf("%q is not in the list of known webhooks", webhook))
		}
	}
	enabledValidationWebhooks := o.getEnabledWebhooks(validatingWebhooks, disabledByDefaultWebhooks)
	if len(enabledValidationWebhooks) > 0 && o.ValidatingWebhookConfigFilePath == "" {
		allErrors = append(allErrors, fmt.Errorf("webhooks %v are enabled but the validating webhook configuration path is empty", enabledValidationWebhooks))
	}
	if o.ValidatingWebhookConfiguration != nil {
		if o.ValidatingWebhookConfiguration.Name == "" {
			allErrors = append(allErrors, errors.New("validating webhook configuration name can't be empty"))
		}
		webhookConfigs := sets.New[string]()
		for _, webhookConfig := range o.ValidatingWebhookConfiguration.Webhooks {
			webhookConfigs.Insert(webhookConfig.Name)
		}
		allErrors = append(allErrors, o.validateWebhookConfiguration(webhookConfigs, enabledValidationWebhooks)...)
	}
	enabledMutatingWebhooks := o.getEnabledWebhooks(mutatingWebhooks, disabledByDefaultWebhooks)
	if len(enabledMutatingWebhooks) > 0 && o.MutatingWebhookConfigFilePath == "" {
		allErrors = append(allErrors, fmt.Errorf("webhooks %v are enabled but the mutating webhook configuration path is empty", enabledMutatingWebhooks))
	}
	if o.MutatingWebhookConfiguration != nil {
		if o.MutatingWebhookConfiguration.Name == "" {
			allErrors = append(allErrors, errors.New("mutating webhook configuration name can't be empty"))
		}
		webhookConfigs := sets.New[string]()
		for _, webhookConfig := range o.MutatingWebhookConfiguration.Webhooks {
			webhookConfigs.Insert(webhookConfig.Name)
		}
		allErrors = append(allErrors, o.validateWebhookConfiguration(webhookConfigs, enabledMutatingWebhooks)...)
	}

	return allErrors
}

func (o *WebhookOptions) getEnabledWebhooks(webhooks, disabledByDefaultWebhooks []string) []string {
	enabledWebhooks := []string{}
	for _, name := range webhooks {
		if genericcontrollermanager.IsControllerEnabled(name, sets.NewString(disabledByDefaultWebhooks...), o.Webhooks) {
			enabledWebhooks = append(enabledWebhooks, name)
		}
	}
	return enabledWebhooks
}

func (o *WebhookOptions) validateWebhookConfiguration(webhookConfigs sets.Set[string], webhooks []string) []error {
	allErrors := []error{}
	for _, name := range webhooks {
		if !webhookConfigs.Has(name) {
			allErrors = append(allErrors, fmt.Errorf("webhook %s is enabled but is not present in the webhook configuration", name))
		} else {
			webhookConfigs.Delete(name)
		}
	}
	if webhookConfigs.Len() != 0 {
		allErrors = append(allErrors, fmt.Errorf("webhook configuration is present for webhooks %v but the webhooks are not present/disabled", webhookConfigs))
	}
	return allErrors
}

func (o *WebhookOptions) ApplyTo(cfg *config.WebhookConfiguration) error {
	if o == nil {
		return nil
	}
	cfg.Webhooks = o.Webhooks

	if o.ValidatingWebhookConfigFilePath != "" {
		config := &admissionregistrationv1.ValidatingWebhookConfiguration{}
		err := LoadConfigurationFromFile(o.ValidatingWebhookConfigFilePath, config)
		if err != nil {
			return fmt.Errorf("%v: from file %v", err.Error(), o.ValidatingWebhookConfigFilePath)
		}
		cfg.ValidatingWebhookConfiguration = config
		o.ValidatingWebhookConfiguration = config
	}

	if o.MutatingWebhookConfigFilePath != "" {
		config := &admissionregistrationv1.MutatingWebhookConfiguration{}
		err := LoadConfigurationFromFile(o.MutatingWebhookConfigFilePath, config)
		if err != nil {
			return fmt.Errorf("%v: from file %v", err.Error(), o.ValidatingWebhookConfigFilePath)
		}
		cfg.MutatingWebhookConfiguration = config
		o.MutatingWebhookConfiguration = config
	}

	return nil
}

func LoadConfigurationFromFile(path string, config runtime.Object) error {
	configDef, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file path %q: %w", path, err)
	}

	decoder := serializer.NewCodecFactory(runtime.NewScheme()).UniversalDecoder(admissionregistrationv1.SchemeGroupVersion)
	_, gvk, err := decoder.Decode(configDef, nil, config)
	if err != nil {
		return fmt.Errorf("failed decoding validating webhook configuration: %w", err)
	}

	if gvk.Group != admissionregistrationv1.SchemeGroupVersion.Group || gvk.Version != admissionregistrationv1.SchemeGroupVersion.Version {
		return fmt.Errorf("unknown group version field %v in validating webhook configuration", gvk)
	}
	klog.V(4).Infoln("Load validation webhook configuration success")
	return nil
}

type WebhookServingOptions struct {
	CACertFile string
	*apiserveroptions.SecureServingOptions
}

func NewWebhookServingOptions(defaults ProviderDefaults) *WebhookServingOptions {
	var (
		bindAddress net.IP
		bindPort    int
	)

	if defaults.WebhookBindAddress != nil {
		bindAddress = *defaults.WebhookBindAddress
	} else {
		bindAddress = netutils.ParseIPSloppy("0.0.0.0")
	}

	if defaults.WebhookBindPort != nil {
		bindPort = *defaults.WebhookBindPort
	} else {
		bindPort = CloudControllerManagerWebhookPort
	}

	return &WebhookServingOptions{
		SecureServingOptions: &apiserveroptions.SecureServingOptions{
			BindAddress: bindAddress,
			BindPort:    bindPort,
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "",
				PairName:      "cloud-controller-manager-webhook",
			},
		},
	}
}

func (o *WebhookServingOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&o.BindAddress, "webhook-bind-address", o.BindAddress, ""+
		"The IP address on which to listen for the --webhook-secure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		fmt.Sprintf("clients. If set to an unspecified address (0.0.0.0 or ::), all interfaces will be used. If unset, defaults to %v.", o.BindAddress))

	fs.IntVar(&o.BindPort, "webhook-secure-port", o.BindPort, "Secure port to serve cloud provider webhooks. If 0, don't serve webhooks at all.")

	fs.StringVar(&o.ServerCert.CertDirectory, "webhook-cert-dir", o.ServerCert.CertDirectory, ""+
		"The directory where the TLS certs are located. "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")

	fs.StringVar(&o.ServerCert.CertKey.CertFile, "webhook-tls-cert-file", o.ServerCert.CertKey.CertFile, ""+
		"File containing the default x509 Certificate for HTTPS. (CA cert, if any, concatenated "+
		"after server cert). If HTTPS serving is enabled, and --tls-cert-file and "+
		"--tls-private-key-file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to the directory specified by --cert-dir.")

	fs.StringVar(&o.ServerCert.CertKey.KeyFile, "webhook-tls-private-key-file", o.ServerCert.CertKey.KeyFile,
		"File containing the default x509 private key matching --tls-cert-file.")

	fs.StringVar(&o.CACertFile, "webhook-ca-cert-file", o.CACertFile, ""+
		"File containing the root CA Certificate  matching --tls-cert-file ")
}

func (o *WebhookServingOptions) Validate() []error {
	allErrors := []error{}
	if o.BindPort < 0 || o.BindPort > 65535 {
		allErrors = append(allErrors, fmt.Errorf("--webhook-secure-port %v must be between 0 and 65535, inclusive. A value of 0 disables the webhook endpoint entirely.", o.BindPort))
	}

	if (len(o.ServerCert.CertKey.CertFile) != 0 || len(o.ServerCert.CertKey.KeyFile) != 0) && o.ServerCert.GeneratedCert != nil {
		allErrors = append(allErrors, fmt.Errorf("cert/key file and in-memory certificate cannot both be set"))
	}

	if (len(o.ServerCert.CertKey.CertFile) != 0 || len(o.ServerCert.CertKey.KeyFile) != 0) && len(o.CACertFile) == 0 {
		allErrors = append(allErrors, fmt.Errorf("ca file needed when cert/key file are provided"))
	}

	return allErrors
}

func (o *WebhookServingOptions) ApplyTo(webhookCfg *config.WebhookConfiguration, cfg **server.SecureServingInfo) error {
	if o == nil {
		return nil
	}

	if o.BindPort <= 0 {
		return nil
	}

	webhookCfg.WebhookAddress = o.BindAddress.String()
	if webhookCfg.WebhookAddress == "0.0.0.0" {
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return fmt.Errorf("failed to get host ip %w", err)
		}
		webhookCfg.WebhookAddress = ip.String()
	}
	webhookCfg.WebhookPort = int32(o.BindPort)

	var err error
	var listener net.Listener
	addr := net.JoinHostPort(o.BindAddress.String(), strconv.Itoa(o.BindPort))

	l := net.ListenConfig{}

	listener, o.BindPort, err = createListener(addr, l)
	if err != nil {
		return fmt.Errorf("failed to create listener: %v", err)
	}

	*cfg = &server.SecureServingInfo{
		Listener: listener,
	}

	serverCertFile, serverKeyFile := o.ServerCert.CertKey.CertFile, o.ServerCert.CertKey.KeyFile
	if len(serverCertFile) != 0 || len(serverKeyFile) != 0 {
		var err error
		(*cfg).Cert, err = dynamiccertificates.NewDynamicServingContentFromFiles("serving-cert", serverCertFile, serverKeyFile)
		if err != nil {
			return err
		}
		caCert, err := os.ReadFile(o.CACertFile)
		if err != nil {
			return err
		}
		webhookCfg.CaBundle = string(caCert)
	} else {
		if err := o.MaybeDefaultWithSelfSignedCerts(webhookCfg.WebhookAddress, nil, []net.IP{netutils.ParseIPSloppy("127.0.0.1")}); err != nil {
			return fmt.Errorf("error creating self-signed certificates for webhook: %w", err)
		}
		(*cfg).Cert = o.ServerCert.GeneratedCert

		cert, _ := o.ServerCert.GeneratedCert.CurrentCertKeyContent()
		certs, err := certutil.ParseCertsPEM(cert)
		if err != nil {
			return fmt.Errorf("error parsing the certs %w", err)
		}
		if len(certs) < 2 {
			return fmt.Errorf("generated cert doesn't have the root cert, has only %v certs", len(certs))
		}
		caPEM := new(bytes.Buffer)
		err = pem.Encode(caPEM, &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: certs[1].Raw,
		})
		if err != nil {
			return fmt.Errorf("error encoding ca cert %w", err)
		}
		webhookCfg.CaBundle = caPEM.String()
	}

	return nil
}

func createListener(addr string, config net.ListenConfig) (net.Listener, int, error) {
	ln, err := config.Listen(context.TODO(), "tcp", addr)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to listen on %v: %v", addr, err)
	}

	// get port
	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		ln.Close()
		return nil, 0, fmt.Errorf("invalid listen address: %q", ln.Addr().String())
	}

	return ln, tcpAddr.Port, nil
}
