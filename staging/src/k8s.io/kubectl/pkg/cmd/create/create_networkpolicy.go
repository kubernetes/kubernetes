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

package create

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/spf13/cobra"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type NetworkRule string

const (
	_Ports     NetworkRule = "ports"
	_Pod       NetworkRule = "pod"
	_Namespace NetworkRule = "namespace"
)

func (r NetworkRule) IsValid() bool {
	switch r {
	case _Ports, _Pod, _Namespace:
		return true
	}
	return false
}

var (
	// Explaining the Regex below:
	// (?P<rule>[a-z]+?) -> Rules: ports, pod (podSelector) or namespace (namespaceSelector)
	// Separator from rule to key:value -> "="
	// (?P<key>.*?) -> Network Protocol type (udp, tcp, sctp) or matchLabels key
	// Separator from key to value -> ":"
	// (?P<value>[\w\-]+) -> Network Protocol Port or matchLabels value
	regexRuleKeyValue = `(?P<rule>[a-z]+?)=(?P<key>.*?):(?P<value>[\w\-]+)`

	regexRuleName = `(\w+?)=`

	networkPolicyLong = templates.LongDesc(i18n.T(`
	Create a network policy with the specified name.`))

	networkPolicyExample = templates.Examples(i18n.T(`
		# Create a network policy for a namespace which prevents all ingress AND egress traffic by 
		# creating the following NetworkPolicy in that namespace.
		kubectl create networkpolicies simple

		# Create a network policy for a namespace that prevents outbound traffic.
		kubectl create networkpolicies policy-np --policy-types=egress

		# Create a network policy includes a podSelector which selects the grouping of pods to which the policy applies.
		kubectl create networkpolicies podselector --pod-selector=app=nginx

		# Create AND rules. If multiple rules are specified, they are connected using a logical AND (Ports AND From/To).
		kubectl create networkpolicies and-ingress-np --ingress=ports=udp:53,tcp:53,pod=app:nginx,namespace=kubernetes.io/metadata.name:default

		# Create OR rules. If multiple rules are specified, they are connected using a logical OR (Ports OR From/To).
		kubectl create networkpolicies or-ingress-np --ingress=ports=udp:53,tcp:53 --ingress=pod=app:nginx --ingress=namespace=kubernetes.io/metadata.name:default

		# Create a network policy with multiple rules, combine logical OR/AND.
		kubectl create networkpolicies multirules --pod-selector app=nginx \
		--ingress ports=udp:53,tcp:53,pod=app:nginx,app:busybox,namespace=kubernetes.io/metadata.name:default  \
		--egress ports=udp:53,pod=app:httpd:2.4-alpine \
		--egress ports=sctp:53,pod=app:busybox

		`))
)

// CreateNetworkPolicyOptions is returned by NewCmdCreateNetworkPolicy
type CreateNetworkPolicyOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name         string
	PodSelector  string
	PolicyTypes  []string
	IngressRules []string
	EgressRules  []string

	Namespace        string
	EnforceNamespace bool

	DryRunStrategy      cmdutil.DryRunStrategy
	DryRunVerifier      *resource.QueryParamVerifier
	ValidationDirective string

	FieldManager     string
	CreateAnnotation bool

	genericclioptions.IOStreams
}

// NewCreateNetworkPolicyOptions creates the CreateNetworkPolicyOptions to be used later
func NewCreateNetworkPolicyOptions(ioStreams genericclioptions.IOStreams) *CreateNetworkPolicyOptions {
	return &CreateNetworkPolicyOptions{
		PrintFlags:   genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:    ioStreams,
		PolicyTypes:  []string{"ingress", "egress"},
		IngressRules: []string{},
		EgressRules:  []string{},
	}
}

// NewCmdCreateNetworkPolicy is a macro command to create a new network policy.
// This command is better known to users as `kubectl create networkpolicies`.
func NewCmdCreateNetworkPolicy(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateNetworkPolicyOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "networkpolicies NAME",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"networkpolicy", "netpol"},
		Short:                 i18n.T("Create a network policy"),
		Long:                  networkPolicyLong,
		Example:               networkPolicyExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.PodSelector, "pod-selector", o.PodSelector, "Each NetworkPolicy includes a podSelector which selects the grouping of pods to which the policy applies.")
	cmd.Flags().StringArrayVar(&o.PolicyTypes, "policy-types", o.PolicyTypes, `Each NetworkPolicy includes a policyTypes list which may include "none", "ingress" or "egress". This argument is optional.`)
	cmd.Flags().StringArrayVar(&o.IngressRules, "ingress", o.IngressRules, "Specify multiple ingress rules (ports, pod=podSelector, namespace=namespaceSelector).")
	cmd.Flags().StringArrayVar(&o.EgressRules, "egress", o.EgressRules, "Specify multiple egress rules (ports, pod=podSelector, namespace=namespaceSelector).")

	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")

	return cmd
}

// Complete completes all the options
func (o *CreateNetworkPolicyOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	o.Name = name

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewQueryParamVerifier(dynamicClient, f.OpenAPIGetter(), resource.QueryParamDryRun)
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	return err
}

// Validate checks to the CreateNetworkPolicyOptions to see if there is sufficient information run the command
func (o *CreateNetworkPolicyOptions) Validate() error {
	_, err := metav1.ParseToLabelSelector(o.PodSelector)
	if err != nil {
		return err
	}

	for _, ptype := range o.PolicyTypes {
		switch ptype {
		case "ingress":
		case "egress":
		case "none":
			continue
		default:
			return fmt.Errorf(`invalid policy-types value (%v). Must be "none", "ingress", or "egress"`, ptype)
		}
	}

	rulevalidation, err := regexp.Compile(regexRuleKeyValue)
	if err != nil {
		return fmt.Errorf("failed to compile the regex")
	}

	re := regexp.MustCompile(regexRuleName)

	for _, rule := range o.IngressRules {
		if match := rulevalidation.MatchString(rule); !match {
			return fmt.Errorf("ingress rule (%s) is invalid and should be in format rule=key:value", rule)
		}

		names := re.FindAllStringSubmatch(rule, -1)
		for _, name := range names {
			if !NetworkRule(name[1]).IsValid() {
				return fmt.Errorf("invalid ingress rule name (%s)", name[1])
			}
		}
	}

	for _, rule := range o.EgressRules {
		if match := rulevalidation.MatchString(rule); !match {
			return fmt.Errorf("egress rule (%s) is invalid and should be in format rule=key:value", rule)
		}

		names := re.FindAllStringSubmatch(rule, -1)
		for _, name := range names {
			if !NetworkRule(name[1]).IsValid() {
				return fmt.Errorf("invalid egress rule name (%s)", name[1])
			}
		}
	}

	return nil
}

// Run performs the execution of 'create networkpolicies' sub command
func (o *CreateNetworkPolicyOptions) Run() error {
	networkpolicy, err := o.createNetworkPolicy()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, networkpolicy, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(networkpolicy.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
	}
	return o.PrintObj(networkpolicy)
}

func (o *CreateNetworkPolicyOptions) createNetworkPolicy() (*networkingv1.NetworkPolicy, error) {
	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	spec, err := o.buildNetworkPolicySpec()
	if err != nil {
		return nil, err
	}

	networkPolicy := &networkingv1.NetworkPolicy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: networkingv1.SchemeGroupVersion.String(),
			Kind:       "NetworkPolicy",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Namespace: namespace,
		},
		Spec: *spec,
	}
	return networkPolicy, nil
}

// buildNetworkPolicySpec builds the .spec from the diverse arguments passed to kubectl
func (o *CreateNetworkPolicyOptions) buildNetworkPolicySpec() (*networkingv1.NetworkPolicySpec, error) {
	var networkPolicySpec networkingv1.NetworkPolicySpec

	policyTypes := []networkingv1.PolicyType{}
	for _, ptype := range o.PolicyTypes {
		if strings.EqualFold(ptype, string(networkingv1.PolicyTypeIngress)) {
			policyTypes = append(policyTypes, networkingv1.PolicyTypeIngress)
		} else if strings.EqualFold(ptype, string(networkingv1.PolicyTypeEgress)) {
			policyTypes = append(policyTypes, networkingv1.PolicyTypeEgress)
		}
	}

	networkPolicySpec.PolicyTypes = policyTypes

	selector, err := metav1.ParseToLabelSelector(o.PodSelector)
	if err != nil {
		return nil, err
	}

	networkPolicySpec.PodSelector = *selector

	networkPolicySpec.Ingress = buildNetworkPolicyIngressRule(o.IngressRules)
	networkPolicySpec.Egress = buildNetworkPolicyEgressRule(o.EgressRules)

	return &networkPolicySpec, nil
}

// buildNetworkPolicyIngressRule builds the .spec.ingress for an network policy object.
// --ingress=ports=udp:53,tcp:53,pod=app:nginx,namespace=kubernetes.io/metadata.name:default
func buildNetworkPolicyIngressRule(rules []string) []networkingv1.NetworkPolicyIngressRule {
	networkPolicyRules := []networkingv1.NetworkPolicyIngressRule{}
	for _, rule := range rules {
		networkRule := networkingv1.NetworkPolicyIngressRule{}
		networkRule.Ports, networkRule.From = buildNetworkPolicyRule(rule)
		networkPolicyRules = append(networkPolicyRules, networkRule)
	}
	return networkPolicyRules
}

// buildNetworkPolicyEgressRule builds the .spec.egress for an network policy object.
// --egress=ports=udp:53,tcp:53,pod=app:nginx,namespace=kubernetes.io/metadata.name:default
func buildNetworkPolicyEgressRule(rules []string) []networkingv1.NetworkPolicyEgressRule {
	networkPolicyRules := []networkingv1.NetworkPolicyEgressRule{}
	for _, rule := range rules {
		networkRule := networkingv1.NetworkPolicyEgressRule{}
		networkRule.Ports, networkRule.To = buildNetworkPolicyRule(rule)
		networkPolicyRules = append(networkPolicyRules, networkRule)
	}
	return networkPolicyRules
}

func buildNetworkPolicyRule(rules string) ([]networkingv1.NetworkPolicyPort, []networkingv1.NetworkPolicyPeer) {
	var ports []networkingv1.NetworkPolicyPort
	var peers []networkingv1.NetworkPolicyPeer

	re := regexp.MustCompile(regexRuleName)
	indexes := re.FindAllStringIndex(rules, -1)
	endIndex := len(rules)
	for i := len(indexes) - 1; i >= 0; i-- {
		beginIndex := indexes[i][0]
		key := NetworkRule(rules[beginIndex : indexes[i][1]-1])
		values := rules[indexes[i][1]:endIndex]
		endIndex = beginIndex - 1

		switch key {
		case _Ports:
			ports = append(ports, buildNetworkPolicyPort(values)...)
		case _Pod:
			peers = append(peers, buildPodNetworkPolicyPeer(values)...)
		case _Namespace:
			peers = append(peers, buildNamespaceNetworkPolicyPeer(values)...)
		}
	}

	return ports, peers
}

func buildNetworkPolicyPort(values string) []networkingv1.NetworkPolicyPort {
	ports := []networkingv1.NetworkPolicyPort{}
	parts := strings.Split(values, ",")

	for _, part := range parts {
		key_val := strings.SplitN(part, ":", 2)
		port := intstr.Parse(key_val[1])
		protocol := v1.ProtocolSCTP
		if strings.EqualFold(key_val[0], string(v1.ProtocolUDP)) {
			protocol = v1.ProtocolUDP
		} else if strings.EqualFold(key_val[0], string(v1.ProtocolTCP)) {
			protocol = v1.ProtocolTCP
		}

		ports = append(ports, networkingv1.NetworkPolicyPort{
			Protocol: &protocol,
			Port:     &port,
		})
	}
	return ports
}

func buildPodNetworkPolicyPeer(values string) []networkingv1.NetworkPolicyPeer {
	peers := []networkingv1.NetworkPolicyPeer{}
	parts := strings.Split(values, ",")

	for _, part := range parts {
		key_val := strings.SplitN(part, ":", 2)

		peers = append(peers, networkingv1.NetworkPolicyPeer{
			PodSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{key_val[0]: key_val[1]},
			},
		})
	}
	return peers
}

func buildNamespaceNetworkPolicyPeer(values string) []networkingv1.NetworkPolicyPeer {
	peers := []networkingv1.NetworkPolicyPeer{}
	parts := strings.Split(values, ",")

	for _, part := range parts {
		key_val := strings.SplitN(part, ":", 2)

		peers = append(peers, networkingv1.NetworkPolicyPeer{
			NamespaceSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{key_val[0]: key_val[1]},
			},
		})
	}
	return peers
}
