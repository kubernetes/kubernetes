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

package cmd

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	pspv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	clientgo "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	PSPLong = templates.LongDesc(i18n.T(`
		Create a PodSecurityPolicy.`))

	pspExample = templates.Examples(i18n.T(`

		# Create a PodSecurityPolicy named "example" that simply prevents the creation of privileged pods.
		kubectl create podsecuritypolicy example --privileged=false --selinux=RunAsAny --supplemental-groups=RunAsAny
		--run-as-user=RunAsAny --fs-group=RunAsAny

		# Create a PodSecurityPolicy named "privileged"
		kubectl create podsecuritypolicy privileged --privileged=true --allowed-cap=* --host-ports=0-65535
		--selinux=RunAsAny --supplemental-groups=RunAsAny --run-as-user=RunAsAny --fs-group=RunAsAny

		# Create a PodSecurityPolicy named "my-psp"
		kubectl create podsecuritypolicy my-psp --selinux="MustRunAs,user=u,role=r,type=t,level=l" --supplemental-groups=RunAsAny
		--run-as-user=RunAsAny --fs-group=RunAsAny

		# Create a PodSecurityPolicy named "run-as-user-example" that allow user ID 10-100 containers run as.
		kubectl create podsecuritypolicy run-as-user-example --selinux=RunAsAny --supplemental-groups=RunAsAny
		--fs-group=RunAsAny --run-as-user="MustRunAs,10-100"
		`))
)

type CreatePSPOptions struct {
	Name                            string
	Privileged                      bool
	DefaultAddCapabilities          []string
	RequiredDropCapabilities        []string
	AllowedCapabilities             []string
	Volumes                         []string
	HostNetwork                     bool
	HostPorts                       []pspv1beta1.HostPortRange
	HostPID                         bool
	HostIPC                         bool
	SELinux                         pspv1beta1.SELinuxStrategyOptions //required
	RunAsUser                       strategyOptions                   //required
	SupplementalGroups              strategyOptions                   //required
	FSGroup                         strategyOptions                   //required
	ReadOnlyRootFilesystem          bool
	DefaultAllowPrivilegeEscalation *bool
	AllowPrivilegeEscalation        *bool
	AllowedHostPaths                []string
	AllowedFlexVolumes              []string

	DryRun       bool
	OutputFormat string
	Client       clientgo.ExtensionsV1beta1Interface
	Mapper       meta.RESTMapper
	Out          io.Writer
	PrintObject  func(obj runtime.Object) error
	PrintSuccess func(mapper meta.RESTMapper, shortOutput bool, out io.Writer, resource, name string, dryRun bool, operation string)
}

func NewCmdCreatePSP(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreatePSPOptions{
		Out: cmdOut,
	}
	cmd := &cobra.Command{
		Use: "podsecuritypolicy NAME --selinux=rule --supplemental-groups=rule --run-as-user=rule --fs-group=rule [other options] [--dry-run]",
		DisableFlagsInUseLine: true,
		Short:   PSPLong,
		Long:    PSPLong,
		Example: pspExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.Validate())
			cmdutil.CheckErr(c.RunCreatePSP())
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().BoolVar(&c.Privileged, "privileged", false, "Wether privileged containers can be created")
	cmd.Flags().StringSliceVar(&c.DefaultAddCapabilities, "default-add-cap", nil, "The capabilities which are added to containers by default, in addition to the runtime defaults")
	cmd.Flags().StringSliceVar(&c.RequiredDropCapabilities, "required-drop-cap", nil, "The capabilities which must be dropped from containers. These capabilities are removed from the default set, and must not be added. Capabilities listed in required-drop-capabilities must not be included in allowed-capabilities or default-add-capabilities")
	cmd.Flags().StringSliceVar(&c.AllowedCapabilities, "allowed-cap", nil, "Provides a whitelist of capabilities that may be added to a container. The default set of capabilities are implicitly allowed. The empty set means that no additional capabilities may be added beyond the default set. * can be used to allow all capabilities.")
	cmd.Flags().StringSliceVar(&c.Volumes, "volumes", nil, "Provides a whitelist of allowed volume types. The allowable values correspond to the volume sources that are defined when creating a volume")
	cmd.Flags().BoolVar(&c.HostNetwork, "host-network", false, "Controls whether the pod may use the node network namespace. Doing so gives the pod access to the loopback device, services listening on localhost, and could be used to snoop on network activity of other pods on the same node.")
	cmd.Flags().StringSlice("host-ports", nil, "Provides a whitelist of ranges of allowable ports in the host network namespace. ")
	cmd.Flags().BoolVar(&c.HostPID, "host-pid", false, "Controls whether the pod containers can share the host process ID namespace.Defined as a list of HostPortRange, with min(inclusive) and max(inclusive).Defaults to no allowed host port.The port range must be in the format of 2000-3000")
	cmd.Flags().BoolVar(&c.HostIPC, "host-ipc", false, "Controls whether the pod containers can share the host IPC namespace.")
	cmd.Flags().StringSlice("selinux", nil, "Required option which will dictate the allowable labels that may be set")
	cmd.Flags().StringSlice("run-as-user", nil, "Required option which controls the what user ID containers run as.")
	cmd.Flags().StringSlice("supplemental-groups", nil, "Required option which controls which group IDs containers add.")
	cmd.Flags().StringSlice("fs-group", nil, "Required option which provides a whitelist of allowed volume types. The allowable values correspond to the volume sources that are defined when creating a volume.")
	cmd.Flags().BoolVar(&c.ReadOnlyRootFilesystem, "readonly-root-fs", false, "Requires that containers must run with a read-only root filesystem (i.e. no writeable layer)")
	cmd.Flags().Bool("default-allow-privilege-escalation", false, "Sets the default for the allow-privilege-escalation option. The default behavior without this is to allow privilege escalation so as to not break setuid binaries. If that behavior is not desired, this field can be used to default to disallow, while still permitting pods to request allow-privilege-escalation explicitly.")
	cmd.Flags().Bool("allow-privilege-escalation", false, "Gates whether or not a user is allowed to set the security context of a container to allowPrivilegeEscalation=true. This defaults to allowed. When set to false, the container’s allowPrivilegeEscalation is defaulted to false.")
	cmd.Flags().StringSliceVar(&c.AllowedHostPaths, "allowed-host-paths", nil, "This specifies a whitelist of host paths that are allowed to be used by hostPath volumes. An empty list means there is no restriction on host paths used")
	cmd.Flags().StringSliceVar(&c.AllowedFlexVolumes, "allowed-flex-volumes", nil, "Provides a whitelist of allowed FlexVolumes. Empty or nil indicates that all FlexVolume drivers may be used")

	return cmd
}

func (c *CreatePSPOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {

	if name, err := NameFromCommandArgs(cmd, args); err == nil {
		c.Name = name
	} else {
		return err
	}

	if portRange, err := newHostRangePort(cmdutil.GetFlagStringSlice(cmd, "host-ports")); err == nil {
		c.HostPorts = portRange
	} else {
		return err
	}

	if selinux, err := newSELinuxOptions(cmdutil.GetFlagStringSlice(cmd, "selinux")); err == nil {
		c.SELinux = selinux
	} else {
		return err
	}

	if stra, err := newStrategyOptions("run-as-user", cmdutil.GetFlagStringSlice(cmd, "run-as-user")); err == nil {
		c.RunAsUser = stra
	} else {
		return err
	}

	if stra, err := newStrategyOptions("supplemental-groups", cmdutil.GetFlagStringSlice(cmd, "supplemental-groups")); err == nil {
		c.SupplementalGroups = stra
	} else {
		return err
	}

	if stra, err := newStrategyOptions("fs-group", cmdutil.GetFlagStringSlice(cmd, "fs-group")); err == nil {
		c.FSGroup = stra
	} else {
		return err
	}

	if cmd.Flags().Changed("default-allow-privilege-escalation") {
		if flag, err := cmd.Flags().GetBool("default-allow-privilege-escalation"); err == nil {
			c.DefaultAllowPrivilegeEscalation = &flag
		} else {
			return err
		}
	}

	if cmd.Flags().Changed("allow-privilege-escalation") {
		if flag, err := cmd.Flags().GetBool("allow-privilege-escalation"); err == nil {
			c.AllowPrivilegeEscalation = &flag
		} else {
			return err
		}
	}

	// Complete other options for Run.
	c.Mapper, _ = f.Object()

	c.DryRun = cmdutil.GetDryRunFlag(cmd)
	c.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	c.PrintObject = func(obj runtime.Object) error {
		return f.PrintObject(cmd, false, c.Mapper, obj, c.Out)
	}
	c.PrintSuccess = f.PrintSuccess
	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	c.Client = clientset.ExtensionsV1beta1()

	return nil
}

func (c *CreatePSPOptions) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}
	if c.SELinux.Rule != "RunAsAny" && c.SELinux.Rule != "MustRunAs" {
		return fmt.Errorf("illegal --selinux rule, must be one of RunAsAny MustRunAs")
	}
	if c.RunAsUser.strategyRule != "RunAsAny" && c.RunAsUser.strategyRule != "MustRunAs" && c.RunAsUser.strategyRule != "MustRunAsNonRoot" {
		return fmt.Errorf("illegal --run-as-user rule, must be one of RunAsAny MustRunAs MustRunAsNonRoot")
	}
	if c.SupplementalGroups.strategyRule != "RunAsAny" && c.SupplementalGroups.strategyRule != "MustRunAs" {
		return fmt.Errorf("illegal --supplemental-groups rule, must be one of RunAsAny MustRunAs")
	}
	if c.FSGroup.strategyRule != "RunAsAny" && c.FSGroup.strategyRule != "MustRunAs" {
		return fmt.Errorf("illegal --fs-group rule, must be one of RunAsAny MustRunAs")
	}
	return nil
}

func (c *CreatePSPOptions) RunCreatePSP() error {
	psp := &pspv1beta1.PodSecurityPolicy{}
	psp.Name = c.Name
	psp.Spec.Privileged = c.Privileged

	psp.Spec.DefaultAddCapabilities = convertToCapability(c.DefaultAddCapabilities)
	psp.Spec.RequiredDropCapabilities = convertToCapability(c.RequiredDropCapabilities)
	psp.Spec.AllowedCapabilities = convertToCapability(c.AllowedCapabilities)
	psp.Spec.Volumes = convertToFSType(c.Volumes)
	psp.Spec.HostNetwork = c.HostNetwork
	psp.Spec.HostPorts = c.HostPorts
	psp.Spec.HostPID = c.HostPID
	psp.Spec.HostIPC = c.HostIPC
	psp.Spec.SELinux = c.SELinux
	psp.Spec.RunAsUser = pspv1beta1.RunAsUserStrategyOptions{
		Rule:   pspv1beta1.RunAsUserStrategy(c.RunAsUser.strategyRule),
		Ranges: c.RunAsUser.ranges,
	}
	psp.Spec.SupplementalGroups = pspv1beta1.SupplementalGroupsStrategyOptions{
		Rule:   pspv1beta1.SupplementalGroupsStrategyType(c.SupplementalGroups.strategyRule),
		Ranges: c.SupplementalGroups.ranges,
	}
	psp.Spec.FSGroup = pspv1beta1.FSGroupStrategyOptions{
		Rule:   pspv1beta1.FSGroupStrategyType(c.FSGroup.strategyRule),
		Ranges: c.FSGroup.ranges,
	}
	psp.Spec.ReadOnlyRootFilesystem = c.ReadOnlyRootFilesystem

	psp.Spec.DefaultAllowPrivilegeEscalation = c.DefaultAllowPrivilegeEscalation
	psp.Spec.AllowPrivilegeEscalation = c.AllowPrivilegeEscalation
	psp.Spec.AllowedHostPaths = convertToAllowedHostPath(c.AllowedHostPaths)
	psp.Spec.AllowedFlexVolumes = convertToAllowedFlexVolume(c.AllowedFlexVolumes)

	if !c.DryRun {
		_, err := c.Client.PodSecurityPolicies().Create(psp)
		if err != nil {
			return err
		}
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		c.PrintSuccess(c.Mapper, useShortOutput, c.Out, "podsecuritypolicies", c.Name, c.DryRun, "created")
		return nil
	}

	return c.PrintObject(psp)
}

func newRangeRule(rules []string) ([]pspv1beta1.IDRange, error) {
	if len(rules) == 0 {
		return nil, nil
	}
	ret := []pspv1beta1.IDRange{}

	for _, rule := range rules {
		if len(rule) == 0 {
			continue
		}
		split := strings.Split(rule, "-")
		if len(split) != 2 {
			return nil, fmt.Errorf("range rule must be in format like 2000-3000")
		}

		rr := pspv1beta1.IDRange{}
		if min, err := strconv.ParseInt(split[0], 10, 64); err == nil {
			rr.Min = min
		} else {
			return nil, err
		}
		if max, err := strconv.ParseInt(split[1], 10, 64); err == nil {
			rr.Max = max
		} else {
			return nil, err
		}
		ret = append(ret, rr)
	}
	return ret, nil
}

func newHostRangePort(rules []string) ([]pspv1beta1.HostPortRange, error) {
	if len(rules) == 0 {
		return nil, nil
	}

	ret := []pspv1beta1.HostPortRange{}

	for _, rule := range rules {
		if len(rule) == 0 {
			continue
		}
		split := strings.Split(rule, "-")
		if len(split) != 2 {
			return nil, fmt.Errorf("range rule must be in format like 2000-3000")
		}

		//var min max int64
		rr := pspv1beta1.HostPortRange{}
		if min, err := strconv.ParseInt(split[0], 10, 32); err == nil {
			rr.Min = int32(min)
		} else {
			return nil, err
		}
		if max, err := strconv.ParseInt(split[1], 10, 32); err == nil {
			rr.Max = int32(max)
		} else {
			return nil, err
		}
		ret = append(ret, rr)
	}
	return ret, nil
}

//RunAsUser,SupplementalGroups,FSGroup share similiar structure, use
//strategyOptions as transit and generated respectively later.
type strategyOptions struct {
	ranges       []pspv1beta1.IDRange
	strategyRule string
}

func newStrategyOptions(optionName string, str []string) (strategyOptions, error) {
	if len(str) == 0 {
		//three options may use this function, all of which are required.
		//so if str is null, return error.
		return strategyOptions{}, fmt.Errorf("--%s is required", optionName)
	}
	if rangeRules, err := newRangeRule(str[1:]); err == nil {
		return strategyOptions{
			ranges:       rangeRules,
			strategyRule: str[0],
		}, nil
	} else {
		return strategyOptions{}, err
	}
}

func convertToFSType(strs []string) []pspv1beta1.FSType {
	if len(strs) == 0 {
		return nil
	}
	ret := []pspv1beta1.FSType{}
	for _, str := range strs {
		ret = append(ret, pspv1beta1.FSType(str))
	}
	return ret
}

func convertToAllowedHostPath(strs []string) []pspv1beta1.AllowedHostPath {
	if len(strs) == 0 {
		return nil
	}
	ret := []pspv1beta1.AllowedHostPath{}
	for _, str := range strs {
		ret = append(ret, pspv1beta1.AllowedHostPath{str})
	}
	return ret
}

func convertToAllowedFlexVolume(strs []string) []pspv1beta1.AllowedFlexVolume {
	if len(strs) == 0 {
		return nil
	}
	ret := []pspv1beta1.AllowedFlexVolume{}
	for _, str := range strs {
		ret = append(ret, pspv1beta1.AllowedFlexVolume{str})
	}
	return ret
}

func convertToCapability(strs []string) []v1.Capability {
	if len(strs) == 0 {
		return nil
	}
	ret := []v1.Capability{}
	for _, str := range strs {
		ret = append(ret, v1.Capability(str))
	}
	return ret
}

func newSELinuxOptions(strs []string) (pspv1beta1.SELinuxStrategyOptions, error) {
	lens := len(strs)
	if lens == 0 {
		return pspv1beta1.SELinuxStrategyOptions{}, fmt.Errorf("--selinux is required")
	}

	if strs[0] != "RunAsAny" && strs[0] != "MustRunAs" {
		return pspv1beta1.SELinuxStrategyOptions{}, fmt.Errorf("wrong --selinux format, first param must be RunAsAny or MustRunAs")
	}

	if lens == 1 {
		return pspv1beta1.SELinuxStrategyOptions{Rule: pspv1beta1.SELinuxStrategy(strs[0])}, nil
	}

	ret := pspv1beta1.SELinuxStrategyOptions{
		Rule:           pspv1beta1.SELinuxStrategy(strs[0]),
		SELinuxOptions: &v1.SELinuxOptions{},
	}
	for _, str := range strs[1:] {
		splitStr := strings.Split(str, "=")
		if len(splitStr) != 2 {
			return pspv1beta1.SELinuxStrategyOptions{}, fmt.Errorf("wrong --selinux format, must be like --selinux=\"MustRunAs,user=u,role=r,type=t,level=l\"")
		}
		key := splitStr[0]
		value := splitStr[1]
		switch key {
		case "user":
			ret.SELinuxOptions.User = value
		case "role":
			ret.SELinuxOptions.Role = value
		case "type":
			ret.SELinuxOptions.Type = value
		case "level":
			ret.SELinuxOptions.Level = value
		default:
			return pspv1beta1.SELinuxStrategyOptions{}, fmt.Errorf("wrong --selinux format, must be like --selinux=\"MustRunAs,user=u,role=r,type=t,level=l\"")
		}
	}
	return ret, nil
}
