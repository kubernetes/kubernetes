/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package vm

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type intRange struct {
	low, high int
}

var intRangeRegexp = regexp.MustCompile("^([0-9]+)-([0-9]+)$")

func (i *intRange) Set(s string) error {
	m := intRangeRegexp.FindStringSubmatch(s)
	if m == nil {
		return fmt.Errorf("invalid range: %s", s)
	}

	low, _ := strconv.Atoi(m[1])
	high, _ := strconv.Atoi(m[2])
	if low > high {
		return fmt.Errorf("invalid range: low > high")
	}

	i.low = low
	i.high = high
	return nil
}

func (i *intRange) String() string {
	return fmt.Sprintf("%d-%d", i.low, i.high)
}

type vnc struct {
	*flags.SearchFlag

	Enable    bool
	Disable   bool
	Port      int
	PortRange intRange
	Password  string
}

func init() {
	cmd := &vnc{}
	cmd.PortRange.Set("5900-5999")
	cli.Register("vm.vnc", cmd)
}

func (cmd *vnc) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.SearchFlag, ctx = flags.NewSearchFlag(ctx, flags.SearchVirtualMachines)
	cmd.SearchFlag.Register(ctx, f)

	f.BoolVar(&cmd.Enable, "enable", false, "Enable VNC")
	f.BoolVar(&cmd.Disable, "disable", false, "Disable VNC")
	f.IntVar(&cmd.Port, "port", -1, "VNC port (-1 for auto-select)")
	f.Var(&cmd.PortRange, "port-range", "VNC port auto-select range")
	f.StringVar(&cmd.Password, "password", "", "VNC password")
}

func (cmd *vnc) Process(ctx context.Context) error {
	if err := cmd.SearchFlag.Process(ctx); err != nil {
		return err
	}
	// Either may be true or none may be true.
	if cmd.Enable && cmd.Disable {
		return flag.ErrHelp
	}

	return nil
}

func (cmd *vnc) Usage() string {
	return "VM..."
}

func (cmd *vnc) Description() string {
	return `Enable or disable VNC for VM.

Port numbers are automatically chosen if not specified.

If neither -enable or -disable is specified, the current state is returned.

Examples:
  govc vm.vnc -enable -password 1234 $vm | awk '{print $2}' | xargs open`
}

func (cmd *vnc) Run(ctx context.Context, f *flag.FlagSet) error {
	vms, err := cmd.loadVMs(f.Args())
	if err != nil {
		return err
	}

	// Actuate settings in VMs
	for _, vm := range vms {
		switch {
		case cmd.Enable:
			vm.enable(cmd.Port, cmd.Password)
		case cmd.Disable:
			vm.disable()
		}
	}

	// Reconfigure VMs to reflect updates
	for _, vm := range vms {
		err = vm.reconfigure()
		if err != nil {
			return err
		}
	}

	return cmd.WriteResult(vncResult(vms))
}

func (cmd *vnc) loadVMs(args []string) ([]*vncVM, error) {
	c, err := cmd.Client()
	if err != nil {
		return nil, err
	}

	vms, err := cmd.VirtualMachines(args)
	if err != nil {
		return nil, err
	}

	var vncVMs []*vncVM
	for _, vm := range vms {
		v, err := newVNCVM(c, vm)
		if err != nil {
			return nil, err
		}
		vncVMs = append(vncVMs, v)
	}

	// Assign vncHosts to vncVMs
	hosts := make(map[string]*vncHost)
	for _, vm := range vncVMs {
		if h, ok := hosts[vm.hostReference().Value]; ok {
			vm.host = h
			continue
		}

		hs := object.NewHostSystem(c, vm.hostReference())
		h, err := newVNCHost(c, hs, cmd.PortRange.low, cmd.PortRange.high)
		if err != nil {
			return nil, err
		}

		hosts[vm.hostReference().Value] = h
		vm.host = h
	}

	return vncVMs, nil
}

type vncVM struct {
	c    *vim25.Client
	vm   *object.VirtualMachine
	mvm  mo.VirtualMachine
	host *vncHost

	curOptions vncOptions
	newOptions vncOptions
}

func newVNCVM(c *vim25.Client, vm *object.VirtualMachine) (*vncVM, error) {
	v := &vncVM{
		c:  c,
		vm: vm,
	}

	virtualMachineProperties := []string{
		"name",
		"config.extraConfig",
		"runtime.host",
	}

	pc := property.DefaultCollector(c)
	ctx := context.TODO()
	err := pc.RetrieveOne(ctx, vm.Reference(), virtualMachineProperties, &v.mvm)
	if err != nil {
		return nil, err
	}

	v.curOptions = vncOptionsFromExtraConfig(v.mvm.Config.ExtraConfig)
	v.newOptions = vncOptionsFromExtraConfig(v.mvm.Config.ExtraConfig)

	return v, nil
}

func (v *vncVM) hostReference() types.ManagedObjectReference {
	return *v.mvm.Runtime.Host
}

func (v *vncVM) enable(port int, password string) error {
	v.newOptions["enabled"] = "true"
	v.newOptions["port"] = fmt.Sprintf("%d", port)
	v.newOptions["password"] = password

	// Find port if auto-select
	if port == -1 {
		// Reuse port if If VM already has a port, reuse it.
		// Otherwise, find unused VNC port on host.
		if p, ok := v.curOptions["port"]; ok && p != "" {
			v.newOptions["port"] = p
		} else {
			port, err := v.host.popUnusedPort()
			if err != nil {
				return err
			}
			v.newOptions["port"] = fmt.Sprintf("%d", port)
		}
	}
	return nil
}

func (v *vncVM) disable() error {
	v.newOptions["enabled"] = "false"
	v.newOptions["port"] = ""
	v.newOptions["password"] = ""
	return nil
}

func (v *vncVM) reconfigure() error {
	if reflect.DeepEqual(v.curOptions, v.newOptions) {
		// No changes to settings
		return nil
	}

	spec := types.VirtualMachineConfigSpec{
		ExtraConfig: v.newOptions.ToExtraConfig(),
	}

	ctx := context.TODO()
	task, err := v.vm.Reconfigure(ctx, spec)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

func (v *vncVM) uri() (string, error) {
	ip, err := v.host.managementIP()
	if err != nil {
		return "", err
	}

	uri := fmt.Sprintf("vnc://:%s@%s:%s",
		v.newOptions["password"],
		ip,
		v.newOptions["port"])

	return uri, nil
}

func (v *vncVM) write(w io.Writer) error {
	if strings.EqualFold(v.newOptions["enabled"], "true") {
		uri, err := v.uri()
		if err != nil {
			return err
		}
		fmt.Printf("%s: %s\n", v.mvm.Name, uri)
	} else {
		fmt.Printf("%s: disabled\n", v.mvm.Name)
	}
	return nil
}

type vncHost struct {
	c     *vim25.Client
	host  *object.HostSystem
	ports map[int]struct{}
	ip    string // This field is populated by `managementIP`
}

func newVNCHost(c *vim25.Client, host *object.HostSystem, low, high int) (*vncHost, error) {
	ports := make(map[int]struct{})
	for i := low; i <= high; i++ {
		ports[i] = struct{}{}
	}

	used, err := loadUsedPorts(c, host.Reference())
	if err != nil {
		return nil, err
	}

	// Remove used ports from range
	for _, u := range used {
		delete(ports, u)
	}

	h := &vncHost{
		c:     c,
		host:  host,
		ports: ports,
	}

	return h, nil
}

func loadUsedPorts(c *vim25.Client, host types.ManagedObjectReference) ([]int, error) {
	ctx := context.TODO()
	ospec := types.ObjectSpec{
		Obj: host,
		SelectSet: []types.BaseSelectionSpec{
			&types.TraversalSpec{
				Type: "HostSystem",
				Path: "vm",
				Skip: types.NewBool(false),
			},
		},
		Skip: types.NewBool(false),
	}

	pspec := types.PropertySpec{
		Type:    "VirtualMachine",
		PathSet: []string{"config.extraConfig"},
	}

	req := types.RetrieveProperties{
		This: c.ServiceContent.PropertyCollector,
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   []types.PropertySpec{pspec},
			},
		},
	}

	var vms []mo.VirtualMachine
	err := mo.RetrievePropertiesForRequest(ctx, c, req, &vms)
	if err != nil {
		return nil, err
	}

	var ports []int
	for _, vm := range vms {
		if vm.Config == nil || vm.Config.ExtraConfig == nil {
			continue
		}

		options := vncOptionsFromExtraConfig(vm.Config.ExtraConfig)
		if ps, ok := options["port"]; ok && ps != "" {
			pi, err := strconv.Atoi(ps)
			if err == nil {
				ports = append(ports, pi)
			}
		}
	}

	return ports, nil
}

func (h *vncHost) popUnusedPort() (int, error) {
	if len(h.ports) == 0 {
		return 0, fmt.Errorf("no unused ports in range")
	}

	// Return first port we get when iterating
	var port int
	for port = range h.ports {
		break
	}
	delete(h.ports, port)
	return port, nil
}

func (h *vncHost) managementIP() (string, error) {
	ctx := context.TODO()
	if h.ip != "" {
		return h.ip, nil
	}

	ips, err := h.host.ManagementIPs(ctx)
	if err != nil {
		return "", err
	}

	if len(ips) > 0 {
		h.ip = ips[0].String()
	} else {
		h.ip = "<unknown>"
	}

	return h.ip, nil
}

type vncResult []*vncVM

func (vms vncResult) MarshalJSON() ([]byte, error) {
	out := make(map[string]string)
	for _, vm := range vms {
		uri, err := vm.uri()
		if err != nil {
			return nil, err
		}
		out[vm.mvm.Name] = uri
	}
	return json.Marshal(out)
}

func (vms vncResult) Write(w io.Writer) error {
	for _, vm := range vms {
		err := vm.write(w)
		if err != nil {
			return err
		}
	}

	return nil
}

type vncOptions map[string]string

var vncPrefix = "RemoteDisplay.vnc."

func vncOptionsFromExtraConfig(ov []types.BaseOptionValue) vncOptions {
	vo := make(vncOptions)
	for _, b := range ov {
		o := b.GetOptionValue()
		if strings.HasPrefix(o.Key, vncPrefix) {
			key := o.Key[len(vncPrefix):]
			if key != "key" {
				vo[key] = o.Value.(string)
			}
		}
	}
	return vo
}

func (vo vncOptions) ToExtraConfig() []types.BaseOptionValue {
	ov := make([]types.BaseOptionValue, 0, 0)
	for k, v := range vo {
		key := vncPrefix + k
		value := v

		o := types.OptionValue{
			Key:   key,
			Value: &value, // Pass pointer to avoid omitempty
		}

		ov = append(ov, &o)
	}

	// Don't know how to deal with the key option, set it to be empty...
	o := types.OptionValue{
		Key:   vncPrefix + "key",
		Value: new(string), // Pass pointer to avoid omitempty
	}

	ov = append(ov, &o)

	return ov
}
