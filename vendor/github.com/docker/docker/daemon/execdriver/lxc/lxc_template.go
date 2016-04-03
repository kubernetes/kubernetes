// +build linux

package lxc

import (
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	nativeTemplate "github.com/docker/docker/daemon/execdriver/native/template"
	"github.com/docker/docker/pkg/stringutils"
	"github.com/opencontainers/runc/libcontainer/label"
)

const LxcTemplate = `
lxc.network.type = none
# root filesystem
{{$ROOTFS := .Rootfs}}
lxc.rootfs = {{$ROOTFS}}

# use a dedicated pts for the container (and limit the number of pseudo terminal
# available)
lxc.pts = 1024

# disable the main console
lxc.console = none

# no controlling tty at all
lxc.tty = 1

{{if .ProcessConfig.Privileged}}
lxc.cgroup.devices.allow = a
{{else}}
# no implicit access to devices
lxc.cgroup.devices.deny = a
#Allow the devices passed to us in the AllowedDevices list.
{{range $allowedDevice := .AllowedDevices}}
lxc.cgroup.devices.allow = {{$allowedDevice.CgroupString}}
{{end}}
{{end}}

# standard mount point
# Use mnt.putold as per https://bugs.launchpad.net/ubuntu/+source/lxc/+bug/986385
lxc.pivotdir = lxc_putold

# lxc.autodev is not compatible with lxc --device switch
lxc.autodev = 0

# NOTICE: These mounts must be applied within the namespace
{{if .ProcessConfig.Privileged}}
# WARNING: mounting procfs and/or sysfs read-write is a known attack vector.
# See e.g. http://blog.zx2c4.com/749 and https://bit.ly/T9CkqJ
# We mount them read-write here, but later, dockerinit will call the Restrict() function to remount them read-only.
# We cannot mount them directly read-only, because that would prevent loading AppArmor profiles.
lxc.mount.entry = proc {{escapeFstabSpaces $ROOTFS}}/proc proc nosuid,nodev,noexec 0 0
lxc.mount.entry = sysfs {{escapeFstabSpaces $ROOTFS}}/sys sysfs nosuid,nodev,noexec 0 0
	{{if .AppArmor}}
lxc.aa_profile = unconfined
	{{end}}
{{else}}
# In non-privileged mode, lxc will automatically mount /proc and /sys in readonly mode
# for security. See: http://man7.org/linux/man-pages/man5/lxc.container.conf.5.html
lxc.mount.auto = proc sys
	{{if .AppArmorProfile}}
lxc.aa_profile = {{.AppArmorProfile}}
	{{end}}
{{end}}

{{if .ProcessConfig.Tty}}
lxc.mount.entry = {{.ProcessConfig.Console}} {{escapeFstabSpaces $ROOTFS}}/dev/console none bind,rw,create=file 0 0
{{end}}

lxc.mount.entry = devpts {{escapeFstabSpaces $ROOTFS}}/dev/pts devpts {{formatMountLabel "newinstance,ptmxmode=0666,nosuid,noexec,create=dir" ""}} 0 0
lxc.mount.entry = shm {{escapeFstabSpaces $ROOTFS}}/dev/shm tmpfs {{formatMountLabel "size=65536k,nosuid,nodev,noexec,create=dir" ""}} 0 0

{{range $value := .Mounts}}
{{$createVal := isDirectory $value.Source}}
{{if $value.Writable}}
lxc.mount.entry = {{$value.Source}} {{escapeFstabSpaces $ROOTFS}}/{{escapeFstabSpaces $value.Destination}} none rbind,rw,create={{$createVal}} 0 0
{{else}}
lxc.mount.entry = {{$value.Source}} {{escapeFstabSpaces $ROOTFS}}/{{escapeFstabSpaces $value.Destination}} none rbind,ro,create={{$createVal}} 0 0
{{end}}
{{end}}

# limits
{{if .Resources}}
{{if .Resources.Memory}}
lxc.cgroup.memory.limit_in_bytes = {{.Resources.Memory}}
lxc.cgroup.memory.soft_limit_in_bytes = {{.Resources.Memory}}
{{with $memSwap := getMemorySwap .Resources}}
lxc.cgroup.memory.memsw.limit_in_bytes = {{$memSwap}}
{{end}}
{{end}}
{{if .Resources.CpuShares}}
lxc.cgroup.cpu.shares = {{.Resources.CpuShares}}
{{end}}
{{if .Resources.CpuPeriod}}
lxc.cgroup.cpu.cfs_period_us = {{.Resources.CpuPeriod}}
{{end}}
{{if .Resources.CpusetCpus}}
lxc.cgroup.cpuset.cpus = {{.Resources.CpusetCpus}}
{{end}}
{{if .Resources.CpusetMems}}
lxc.cgroup.cpuset.mems = {{.Resources.CpusetMems}}
{{end}}
{{if .Resources.CpuQuota}}
lxc.cgroup.cpu.cfs_quota_us = {{.Resources.CpuQuota}}
{{end}}
{{if .Resources.BlkioWeight}}
lxc.cgroup.blkio.weight = {{.Resources.BlkioWeight}}
{{end}}
{{if .Resources.OomKillDisable}}
lxc.cgroup.memory.oom_control = {{.Resources.OomKillDisable}}
{{end}}
{{if .Resources.MemorySwappiness}}
lxc.cgroup.memory.swappiness = {{.Resources.MemorySwappiness}}
{{end}}
{{end}}

{{if .LxcConfig}}
{{range $value := .LxcConfig}}
lxc.{{$value}}
{{end}}
{{end}}

{{if .Network.Interface}}
{{if .Network.Interface.IPAddress}}
lxc.network.ipv4 = {{.Network.Interface.IPAddress}}/{{.Network.Interface.IPPrefixLen}}
{{end}}
{{if .Network.Interface.Gateway}}
lxc.network.ipv4.gateway = {{.Network.Interface.Gateway}}
{{end}}
{{if .Network.Interface.MacAddress}}
lxc.network.hwaddr = {{.Network.Interface.MacAddress}}
{{end}}
{{end}}
{{if .ProcessConfig.Env}}
lxc.utsname = {{getHostname .ProcessConfig.Env}}
{{end}}

{{if .ProcessConfig.Privileged}}
# No cap values are needed, as lxc is starting in privileged mode
{{else}}
	{{ with keepCapabilities .CapAdd .CapDrop }}
		{{range .}}
lxc.cap.keep = {{.}}
		{{end}}
	{{else}}
		{{ with dropList .CapDrop }}
		{{range .}}
lxc.cap.drop = {{.}}
		{{end}}
		{{end}}
	{{end}}
{{end}}
`

var LxcTemplateCompiled *template.Template

// Escape spaces in strings according to the fstab documentation, which is the
// format for "lxc.mount.entry" lines in lxc.conf. See also "man 5 fstab".
func escapeFstabSpaces(field string) string {
	return strings.Replace(field, " ", "\\040", -1)
}

func keepCapabilities(adds []string, drops []string) ([]string, error) {
	container := nativeTemplate.New()
	logrus.Debugf("adds %s drops %s\n", adds, drops)
	caps, err := execdriver.TweakCapabilities(container.Capabilities, adds, drops)
	if err != nil {
		return nil, err
	}
	var newCaps []string
	for _, cap := range caps {
		logrus.Debugf("cap %s\n", cap)
		realCap := execdriver.GetCapability(cap)
		numCap := fmt.Sprintf("%d", realCap.Value)
		newCaps = append(newCaps, numCap)
	}

	return newCaps, nil
}

func dropList(drops []string) ([]string, error) {
	if stringutils.InSlice(drops, "all") {
		var newCaps []string
		for _, capName := range execdriver.GetAllCapabilities() {
			cap := execdriver.GetCapability(capName)
			logrus.Debugf("drop cap %s\n", cap.Key)
			numCap := fmt.Sprintf("%d", cap.Value)
			newCaps = append(newCaps, numCap)
		}
		return newCaps, nil
	}
	return []string{}, nil
}

func isDirectory(source string) string {
	f, err := os.Stat(source)
	logrus.Debugf("dir: %s\n", source)
	if err != nil {
		if os.IsNotExist(err) {
			return "dir"
		}
		return ""
	}
	if f.IsDir() {
		return "dir"
	}
	return "file"
}

func getMemorySwap(v *execdriver.Resources) int64 {
	// By default, MemorySwap is set to twice the size of RAM.
	// If you want to omit MemorySwap, set it to `-1'.
	if v.MemorySwap < 0 {
		return 0
	}
	return v.Memory * 2
}

func getLabel(c map[string][]string, name string) string {
	label := c["label"]
	for _, l := range label {
		parts := strings.SplitN(l, "=", 2)
		if strings.TrimSpace(parts[0]) == name {
			return strings.TrimSpace(parts[1])
		}
	}
	return ""
}

func getHostname(env []string) string {
	for _, kv := range env {
		parts := strings.SplitN(kv, "=", 2)
		if parts[0] == "HOSTNAME" && len(parts) == 2 {
			return parts[1]
		}
	}
	return ""
}

func init() {
	var err error
	funcMap := template.FuncMap{
		"getMemorySwap":     getMemorySwap,
		"escapeFstabSpaces": escapeFstabSpaces,
		"formatMountLabel":  label.FormatMountLabel,
		"isDirectory":       isDirectory,
		"keepCapabilities":  keepCapabilities,
		"dropList":          dropList,
		"getHostname":       getHostname,
	}
	LxcTemplateCompiled, err = template.New("lxc").Funcs(funcMap).Parse(LxcTemplate)
	if err != nil {
		panic(err)
	}
}
