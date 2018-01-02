// +build linux freebsd solaris

package container

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	mounttypes "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/volume"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

const (
	containerSecretMountPath = "/run/secrets"
)

// ExitStatus provides exit reasons for a container.
type ExitStatus struct {
	// The exit code with which the container exited.
	ExitCode int

	// Whether the container encountered an OOM.
	OOMKilled bool
}

// TrySetNetworkMount attempts to set the network mounts given a provided destination and
// the path to use for it; return true if the given destination was a network mount file
func (container *Container) TrySetNetworkMount(destination string, path string) bool {
	if destination == "/etc/resolv.conf" {
		container.ResolvConfPath = path
		return true
	}
	if destination == "/etc/hostname" {
		container.HostnamePath = path
		return true
	}
	if destination == "/etc/hosts" {
		container.HostsPath = path
		return true
	}

	return false
}

// BuildHostnameFile writes the container's hostname file.
func (container *Container) BuildHostnameFile() error {
	hostnamePath, err := container.GetRootResourcePath("hostname")
	if err != nil {
		return err
	}
	container.HostnamePath = hostnamePath
	return ioutil.WriteFile(container.HostnamePath, []byte(container.Config.Hostname+"\n"), 0644)
}

// NetworkMounts returns the list of network mounts.
func (container *Container) NetworkMounts() []Mount {
	var mounts []Mount
	shared := container.HostConfig.NetworkMode.IsContainer()
	if container.ResolvConfPath != "" {
		if _, err := os.Stat(container.ResolvConfPath); err != nil {
			logrus.Warnf("ResolvConfPath set to %q, but can't stat this filename (err = %v); skipping", container.ResolvConfPath, err)
		} else {
			if !container.HasMountFor("/etc/resolv.conf") {
				label.Relabel(container.ResolvConfPath, container.MountLabel, shared)
			}
			writable := !container.HostConfig.ReadonlyRootfs
			if m, exists := container.MountPoints["/etc/resolv.conf"]; exists {
				writable = m.RW
			}
			mounts = append(mounts, Mount{
				Source:      container.ResolvConfPath,
				Destination: "/etc/resolv.conf",
				Writable:    writable,
				Propagation: string(volume.DefaultPropagationMode),
			})
		}
	}
	if container.HostnamePath != "" {
		if _, err := os.Stat(container.HostnamePath); err != nil {
			logrus.Warnf("HostnamePath set to %q, but can't stat this filename (err = %v); skipping", container.HostnamePath, err)
		} else {
			if !container.HasMountFor("/etc/hostname") {
				label.Relabel(container.HostnamePath, container.MountLabel, shared)
			}
			writable := !container.HostConfig.ReadonlyRootfs
			if m, exists := container.MountPoints["/etc/hostname"]; exists {
				writable = m.RW
			}
			mounts = append(mounts, Mount{
				Source:      container.HostnamePath,
				Destination: "/etc/hostname",
				Writable:    writable,
				Propagation: string(volume.DefaultPropagationMode),
			})
		}
	}
	if container.HostsPath != "" {
		if _, err := os.Stat(container.HostsPath); err != nil {
			logrus.Warnf("HostsPath set to %q, but can't stat this filename (err = %v); skipping", container.HostsPath, err)
		} else {
			if !container.HasMountFor("/etc/hosts") {
				label.Relabel(container.HostsPath, container.MountLabel, shared)
			}
			writable := !container.HostConfig.ReadonlyRootfs
			if m, exists := container.MountPoints["/etc/hosts"]; exists {
				writable = m.RW
			}
			mounts = append(mounts, Mount{
				Source:      container.HostsPath,
				Destination: "/etc/hosts",
				Writable:    writable,
				Propagation: string(volume.DefaultPropagationMode),
			})
		}
	}
	return mounts
}

// CopyImagePathContent copies files in destination to the volume.
func (container *Container) CopyImagePathContent(v volume.Volume, destination string) error {
	rootfs, err := symlink.FollowSymlinkInScope(filepath.Join(container.BaseFS, destination), container.BaseFS)
	if err != nil {
		return err
	}

	if _, err = ioutil.ReadDir(rootfs); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	id := stringid.GenerateNonCryptoID()
	path, err := v.Mount(id)
	if err != nil {
		return err
	}

	defer func() {
		if err := v.Unmount(id); err != nil {
			logrus.Warnf("error while unmounting volume %s: %v", v.Name(), err)
		}
	}()
	if err := label.Relabel(path, container.MountLabel, true); err != nil && err != unix.ENOTSUP {
		return err
	}
	return copyExistingContents(rootfs, path)
}

// ShmResourcePath returns path to shm
func (container *Container) ShmResourcePath() (string, error) {
	return container.GetRootResourcePath("shm")
}

// HasMountFor checks if path is a mountpoint
func (container *Container) HasMountFor(path string) bool {
	_, exists := container.MountPoints[path]
	return exists
}

// UnmountIpcMounts uses the provided unmount function to unmount shm and mqueue if they were mounted
func (container *Container) UnmountIpcMounts(unmount func(pth string) error) {
	if container.HostConfig.IpcMode.IsContainer() || container.HostConfig.IpcMode.IsHost() {
		return
	}

	var warnings []string

	if !container.HasMountFor("/dev/shm") {
		shmPath, err := container.ShmResourcePath()
		if err != nil {
			logrus.Error(err)
			warnings = append(warnings, err.Error())
		} else if shmPath != "" {
			if err := unmount(shmPath); err != nil && !os.IsNotExist(err) {
				if mounted, mErr := mount.Mounted(shmPath); mounted || mErr != nil {
					warnings = append(warnings, fmt.Sprintf("failed to umount %s: %v", shmPath, err))
				}
			}

		}
	}

	if len(warnings) > 0 {
		logrus.Warnf("failed to cleanup ipc mounts:\n%v", strings.Join(warnings, "\n"))
	}
}

// IpcMounts returns the list of IPC mounts
func (container *Container) IpcMounts() []Mount {
	var mounts []Mount

	if !container.HasMountFor("/dev/shm") {
		label.SetFileLabel(container.ShmPath, container.MountLabel)
		mounts = append(mounts, Mount{
			Source:      container.ShmPath,
			Destination: "/dev/shm",
			Writable:    true,
			Propagation: string(volume.DefaultPropagationMode),
		})
	}

	return mounts
}

// SecretMounts returns the mounts for the secret path.
func (container *Container) SecretMounts() []Mount {
	var mounts []Mount
	for _, r := range container.SecretReferences {
		if r.File == nil {
			continue
		}
		mounts = append(mounts, Mount{
			Source:      container.SecretFilePath(*r),
			Destination: getSecretTargetPath(r),
			Writable:    false,
		})
	}

	return mounts
}

// UnmountSecrets unmounts the local tmpfs for secrets
func (container *Container) UnmountSecrets() error {
	if _, err := os.Stat(container.SecretMountPath()); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	return detachMounted(container.SecretMountPath())
}

// ConfigMounts returns the mounts for configs.
func (container *Container) ConfigMounts() []Mount {
	var mounts []Mount
	for _, configRef := range container.ConfigReferences {
		if configRef.File == nil {
			continue
		}
		mounts = append(mounts, Mount{
			Source:      container.ConfigFilePath(*configRef),
			Destination: configRef.File.Name,
			Writable:    false,
		})
	}

	return mounts
}

// UpdateContainer updates configuration of a container. Callers must hold a Lock on the Container.
func (container *Container) UpdateContainer(hostConfig *containertypes.HostConfig) error {
	// update resources of container
	resources := hostConfig.Resources
	cResources := &container.HostConfig.Resources

	// validate NanoCPUs, CPUPeriod, and CPUQuota
	// Because NanoCPU effectively updates CPUPeriod/CPUQuota,
	// once NanoCPU is already set, updating CPUPeriod/CPUQuota will be blocked, and vice versa.
	// In the following we make sure the intended update (resources) does not conflict with the existing (cResource).
	if resources.NanoCPUs > 0 && cResources.CPUPeriod > 0 {
		return fmt.Errorf("Conflicting options: Nano CPUs cannot be updated as CPU Period has already been set")
	}
	if resources.NanoCPUs > 0 && cResources.CPUQuota > 0 {
		return fmt.Errorf("Conflicting options: Nano CPUs cannot be updated as CPU Quota has already been set")
	}
	if resources.CPUPeriod > 0 && cResources.NanoCPUs > 0 {
		return fmt.Errorf("Conflicting options: CPU Period cannot be updated as NanoCPUs has already been set")
	}
	if resources.CPUQuota > 0 && cResources.NanoCPUs > 0 {
		return fmt.Errorf("Conflicting options: CPU Quota cannot be updated as NanoCPUs has already been set")
	}

	if resources.BlkioWeight != 0 {
		cResources.BlkioWeight = resources.BlkioWeight
	}
	if resources.CPUShares != 0 {
		cResources.CPUShares = resources.CPUShares
	}
	if resources.NanoCPUs != 0 {
		cResources.NanoCPUs = resources.NanoCPUs
	}
	if resources.CPUPeriod != 0 {
		cResources.CPUPeriod = resources.CPUPeriod
	}
	if resources.CPUQuota != 0 {
		cResources.CPUQuota = resources.CPUQuota
	}
	if resources.CpusetCpus != "" {
		cResources.CpusetCpus = resources.CpusetCpus
	}
	if resources.CpusetMems != "" {
		cResources.CpusetMems = resources.CpusetMems
	}
	if resources.Memory != 0 {
		// if memory limit smaller than already set memoryswap limit and doesn't
		// update the memoryswap limit, then error out.
		if resources.Memory > cResources.MemorySwap && resources.MemorySwap == 0 {
			return fmt.Errorf("Memory limit should be smaller than already set memoryswap limit, update the memoryswap at the same time")
		}
		cResources.Memory = resources.Memory
	}
	if resources.MemorySwap != 0 {
		cResources.MemorySwap = resources.MemorySwap
	}
	if resources.MemoryReservation != 0 {
		cResources.MemoryReservation = resources.MemoryReservation
	}
	if resources.KernelMemory != 0 {
		cResources.KernelMemory = resources.KernelMemory
	}

	// update HostConfig of container
	if hostConfig.RestartPolicy.Name != "" {
		if container.HostConfig.AutoRemove && !hostConfig.RestartPolicy.IsNone() {
			return fmt.Errorf("Restart policy cannot be updated because AutoRemove is enabled for the container")
		}
		container.HostConfig.RestartPolicy = hostConfig.RestartPolicy
	}

	return nil
}

// DetachAndUnmount uses a detached mount on all mount destinations, then
// unmounts each volume normally.
// This is used from daemon/archive for `docker cp`
func (container *Container) DetachAndUnmount(volumeEventLog func(name, action string, attributes map[string]string)) error {
	networkMounts := container.NetworkMounts()
	mountPaths := make([]string, 0, len(container.MountPoints)+len(networkMounts))

	for _, mntPoint := range container.MountPoints {
		dest, err := container.GetResourcePath(mntPoint.Destination)
		if err != nil {
			logrus.Warnf("Failed to get volume destination path for container '%s' at '%s' while lazily unmounting: %v", container.ID, mntPoint.Destination, err)
			continue
		}
		mountPaths = append(mountPaths, dest)
	}

	for _, m := range networkMounts {
		dest, err := container.GetResourcePath(m.Destination)
		if err != nil {
			logrus.Warnf("Failed to get volume destination path for container '%s' at '%s' while lazily unmounting: %v", container.ID, m.Destination, err)
			continue
		}
		mountPaths = append(mountPaths, dest)
	}

	for _, mountPath := range mountPaths {
		if err := detachMounted(mountPath); err != nil {
			logrus.Warnf("%s unmountVolumes: Failed to do lazy umount fo volume '%s': %v", container.ID, mountPath, err)
		}
	}
	return container.UnmountVolumes(volumeEventLog)
}

// copyExistingContents copies from the source to the destination and
// ensures the ownership is appropriately set.
func copyExistingContents(source, destination string) error {
	volList, err := ioutil.ReadDir(source)
	if err != nil {
		return err
	}
	if len(volList) > 0 {
		srcList, err := ioutil.ReadDir(destination)
		if err != nil {
			return err
		}
		if len(srcList) == 0 {
			// If the source volume is empty, copies files from the root into the volume
			if err := chrootarchive.NewArchiver(nil).CopyWithTar(source, destination); err != nil {
				return err
			}
		}
	}
	return copyOwnership(source, destination)
}

// copyOwnership copies the permissions and uid:gid of the source file
// to the destination file
func copyOwnership(source, destination string) error {
	stat, err := system.Stat(source)
	if err != nil {
		return err
	}

	destStat, err := system.Stat(destination)
	if err != nil {
		return err
	}

	// In some cases, even though UID/GID match and it would effectively be a no-op,
	// this can return a permission denied error... for example if this is an NFS
	// mount.
	// Since it's not really an error that we can't chown to the same UID/GID, don't
	// even bother trying in such cases.
	if stat.UID() != destStat.UID() || stat.GID() != destStat.GID() {
		if err := os.Chown(destination, int(stat.UID()), int(stat.GID())); err != nil {
			return err
		}
	}

	if stat.Mode() != destStat.Mode() {
		return os.Chmod(destination, os.FileMode(stat.Mode()))
	}
	return nil
}

// TmpfsMounts returns the list of tmpfs mounts
func (container *Container) TmpfsMounts() ([]Mount, error) {
	var mounts []Mount
	for dest, data := range container.HostConfig.Tmpfs {
		mounts = append(mounts, Mount{
			Source:      "tmpfs",
			Destination: dest,
			Data:        data,
		})
	}
	for dest, mnt := range container.MountPoints {
		if mnt.Type == mounttypes.TypeTmpfs {
			data, err := volume.ConvertTmpfsOptions(mnt.Spec.TmpfsOptions, mnt.Spec.ReadOnly)
			if err != nil {
				return nil, err
			}
			mounts = append(mounts, Mount{
				Source:      "tmpfs",
				Destination: dest,
				Data:        data,
			})
		}
	}
	return mounts, nil
}

// cleanResourcePath cleans a resource path and prepares to combine with mnt path
func cleanResourcePath(path string) string {
	return filepath.Join(string(os.PathSeparator), path)
}

// EnableServiceDiscoveryOnDefaultNetwork Enable service discovery on default network
func (container *Container) EnableServiceDiscoveryOnDefaultNetwork() bool {
	return false
}

// GetMountPoints gives a platform specific transformation to types.MountPoint. Callers must hold a Container lock.
func (container *Container) GetMountPoints() []types.MountPoint {
	mountPoints := make([]types.MountPoint, 0, len(container.MountPoints))
	for _, m := range container.MountPoints {
		mountPoints = append(mountPoints, types.MountPoint{
			Type:        m.Type,
			Name:        m.Name,
			Source:      m.Path(),
			Destination: m.Destination,
			Driver:      m.Driver,
			Mode:        m.Mode,
			RW:          m.RW,
			Propagation: m.Propagation,
		})
	}
	return mountPoints
}
