//+build windows

package windows

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/opengcs/client"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/log"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

// newContainerConfig generates a hcsshim container configuration from the
// provided OCI Spec
func newContainerConfig(ctx context.Context, owner, id string, spec *specs.Spec) (*hcsshim.ContainerConfig, error) {
	var (
		conf = &hcsshim.ContainerConfig{
			SystemType: "Container",
			Name:       id,
			Owner:      owner,
			HostName:   spec.Hostname,
		}
	)

	if spec.Windows.Network != nil {
		conf.AllowUnqualifiedDNSQuery = spec.Windows.Network.AllowUnqualifiedDNSQuery
		conf.EndpointList = spec.Windows.Network.EndpointList
		conf.NetworkSharedContainerName = spec.Windows.Network.NetworkSharedContainerName
		if spec.Windows.Network.DNSSearchList != nil {
			conf.DNSSearchList = strings.Join(spec.Windows.Network.DNSSearchList, ",")
		}
	}

	return conf, nil
}

// newWindowsContainerConfig generates a hcsshim Windows container
// configuration from the provided OCI Spec
func newWindowsContainerConfig(ctx context.Context, owner, id string, spec *specs.Spec) (*hcsshim.ContainerConfig, error) {
	conf, err := newContainerConfig(ctx, owner, id, spec)
	if err != nil {
		return nil, err
	}
	conf.IgnoreFlushesDuringBoot = spec.Windows.IgnoreFlushesDuringBoot

	if len(spec.Windows.LayerFolders) < 1 {
		return nil, errors.Wrap(errdefs.ErrInvalidArgument,
			"spec.Windows.LayerFolders must have at least 1 layers")
	}
	var (
		layerFolders    = spec.Windows.LayerFolders
		homeDir         = filepath.Dir(layerFolders[0])
		layerFolderPath = filepath.Join(homeDir, id)
	)

	// TODO: use the create request Mount for those
	for _, layerPath := range layerFolders {
		_, filename := filepath.Split(layerPath)
		guid, err := hcsshim.NameToGuid(filename)
		if err != nil {
			return nil, errors.Wrapf(err, "unable to get GUID for %s", filename)
		}
		conf.Layers = append(conf.Layers, hcsshim.Layer{
			ID:   guid.ToString(),
			Path: layerPath,
		})
	}

	var (
		di = hcsshim.DriverInfo{
			Flavour: 1, // filter driver
			HomeDir: homeDir,
		}
	)
	conf.LayerFolderPath = layerFolderPath

	// TODO: Once there is a snapshotter for windows, this can be deleted.
	// The R/W Layer should come from the Rootfs Mounts provided
	//
	// Windows doesn't support creating a container with a readonly
	// filesystem, so always create a RW one
	if err = hcsshim.CreateSandboxLayer(di, id, layerFolders[0], layerFolders); err != nil {
		return nil, errors.Wrapf(err, "failed to create sandbox layer for %s: layers: %#v, driverInfo: %#v",
			id, layerFolders, di)
	}
	defer func() {
		if err != nil {
			removeLayer(ctx, conf.LayerFolderPath)
		}
	}()

	if err = hcsshim.ActivateLayer(di, id); err != nil {
		return nil, errors.Wrapf(err, "failed to activate layer %s", conf.LayerFolderPath)
	}

	if err = hcsshim.PrepareLayer(di, id, layerFolders); err != nil {
		return nil, errors.Wrapf(err, "failed to prepare layer %s", conf.LayerFolderPath)
	}

	conf.VolumePath, err = hcsshim.GetLayerMountPath(di, id)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to getmount path for layer %s: driverInfo: %#v", id, di)
	}

	if spec.Windows.HyperV != nil {
		conf.HvPartition = true
		for _, layerPath := range layerFolders {
			utilityVMPath := spec.Windows.HyperV.UtilityVMPath
			_, err := os.Stat(utilityVMPath)
			if err == nil {
				conf.HvRuntime = &hcsshim.HvRuntime{ImagePath: utilityVMPath}
				break
			} else if !os.IsNotExist(err) {
				return nil, errors.Wrapf(err, "failed to access layer %s", layerPath)
			}
		}
	}

	if spec.Windows.CredentialSpec != nil {
		conf.Credentials = spec.Windows.CredentialSpec.(string)
	}

	if len(spec.Mounts) > 0 {
		mds := make([]hcsshim.MappedDir, len(spec.Mounts))
		for i, mount := range spec.Mounts {
			mds[i] = hcsshim.MappedDir{
				HostPath:      mount.Source,
				ContainerPath: mount.Destination,
				ReadOnly:      false,
			}
			for _, o := range mount.Options {
				if strings.ToLower(o) == "ro" {
					mds[i].ReadOnly = true
				}
			}
		}
		conf.MappedDirectories = mds
	}

	return conf, nil
}

// newLinuxConfig generates a hcsshim Linux container configuration from the
// provided OCI Spec
func newLinuxConfig(ctx context.Context, owner, id string, spec *specs.Spec) (*hcsshim.ContainerConfig, error) {
	conf, err := newContainerConfig(ctx, owner, id, spec)
	if err != nil {
		return nil, err
	}

	conf.ContainerType = "Linux"
	conf.HvPartition = true

	if len(spec.Windows.LayerFolders) < 1 {
		return nil, errors.Wrap(errdefs.ErrInvalidArgument,
			"spec.Windows.LayerFolders must have at least 1 layer")
	}
	var (
		layerFolders = spec.Windows.LayerFolders
	)

	config := &client.Config{}
	if err := config.GenerateDefault(nil); err != nil {
		return nil, err
	}

	conf.HvRuntime = &hcsshim.HvRuntime{
		ImagePath:           config.KirdPath,
		LinuxKernelFile:     config.KernelFile,
		LinuxInitrdFile:     config.InitrdFile,
		LinuxBootParameters: config.BootParameters,
	}

	// TODO: use the create request Mount for those
	for _, layerPath := range layerFolders {
		_, filename := filepath.Split(layerPath)
		guid, err := hcsshim.NameToGuid(filename)
		if err != nil {
			return nil, errors.Wrapf(err, "unable to get GUID for %s", filename)
		}
		conf.Layers = append(conf.Layers, hcsshim.Layer{
			ID:   guid.ToString(),
			Path: filepath.Join(layerPath, "layer.vhd"),
		})
	}

	return conf, nil
}

// removeLayer deletes the given layer, all associated containers must have
// been shutdown for this to succeed.
func removeLayer(ctx context.Context, path string) error {
	var (
		err        error
		layerID    = filepath.Base(path)
		parentPath = filepath.Dir(path)
		di         = hcsshim.DriverInfo{
			Flavour: 1, // filter driver
			HomeDir: parentPath,
		}
	)

	if err = hcsshim.UnprepareLayer(di, layerID); err != nil {
		log.G(ctx).WithError(err).Warnf("failed to unprepare layer %s for removal", path)
	}

	if err = hcsshim.DeactivateLayer(di, layerID); err != nil {
		log.G(ctx).WithError(err).Warnf("failed to deactivate layer %s for removal", path)
	}

	removePath := filepath.Join(parentPath, fmt.Sprintf("%s-removing", layerID))
	if err = os.Rename(path, removePath); err != nil {
		log.G(ctx).WithError(err).Warnf("failed to rename container layer %s for removal", path)
		removePath = path
	}

	if err = hcsshim.DestroyLayer(di, removePath); err != nil {
		log.G(ctx).WithError(err).Errorf("failed to remove container layer %s", removePath)
		return err
	}

	return nil
}

func newProcessConfig(processSpec *specs.Process, pset *pipeSet) *hcsshim.ProcessConfig {
	conf := &hcsshim.ProcessConfig{
		EmulateConsole:   pset.src.Terminal,
		CreateStdInPipe:  pset.stdin != nil,
		CreateStdOutPipe: pset.stdout != nil,
		CreateStdErrPipe: pset.stderr != nil,
		User:             processSpec.User.Username,
		Environment:      make(map[string]string),
		WorkingDirectory: processSpec.Cwd,
	}

	if processSpec.ConsoleSize != nil {
		conf.ConsoleSize = [2]uint{processSpec.ConsoleSize.Height, processSpec.ConsoleSize.Width}
	}

	// Convert OCI Env format to HCS's
	for _, s := range processSpec.Env {
		arr := strings.SplitN(s, "=", 2)
		if len(arr) == 2 {
			conf.Environment[arr[0]] = arr[1]
		}
	}

	return conf
}

func newWindowsProcessConfig(processSpec *specs.Process, pset *pipeSet) *hcsshim.ProcessConfig {
	conf := newProcessConfig(processSpec, pset)
	conf.CommandLine = strings.Join(processSpec.Args, " ")
	return conf
}

func newLinuxProcessConfig(processSpec *specs.Process, pset *pipeSet) (*hcsshim.ProcessConfig, error) {
	conf := newProcessConfig(processSpec, pset)
	conf.CommandArgs = processSpec.Args
	return conf, nil
}
