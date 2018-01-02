package daemon

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/docker/docker/container"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/libcontainerd"
	"golang.org/x/sys/windows/registry"
)

const (
	credentialSpecRegistryLocation = `SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers\CredentialSpecs`
	credentialSpecFileLocation     = "CredentialSpecs"
)

func (daemon *Daemon) getLibcontainerdCreateOptions(container *container.Container) ([]libcontainerd.CreateOption, error) {
	createOptions := []libcontainerd.CreateOption{}

	// Are we going to run as a Hyper-V container?
	hvOpts := &libcontainerd.HyperVIsolationOption{}
	if container.HostConfig.Isolation.IsDefault() {
		// Container is set to use the default, so take the default from the daemon configuration
		hvOpts.IsHyperV = daemon.defaultIsolation.IsHyperV()
	} else {
		// Container is requesting an isolation mode. Honour it.
		hvOpts.IsHyperV = container.HostConfig.Isolation.IsHyperV()
	}

	dnsSearch := daemon.getDNSSearchSettings(container)

	// Generate the layer folder of the layer options
	layerOpts := &libcontainerd.LayerOption{}
	m, err := container.RWLayer.Metadata()
	if err != nil {
		return nil, fmt.Errorf("failed to get layer metadata - %s", err)
	}
	layerOpts.LayerFolderPath = m["dir"]

	// Generate the layer paths of the layer options
	img, err := daemon.stores[container.Platform].imageStore.Get(container.ImageID)
	if err != nil {
		return nil, fmt.Errorf("failed to graph.Get on ImageID %s - %s", container.ImageID, err)
	}
	// Get the layer path for each layer.
	max := len(img.RootFS.DiffIDs)
	for i := 1; i <= max; i++ {
		img.RootFS.DiffIDs = img.RootFS.DiffIDs[:i]
		layerPath, err := layer.GetLayerPath(daemon.stores[container.Platform].layerStore, img.RootFS.ChainID())
		if err != nil {
			return nil, fmt.Errorf("failed to get layer path from graphdriver %s for ImageID %s - %s", daemon.stores[container.Platform].layerStore, img.RootFS.ChainID(), err)
		}
		// Reverse order, expecting parent most first
		layerOpts.LayerPaths = append([]string{layerPath}, layerOpts.LayerPaths...)
	}

	// Get endpoints for the libnetwork allocated networks to the container
	var epList []string
	AllowUnqualifiedDNSQuery := false
	gwHNSID := ""
	if container.NetworkSettings != nil {
		for n := range container.NetworkSettings.Networks {
			sn, err := daemon.FindNetwork(n)
			if err != nil {
				continue
			}

			ep, err := container.GetEndpointInNetwork(sn)
			if err != nil {
				continue
			}

			data, err := ep.DriverInfo()
			if err != nil {
				continue
			}

			if data["GW_INFO"] != nil {
				gwInfo := data["GW_INFO"].(map[string]interface{})
				if gwInfo["hnsid"] != nil {
					gwHNSID = gwInfo["hnsid"].(string)
				}
			}

			if data["hnsid"] != nil {
				epList = append(epList, data["hnsid"].(string))
			}

			if data["AllowUnqualifiedDNSQuery"] != nil {
				AllowUnqualifiedDNSQuery = true
			}
		}
	}

	if gwHNSID != "" {
		epList = append(epList, gwHNSID)
	}

	// Read and add credentials from the security options if a credential spec has been provided.
	if container.HostConfig.SecurityOpt != nil {
		for _, sOpt := range container.HostConfig.SecurityOpt {
			sOpt = strings.ToLower(sOpt)
			if !strings.Contains(sOpt, "=") {
				return nil, fmt.Errorf("invalid security option: no equals sign in supplied value %s", sOpt)
			}
			var splitsOpt []string
			splitsOpt = strings.SplitN(sOpt, "=", 2)
			if len(splitsOpt) != 2 {
				return nil, fmt.Errorf("invalid security option: %s", sOpt)
			}
			if splitsOpt[0] != "credentialspec" {
				return nil, fmt.Errorf("security option not supported: %s", splitsOpt[0])
			}

			credentialsOpts := &libcontainerd.CredentialsOption{}
			var (
				match   bool
				csValue string
				err     error
			)
			if match, csValue = getCredentialSpec("file://", splitsOpt[1]); match {
				if csValue == "" {
					return nil, fmt.Errorf("no value supplied for file:// credential spec security option")
				}
				if credentialsOpts.Credentials, err = readCredentialSpecFile(container.ID, daemon.root, filepath.Clean(csValue)); err != nil {
					return nil, err
				}
			} else if match, csValue = getCredentialSpec("registry://", splitsOpt[1]); match {
				if csValue == "" {
					return nil, fmt.Errorf("no value supplied for registry:// credential spec security option")
				}
				if credentialsOpts.Credentials, err = readCredentialSpecRegistry(container.ID, csValue); err != nil {
					return nil, err
				}
			} else {
				return nil, fmt.Errorf("invalid credential spec security option - value must be prefixed file:// or registry:// followed by a value")
			}
			createOptions = append(createOptions, credentialsOpts)
		}
	}

	// Now add the remaining options.
	createOptions = append(createOptions, &libcontainerd.FlushOption{IgnoreFlushesDuringBoot: !container.HasBeenStartedBefore})
	createOptions = append(createOptions, hvOpts)
	createOptions = append(createOptions, layerOpts)

	var networkSharedContainerID string
	if container.HostConfig.NetworkMode.IsContainer() {
		networkSharedContainerID = container.NetworkSharedContainerID
		for _, ep := range container.SharedEndpointList {
			epList = append(epList, ep)
		}
	}

	createOptions = append(createOptions, &libcontainerd.NetworkEndpointsOption{
		Endpoints:                epList,
		AllowUnqualifiedDNSQuery: AllowUnqualifiedDNSQuery,
		DNSSearchList:            dnsSearch,
		NetworkSharedContainerID: networkSharedContainerID,
	})
	return createOptions, nil
}

// getCredentialSpec is a helper function to get the value of a credential spec supplied
// on the CLI, stripping the prefix
func getCredentialSpec(prefix, value string) (bool, string) {
	if strings.HasPrefix(value, prefix) {
		return true, strings.TrimPrefix(value, prefix)
	}
	return false, ""
}

// readCredentialSpecRegistry is a helper function to read a credential spec from
// the registry. If not found, we return an empty string and warn in the log.
// This allows for staging on machines which do not have the necessary components.
func readCredentialSpecRegistry(id, name string) (string, error) {
	var (
		k   registry.Key
		err error
		val string
	)
	if k, err = registry.OpenKey(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.QUERY_VALUE); err != nil {
		return "", fmt.Errorf("failed handling spec %q for container %s - %s could not be opened", name, id, credentialSpecRegistryLocation)
	}
	if val, _, err = k.GetStringValue(name); err != nil {
		if err == registry.ErrNotExist {
			return "", fmt.Errorf("credential spec %q for container %s as it was not found", name, id)
		}
		return "", fmt.Errorf("error %v reading credential spec %q from registry for container %s", err, name, id)
	}
	return val, nil
}

// readCredentialSpecFile is a helper function to read a credential spec from
// a file. If not found, we return an empty string and warn in the log.
// This allows for staging on machines which do not have the necessary components.
func readCredentialSpecFile(id, root, location string) (string, error) {
	if filepath.IsAbs(location) {
		return "", fmt.Errorf("invalid credential spec - file:// path cannot be absolute")
	}
	base := filepath.Join(root, credentialSpecFileLocation)
	full := filepath.Join(base, location)
	if !strings.HasPrefix(full, base) {
		return "", fmt.Errorf("invalid credential spec - file:// path must be under %s", base)
	}
	bcontents, err := ioutil.ReadFile(full)
	if err != nil {
		return "", fmt.Errorf("credential spec '%s' for container %s as the file could not be read: %q", full, id, err)
	}
	return string(bcontents[:]), nil
}
