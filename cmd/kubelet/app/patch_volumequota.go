package app

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/emptydirquota"
)

var (
	volumeConfigKind       = "VolumeConfig"
	volumeConfigAPIVersion = "kubelet.config.openshift.io/v1"
)

// Miror the TypeMeta from k8s since it doesn't have yaml tags.
// We don't really use this right now anyway.  We just want
// the on-disk format of the config to be right in case we
// do want to version/parse the config in the k8s way later.
type TypeMeta struct {
	Kind       string `yaml:"kind"`
	APIVersion string `yaml:"apiVersion"`
}

// VolumeConfig contains options for configuring volumes on the node.
type VolumeConfig struct {
	TypeMeta `yaml:"typeMeta,inline"`

	// LocalQuota contains options for controlling local volume quota on the node.
	LocalQuota LocalQuota `yaml:"localQuota"`
}

// LocalQuota contains options for controlling local volume quota on the node.
type LocalQuota struct {
	// perFSGroup can be specified to enable a quota on local storage use per unique FSGroup ID.
	// At present this is only implemented for emptyDir volumes, and if the underlying
	// volumeDirectory is on an XFS filesystem.
	PerFSGroup string `yaml:"perFSGroup"`
}

// THIS IS PART OF AN OPENSHIFT CARRY PATCH
// PatchVolumePluginsForLocalQuota checks if the node config specifies a local storage
// perFSGroup quota, and if so will test that the volumeDirectory is on a
// filesystem suitable for quota enforcement. If checks pass the k8s emptyDir
// volume plugin will be replaced with a wrapper version which adds quota
// functionality.
func PatchVolumePluginsForLocalQuota(rootdir string, plugins *[]volume.VolumePlugin) error {
	// This is hardcoded in the short term so the we can pull the
	// node and volume config files out of the same configmap
	volumeConfigFilePath := "/etc/origin/node/volume-config.yaml"

	stat, err := os.Stat(volumeConfigFilePath)
	if os.IsNotExist(err) {
		return nil
	}
	if stat.Size() == 0 {
		return nil
	}

	volumeConfigFile, err := ioutil.ReadFile(volumeConfigFilePath)
	if err != nil {
		return fmt.Errorf("failed to read %s: %v", volumeConfigFilePath, err)
	}

	var volumeConfig VolumeConfig
	err = yaml.Unmarshal(volumeConfigFile, &volumeConfig)
	if err != nil {
		return fmt.Errorf("failed to unmarshal %s: %v", volumeConfigFilePath, err)
	}

	if volumeConfig.Kind != volumeConfigKind || volumeConfig.APIVersion != volumeConfigAPIVersion {
		return fmt.Errorf("expected kind \"%s\" and apiVersion \"%s\" for volume config file", volumeConfigKind, volumeConfigAPIVersion)
	}

	quota, err := resource.ParseQuantity(volumeConfig.LocalQuota.PerFSGroup)
	if err != nil {
		return fmt.Errorf("unable to parse \"%s\" as a quantity", volumeConfig.LocalQuota.PerFSGroup)
	}

	if quota.Value() <= 0 {
		return fmt.Errorf("perFSGroup must be a positive quantity")
	}

	klog.V(2).Info("replacing empty-dir volume plugin with quota wrapper")

	quotaApplicator, err := emptydirquota.NewQuotaApplicator(rootdir)
	if err != nil {
		return fmt.Errorf("could not set local quota: %v", err)
	}

	// Create a volume spec with emptyDir we can use to search for the
	// emptyDir plugin with CanSupport()
	emptyDirSpec := &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		},
	}

	wrappedEmptyDirPlugin := false
	for idx, plugin := range *plugins {
		// Can't really do type checking or use a constant here as they are not exported:
		if plugin.CanSupport(emptyDirSpec) {
			wrapper := emptydirquota.EmptyDirQuotaPlugin{
				VolumePlugin:    plugin,
				Quota:           quota,
				QuotaApplicator: quotaApplicator,
			}
			(*plugins)[idx] = &wrapper
			wrappedEmptyDirPlugin = true
		}
	}

	if !wrappedEmptyDirPlugin {
		klog.Fatal(errors.New("no plugin handling EmptyDir was found, unable to apply local quotas"))
	}

	return nil
}
