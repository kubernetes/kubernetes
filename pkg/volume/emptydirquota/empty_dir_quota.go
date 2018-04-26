package emptydirquota

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
)

var _ volume.VolumePlugin = &EmptyDirQuotaPlugin{}
var _ volume.Mounter = &emptyDirQuotaMounter{}

// EmptyDirQuotaPlugin is a simple wrapper for the k8s empty dir plugin mounter.
type EmptyDirQuotaPlugin struct {
	// the actual k8s emptyDir volume plugin we will pass method calls to.
	// TODO: do we need to implement unmount
	volume.VolumePlugin

	// The default quota to apply to each node:
	Quota resource.Quantity

	// QuotaApplicator is passed to actual volume mounters so they can apply
	// quota for the supported filesystem.
	QuotaApplicator QuotaApplicator
}

func (plugin *EmptyDirQuotaPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	volMounter, err := plugin.VolumePlugin.NewMounter(spec, pod, opts)
	if err != nil {
		return volMounter, err
	}

	// Because we cannot access several fields on the k8s emptyDir struct, and
	// we do not wish to modify k8s code for this, we have to grab a reference
	// to them ourselves.
	// This logic is the same as k8s.io/kubernetes/pkg/volume/empty_dir:
	medium := v1.StorageMediumDefault
	if spec.Volume.EmptyDir != nil { // Support a non-specified source as EmptyDir.
		medium = spec.Volume.EmptyDir.Medium
	}

	// Wrap the mounter object with our own to add quota functionality:
	wrapperEmptyDir := &emptyDirQuotaMounter{
		wrapped:         volMounter,
		pod:             pod,
		medium:          medium,
		quota:           plugin.Quota,
		quotaApplicator: plugin.QuotaApplicator,
	}
	return wrapperEmptyDir, err
}

// emptyDirQuotaMounter is a wrapper plugin mounter for the k8s empty dir mounter itself.
// This plugin just extends and adds the functionality to apply a
// quota for the pods FSGroup on an XFS filesystem.
type emptyDirQuotaMounter struct {
	wrapped         volume.Mounter
	pod             *v1.Pod
	medium          v1.StorageMedium
	quota           resource.Quantity
	quotaApplicator QuotaApplicator
}

func (edq *emptyDirQuotaMounter) CanMount() error {
	return edq.wrapped.CanMount()
}

// Must implement SetUp as well, otherwise the internal Mounter.SetUp calls its
// own SetUpAt method, not the one we need.

func (edq *emptyDirQuotaMounter) SetUp(opts volume.MounterArgs) error {
	return edq.SetUpAt(edq.GetPath(), opts)
}

func (edq *emptyDirQuotaMounter) SetUpAt(dir string, opts volume.MounterArgs) error {
	err := edq.wrapped.SetUpAt(dir, opts)
	if err == nil {
		err = edq.quotaApplicator.Apply(dir, edq.medium, edq.pod, opts.FsGroup, edq.quota)
	}
	return err
}

func (edq *emptyDirQuotaMounter) GetAttributes() volume.Attributes {
	return edq.wrapped.GetAttributes()
}

func (edq *emptyDirQuotaMounter) GetMetrics() (*volume.Metrics, error) {
	return edq.wrapped.GetMetrics()
}

func (edq *emptyDirQuotaMounter) GetPath() string {
	return edq.wrapped.GetPath()
}
