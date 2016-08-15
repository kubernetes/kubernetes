/*
Copyright 2015 The Kubernetes Authors.

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

package flocker

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/env"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"

	flockerclient "github.com/ClusterHQ/flocker-go"
)

const (
	flockerPluginName = "kubernetes.io/flocker"

	defaultHost           = "localhost"
	defaultPort           = 4523
	defaultCACertFile     = "/etc/flocker/cluster.crt"
	defaultClientKeyFile  = "/etc/flocker/apiuser.key"
	defaultClientCertFile = "/etc/flocker/apiuser.crt"

	timeoutWaitingForVolume = 2 * time.Minute
	tickerWaitingForVolume  = 5 * time.Second
)

func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&flockerPlugin{}}
}

type flockerPlugin struct {
	host volume.VolumeHost
}

type flocker struct {
	datasetName string
	path        string
	pod         *api.Pod
	mounter     mount.Interface
	plugin      *flockerPlugin
}

func (p *flockerPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	return nil
}

func (p *flockerPlugin) GetPluginName() string {
	return flockerPluginName
}

func (p *flockerPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.DatasetName, nil
}

func (p *flockerPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Flocker != nil) ||
		(spec.Volume != nil && spec.Volume.Flocker != nil)
}

func (p *flockerPlugin) RequiresRemount() bool {
	return false
}

func (p *flockerPlugin) getFlockerVolumeSource(spec *volume.Spec) (*api.FlockerVolumeSource, bool) {
	// AFAIK this will always be r/w, but perhaps for the future it will be needed
	readOnly := false

	if spec.Volume != nil && spec.Volume.Flocker != nil {
		return spec.Volume.Flocker, readOnly
	}
	return spec.PersistentVolume.Spec.Flocker, readOnly
}

func (p *flockerPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	source, readOnly := p.getFlockerVolumeSource(spec)
	mounter := flockerMounter{
		flocker: &flocker{
			datasetName: source.DatasetName,
			pod:         pod,
			mounter:     p.host.GetMounter(),
			plugin:      p,
		},
		exe:      exec.New(),
		opts:     opts,
		readOnly: readOnly,
	}
	return &mounter, nil
}

func (p *flockerPlugin) NewUnmounter(datasetName string, podUID types.UID) (volume.Unmounter, error) {
	// Flocker agent will take care of this, there is nothing we can do here
	return nil, nil
}

func (p *flockerPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	flockerVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Flocker: &api.FlockerVolumeSource{
				DatasetName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(flockerVolume), nil
}

type flockerMounter struct {
	*flocker
	client   flockerclient.Clientable
	exe      exec.Interface
	opts     volume.VolumeOptions
	readOnly bool
	volume.MetricsNil
}

func (b flockerMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}
func (b flockerMounter) GetPath() string {
	return b.flocker.path
}

func (b flockerMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.flocker.datasetName, fsGroup)
}

// newFlockerClient uses environment variables and pod attributes to return a
// flocker client capable of talking with the Flocker control service.
func (b flockerMounter) newFlockerClient() (*flockerclient.Client, error) {
	host := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_HOST", defaultHost)
	port, err := env.GetEnvAsIntOrFallback("FLOCKER_CONTROL_SERVICE_PORT", defaultPort)

	if err != nil {
		return nil, err
	}
	caCertPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CA_FILE", defaultCACertFile)
	keyPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CLIENT_KEY_FILE", defaultClientKeyFile)
	certPath := env.GetEnvAsStringOrFallback("FLOCKER_CONTROL_SERVICE_CLIENT_CERT_FILE", defaultClientCertFile)

	hostIP, err := b.plugin.host.GetHostIP()
	if err != nil {
		return nil, err
	}

	c, err := flockerclient.NewClient(host, port, hostIP.String(), caCertPath, keyPath, certPath)
	return c, err
}

/*
SetUpAt will setup a Flocker volume following this flow of calls to the Flocker
control service:

1. Get the dataset id for the given volume name/dir
2. It should already be there, if it's not the user needs to manually create it
3. Check the current Primary UUID
4. If it doesn't match with the Primary UUID that we got on 2, then we will
   need to update the Primary UUID for this volume.
5. Wait until the Primary UUID was updated or timeout.
*/
func (b flockerMounter) SetUpAt(dir string, fsGroup *int64) error {
	if b.client == nil {
		c, err := b.newFlockerClient()
		if err != nil {
			return err
		}
		b.client = c
	}

	datasetID, err := b.client.GetDatasetID(dir)
	if err != nil {
		return err
	}

	s, err := b.client.GetDatasetState(datasetID)
	if err != nil {
		return fmt.Errorf("The volume '%s' is not available in Flocker. You need to create this manually with Flocker CLI before using it.", dir)
	}

	primaryUUID, err := b.client.GetPrimaryUUID()
	if err != nil {
		return err
	}

	if s.Primary != primaryUUID {
		if err := b.updateDatasetPrimary(datasetID, primaryUUID); err != nil {
			return err
		}
		newState, err := b.client.GetDatasetState(datasetID)
		if err != nil {
			return fmt.Errorf("The volume '%s' migrated unsuccessfully.", datasetID)
		}
		b.flocker.path = newState.Path
	} else {
		b.flocker.path = s.Path
	}

	return nil
}

// updateDatasetPrimary will update the primary in Flocker and wait for it to
// be ready. If it never gets to ready state it will timeout and error.
func (b flockerMounter) updateDatasetPrimary(datasetID, primaryUUID string) error {
	// We need to update the primary and wait for it to be ready
	_, err := b.client.UpdatePrimaryForDataset(primaryUUID, datasetID)
	if err != nil {
		return err
	}

	timeoutChan := time.NewTimer(timeoutWaitingForVolume)
	defer timeoutChan.Stop()
	tickChan := time.NewTicker(tickerWaitingForVolume)
	defer tickChan.Stop()

	for {
		if s, err := b.client.GetDatasetState(datasetID); err == nil && s.Primary == primaryUUID {
			return nil
		}

		select {
		case <-timeoutChan.C:
			return fmt.Errorf(
				"Timed out waiting for the dataset_id: '%s' to be moved to the primary: '%s'\n%v",
				datasetID, primaryUUID, err,
			)
		case <-tickChan.C:
			break
		}
	}

}

func getVolumeSource(spec *volume.Spec) (*api.FlockerVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Flocker != nil {
		return spec.Volume.Flocker, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Flocker != nil {
		return spec.PersistentVolume.Spec.Flocker, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Flocker volume type")
}
