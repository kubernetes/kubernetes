/*
Copyright 2017 The Kubernetes Authors.

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

package portworx

import (
	"fmt"

	"github.com/golang/glog"
	osdapi "github.com/libopenstorage/openstorage/api"
	osdclient "github.com/libopenstorage/openstorage/api/client"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	osdspec "github.com/libopenstorage/openstorage/api/spec"
	volumeapi "github.com/libopenstorage/openstorage/volume"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	osdMgmtPort       = "9001"
	osdDriverVersion  = "v1"
	pxdDriverName     = "pxd"
	pvcClaimLabel     = "pvc"
	pvcNamespaceLabel = "namespace"
	pxServiceName     = "portworx-service"
	pxDriverName      = "pxd-sched"
)

type PortworxVolumeUtil struct {
	portworxClient *osdclient.Client
}

// CreateVolume creates a Portworx volume.
func (util *PortworxVolumeUtil) CreateVolume(p *portworxVolumeProvisioner) (string, int64, map[string]string, error) {
	driver, err := util.getPortworxDriver(p.plugin.host, false /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", 0, nil, err
	}

	glog.Infof("Creating Portworx volume for PVC: %v", p.options.PVC.Name)

	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// Portworx Volumes are specified in GiB
	requestGiB := volutil.RoundUpToGiB(capacity)

	// Perform a best-effort parsing of parameters. Portworx 1.2.9 and later parses volume parameters from
	// spec.VolumeLabels. So even if below SpecFromOpts() fails to parse certain parameters or
	// doesn't support new parameters, the server-side processing will parse it correctly.
	// We still need to call SpecFromOpts() here to handle cases where someone is running Portworx 1.2.8 and lower.
	specHandler := osdspec.NewSpecHandler()
	spec, locator, source, _ := specHandler.SpecFromOpts(p.options.Parameters)
	if spec == nil {
		spec = specHandler.DefaultSpec()
	}

	// Pass all parameters as volume labels for Portworx server-side processing
	if len(p.options.Parameters) > 0 {
		spec.VolumeLabels = p.options.Parameters
	} else {
		spec.VolumeLabels = make(map[string]string, 0)
	}

	// Update the requested size in the spec
	spec.Size = uint64(requestGiB * volutil.GIB)

	// Change the Portworx Volume name to PV name
	if locator == nil {
		locator = &osdapi.VolumeLocator{
			VolumeLabels: make(map[string]string),
		}
	}
	locator.Name = p.options.PVName

	// Add claim Name as a part of Portworx Volume Labels
	locator.VolumeLabels[pvcClaimLabel] = p.options.PVC.Name
	locator.VolumeLabels[pvcNamespaceLabel] = p.options.PVC.Namespace

	for k, v := range p.options.PVC.Annotations {
		if _, present := spec.VolumeLabels[k]; present {
			glog.Warningf("not saving annotation: %s=%s in spec labels due to an existing key", k, v)
			continue
		}
		spec.VolumeLabels[k] = v
	}

	volumeID, err := driver.Create(locator, source, spec)
	if err != nil {
		glog.Errorf("Error creating Portworx Volume : %v", err)
		return "", 0, nil, err
	}

	glog.Infof("Successfully created Portworx volume for PVC: %v", p.options.PVC.Name)
	return volumeID, requestGiB, nil, err
}

// DeleteVolume deletes a Portworx volume
func (util *PortworxVolumeUtil) DeleteVolume(d *portworxVolumeDeleter) error {
	driver, err := util.getPortworxDriver(d.plugin.host, false /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Delete(d.volumeID)
	if err != nil {
		glog.Errorf("Error deleting Portworx Volume (%v): %v", d.volName, err)
		return err
	}
	return nil
}

// AttachVolume attaches a Portworx Volume
func (util *PortworxVolumeUtil) AttachVolume(m *portworxVolumeMounter, attachOptions map[string]string) (string, error) {
	driver, err := util.getPortworxDriver(m.plugin.host, true /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", err
	}

	devicePath, err := driver.Attach(m.volName, attachOptions)
	if err != nil {
		glog.Errorf("Error attaching Portworx Volume (%v): %v", m.volName, err)
		return "", err
	}
	return devicePath, nil
}

// DetachVolume detaches a Portworx Volume
func (util *PortworxVolumeUtil) DetachVolume(u *portworxVolumeUnmounter) error {
	driver, err := util.getPortworxDriver(u.plugin.host, true /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Detach(u.volName, false /*doNotForceDetach*/)
	if err != nil {
		glog.Errorf("Error detaching Portworx Volume (%v): %v", u.volName, err)
		return err
	}
	return nil
}

// MountVolume mounts a Portworx Volume on the specified mountPath
func (util *PortworxVolumeUtil) MountVolume(m *portworxVolumeMounter, mountPath string) error {
	driver, err := util.getPortworxDriver(m.plugin.host, true /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Mount(m.volName, mountPath)
	if err != nil {
		glog.Errorf("Error mounting Portworx Volume (%v) on Path (%v): %v", m.volName, mountPath, err)
		return err
	}
	return nil
}

// UnmountVolume unmounts a Portworx Volume
func (util *PortworxVolumeUtil) UnmountVolume(u *portworxVolumeUnmounter, mountPath string) error {
	driver, err := util.getPortworxDriver(u.plugin.host, true /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Unmount(u.volName, mountPath)
	if err != nil {
		glog.Errorf("Error unmounting Portworx Volume (%v) on Path (%v): %v", u.volName, mountPath, err)
		return err
	}
	return nil
}

func (util *PortworxVolumeUtil) ResizeVolume(spec *volume.Spec, newSize resource.Quantity, volumeHost volume.VolumeHost) error {
	driver, err := util.getPortworxDriver(volumeHost, false /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	vols, err := driver.Inspect([]string{spec.Name()})
	if err != nil {
		return err
	}

	if len(vols) != 1 {
		return fmt.Errorf("failed to inspect Portworx volume: %s. Found: %d volumes", spec.Name(), len(vols))
	}

	vol := vols[0]
	newSizeInBytes := uint64(volutil.RoundUpToGiB(newSize) * volutil.GIB)
	if vol.Spec.Size >= newSizeInBytes {
		glog.Infof("Portworx volume: %s already at size: %d greater than or equal to new "+
			"requested size: %d. Skipping resize.", spec.Name(), vol.Spec.Size, newSizeInBytes)
		return nil
	}

	vol.Spec.Size = newSizeInBytes
	err = driver.Set(spec.Name(), vol.Locator, vol.Spec)
	if err != nil {
		return err
	}

	// check if the volume's size actually got updated
	vols, err = driver.Inspect([]string{spec.Name()})
	if err != nil {
		return err
	}

	if len(vols) != 1 {
		return fmt.Errorf("failed to inspect resized Portworx volume: %s. Found: %d volumes", spec.Name(), len(vols))
	}

	updatedVol := vols[0]
	if updatedVol.Spec.Size < vol.Spec.Size {
		return fmt.Errorf("Portworx volume: %s doesn't match expected size after resize. expected:%v actual:%v",
			spec.Name(), vol.Spec.Size, updatedVol.Spec.Size)
	}

	return nil
}

func isClientValid(client *osdclient.Client) (bool, error) {
	if client == nil {
		return false, nil
	}

	_, err := client.Versions(osdapi.OsdVolumePath)
	if err != nil {
		glog.Errorf("portworx client failed driver versions check. Err: %v", err)
		return false, err
	}

	return true, nil
}

func createDriverClient(hostname string) (*osdclient.Client, error) {
	client, err := volumeclient.NewDriverClient("http://"+hostname+":"+osdMgmtPort,
		pxdDriverName, osdDriverVersion, pxDriverName)
	if err != nil {
		return nil, err
	}

	if isValid, err := isClientValid(client); isValid {
		return client, nil
	} else {
		return nil, err
	}
}

// getPortworxDriver() returns a Portworx volume driver which can be used for volume operations
// localOnly: If true, the returned driver will be connected to Portworx API server on volume host.
//            If false, driver will be connected to API server on volume host or Portworx k8s service cluster IP
//            This flag is required to explicitly force certain operations (mount, unmount, detach, attach) to
//            go to the volume host instead of the k8s service which might route it to any host. This pertains to how
//            Portworx mounts and attaches a volume to the running container. The node getting these requests needs to
//            see the pod container mounts (specifically /var/lib/kubelet/pods/<pod_id>)
//            Operations like create and delete volume don't need to be restricted to local volume host since
//            any node in the Portworx cluster can co-ordinate the create/delete request and forward the operations to
//            the Portworx node that will own/owns the data.
func (util *PortworxVolumeUtil) getPortworxDriver(volumeHost volume.VolumeHost, localOnly bool) (volumeapi.VolumeDriver, error) {
	var err error
	if localOnly {
		util.portworxClient, err = createDriverClient(volumeHost.GetHostName())
		if err != nil {
			return nil, err
		} else {
			glog.V(4).Infof("Using portworx local service at: %v as api endpoint", volumeHost.GetHostName())
			return volumeclient.VolumeDriver(util.portworxClient), nil
		}
	}

	// check if existing saved client is valid
	if isValid, _ := isClientValid(util.portworxClient); isValid {
		return volumeclient.VolumeDriver(util.portworxClient), nil
	}

	// create new client
	util.portworxClient, err = createDriverClient(volumeHost.GetHostName()) // for backward compatibility
	if err != nil || util.portworxClient == nil {
		// Create client from portworx service
		kubeClient := volumeHost.GetKubeClient()
		if kubeClient == nil {
			glog.Error("Failed to get kubeclient when creating portworx client")
			return nil, nil
		}

		opts := metav1.GetOptions{}
		svc, err := kubeClient.CoreV1().Services(api.NamespaceSystem).Get(pxServiceName, opts)
		if err != nil {
			glog.Errorf("Failed to get service. Err: %v", err)
			return nil, err
		}

		if svc == nil {
			glog.Errorf("Service: %v not found. Consult Portworx docs to deploy it.", pxServiceName)
			return nil, err
		}

		util.portworxClient, err = createDriverClient(svc.Spec.ClusterIP)
		if err != nil || util.portworxClient == nil {
			glog.Errorf("Failed to connect to portworx service. Err: %v", err)
			return nil, err
		}

		glog.Infof("Using portworx cluster service at: %v as api endpoint", svc.Spec.ClusterIP)
	} else {
		glog.Infof("Using portworx service at: %v as api endpoint", volumeHost.GetHostName())
	}

	return volumeclient.VolumeDriver(util.portworxClient), nil
}
