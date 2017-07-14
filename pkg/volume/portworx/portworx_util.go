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
	"github.com/golang/glog"
	osdapi "github.com/libopenstorage/openstorage/api"
	osdclient "github.com/libopenstorage/openstorage/api/client"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	osdspec "github.com/libopenstorage/openstorage/api/spec"
	volumeapi "github.com/libopenstorage/openstorage/volume"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	osdMgmtPort      = "9001"
	osdDriverVersion = "v1"
	pxdDriverName    = "pxd"
	pvcClaimLabel    = "pvc"
	pxServiceName    = "portworx-service"
)

type PortworxVolumeUtil struct {
	portworxClient *osdclient.Client
}

// CreateVolume creates a Portworx volume.
func (util *PortworxVolumeUtil) CreateVolume(p *portworxVolumeProvisioner) (string, int, map[string]string, error) {
	driver, err := util.getPortworxDriver(p.plugin.host, false /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", 0, nil, err
	}

	glog.Infof("Creating Portworx volume for PVC: %v", p.options.PVC.Name)

	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// Portworx Volumes are specified in GB
	requestGB := int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024))

	specHandler := osdspec.NewSpecHandler()
	spec, err := specHandler.SpecFromOpts(p.options.Parameters)
	if err != nil {
		glog.Errorf("Error parsing parameters for PVC: %v. Err: %v", p.options.PVC.Name, err)
		return "", 0, nil, err
	}
	spec.Size = uint64(requestGB * 1024 * 1024 * 1024)
	source := osdapi.Source{}
	locator := osdapi.VolumeLocator{
		Name: p.options.PVName,
	}
	// Add claim Name as a part of Portworx Volume Labels
	locator.VolumeLabels = make(map[string]string)
	locator.VolumeLabels[pvcClaimLabel] = p.options.PVC.Name
	volumeID, err := driver.Create(&locator, &source, spec)
	if err != nil {
		glog.Errorf("Error creating Portworx Volume : %v", err)
	}

	glog.Infof("Successfully created Portworx volume for PVC: %v", p.options.PVC.Name)
	return volumeID, requestGB, nil, err
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
func (util *PortworxVolumeUtil) AttachVolume(m *portworxVolumeMounter) (string, error) {
	driver, err := util.getPortworxDriver(m.plugin.host, true /*localOnly*/)
	if err != nil || driver == nil {
		glog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", err
	}

	devicePath, err := driver.Attach(m.volName)
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

	err = driver.Detach(u.volName)
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
		pxdDriverName, osdDriverVersion)
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
