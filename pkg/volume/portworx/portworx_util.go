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

	osdapi "github.com/libopenstorage/openstorage/api"
	osdclient "github.com/libopenstorage/openstorage/api/client"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	osdspec "github.com/libopenstorage/openstorage/api/spec"
	volumeapi "github.com/libopenstorage/openstorage/volume"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	osdMgmtDefaultPort = 9001
	osdDriverVersion   = "v1"
	pxdDriverName      = "pxd"
	pvcClaimLabel      = "pvc"
	pvcNamespaceLabel  = "namespace"
	pxServiceName      = "portworx-service"
	pxDriverName       = "pxd-sched"
)

type portworxVolumeUtil struct {
	portworxClient *osdclient.Client
}

// CreateVolume creates a Portworx volume.
func (util *portworxVolumeUtil) CreateVolume(p *portworxVolumeProvisioner) (string, int64, map[string]string, error) {
	driver, err := util.getPortworxDriver(p.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", 0, nil, err
	}

	klog.Infof("Creating Portworx volume for PVC: %v", p.options.PVC.Name)

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
			klog.Warningf("not saving annotation: %s=%s in spec labels due to an existing key", k, v)
			continue
		}
		spec.VolumeLabels[k] = v
	}

	volumeID, err := driver.Create(locator, source, spec)
	if err != nil {
		klog.Errorf("Error creating Portworx Volume : %v", err)
		return "", 0, nil, err
	}

	klog.Infof("Successfully created Portworx volume for PVC: %v", p.options.PVC.Name)
	return volumeID, requestGiB, nil, err
}

// DeleteVolume deletes a Portworx volume
func (util *portworxVolumeUtil) DeleteVolume(d *portworxVolumeDeleter) error {
	driver, err := util.getPortworxDriver(d.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Delete(d.volumeID)
	if err != nil {
		klog.Errorf("Error deleting Portworx Volume (%v): %v", d.volName, err)
		return err
	}
	return nil
}

// AttachVolume attaches a Portworx Volume
func (util *portworxVolumeUtil) AttachVolume(m *portworxVolumeMounter, attachOptions map[string]string) (string, error) {
	driver, err := util.getLocalPortworxDriver(m.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return "", err
	}

	devicePath, err := driver.Attach(m.volName, attachOptions)
	if err != nil {
		klog.Errorf("Error attaching Portworx Volume (%v): %v", m.volName, err)
		return "", err
	}
	return devicePath, nil
}

// DetachVolume detaches a Portworx Volume
func (util *portworxVolumeUtil) DetachVolume(u *portworxVolumeUnmounter) error {
	driver, err := util.getLocalPortworxDriver(u.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Detach(u.volName, false /*doNotForceDetach*/)
	if err != nil {
		klog.Errorf("Error detaching Portworx Volume (%v): %v", u.volName, err)
		return err
	}
	return nil
}

// MountVolume mounts a Portworx Volume on the specified mountPath
func (util *portworxVolumeUtil) MountVolume(m *portworxVolumeMounter, mountPath string) error {
	driver, err := util.getLocalPortworxDriver(m.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Mount(m.volName, mountPath)
	if err != nil {
		klog.Errorf("Error mounting Portworx Volume (%v) on Path (%v): %v", m.volName, mountPath, err)
		return err
	}
	return nil
}

// UnmountVolume unmounts a Portworx Volume
func (util *portworxVolumeUtil) UnmountVolume(u *portworxVolumeUnmounter, mountPath string) error {
	driver, err := util.getLocalPortworxDriver(u.plugin.host)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
		return err
	}

	err = driver.Unmount(u.volName, mountPath)
	if err != nil {
		klog.Errorf("Error unmounting Portworx Volume (%v) on Path (%v): %v", u.volName, mountPath, err)
		return err
	}
	return nil
}

func (util *portworxVolumeUtil) ResizeVolume(spec *volume.Spec, newSize resource.Quantity, volumeHost volume.VolumeHost) error {
	driver, err := util.getPortworxDriver(volumeHost)
	if err != nil || driver == nil {
		klog.Errorf("Failed to get portworx driver. Err: %v", err)
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
		klog.Infof("Portworx volume: %s already at size: %d greater than or equal to new "+
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
		klog.Errorf("portworx client failed driver versions check. Err: %v", err)
		return false, err
	}

	return true, nil
}

func createDriverClient(hostname string, port int32) (*osdclient.Client, error) {
	client, err := volumeclient.NewDriverClient(fmt.Sprintf("http://%s:%d", hostname, port),
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

// getPortworxDriver returns a Portworx volume driver which can be used for cluster wide operations.
//   Operations like create and delete volume don't need to be restricted to local volume host since
//   any node in the Portworx cluster can co-ordinate the create/delete request and forward the operations to
//   the Portworx node that will own/owns the data.
func (util *portworxVolumeUtil) getPortworxDriver(volumeHost volume.VolumeHost) (volumeapi.VolumeDriver, error) {
	// check if existing saved client is valid
	if isValid, _ := isClientValid(util.portworxClient); isValid {
		return volumeclient.VolumeDriver(util.portworxClient), nil
	}

	// create new client
	var err error
	util.portworxClient, err = createDriverClient(volumeHost.GetHostName(), osdMgmtDefaultPort) // for backward compatibility
	if err != nil || util.portworxClient == nil {
		// Create client from portworx k8s service.
		svc, err := getPortworxService(volumeHost)
		if err != nil {
			return nil, err
		}

		// The port here is always the default one since  it's the service port
		util.portworxClient, err = createDriverClient(svc.Spec.ClusterIP, osdMgmtDefaultPort)
		if err != nil || util.portworxClient == nil {
			klog.Errorf("Failed to connect to portworx service. Err: %v", err)
			return nil, err
		}

		klog.Infof("Using portworx cluster service at: %v:%d as api endpoint",
			svc.Spec.ClusterIP, osdMgmtDefaultPort)
	} else {
		klog.Infof("Using portworx service at: %v:%d as api endpoint",
			volumeHost.GetHostName(), osdMgmtDefaultPort)
	}

	return volumeclient.VolumeDriver(util.portworxClient), nil
}

// getLocalPortworxDriver returns driver connected to Portworx API server on volume host.
//   This is required to force certain operations (mount, unmount, detach, attach) to
//   go to the volume host instead of the k8s service which might route it to any host. This pertains to how
//   Portworx mounts and attaches a volume to the running container. The node getting these requests needs to
//   see the pod container mounts (specifically /var/lib/kubelet/pods/<pod_id>)
func (util *portworxVolumeUtil) getLocalPortworxDriver(volumeHost volume.VolumeHost) (volumeapi.VolumeDriver, error) {
	if util.portworxClient != nil {
		// check if existing saved client is valid
		if isValid, _ := isClientValid(util.portworxClient); isValid {
			return volumeclient.VolumeDriver(util.portworxClient), nil
		}
	}

	// Lookup port
	svc, err := getPortworxService(volumeHost)
	if err != nil {
		return nil, err
	}

	osgMgmtPort := lookupPXAPIPortFromService(svc)
	util.portworxClient, err = createDriverClient(volumeHost.GetHostName(), osgMgmtPort)
	if err != nil {
		return nil, err
	}

	klog.Infof("Using portworx local service at: %v:%d as api endpoint",
		volumeHost.GetHostName(), osgMgmtPort)
	return volumeclient.VolumeDriver(util.portworxClient), nil
}

// lookupPXAPIPortFromService goes over all the ports in the given service and returns the target
// port for osdMgmtDefaultPort
func lookupPXAPIPortFromService(svc *v1.Service) int32 {
	for _, p := range svc.Spec.Ports {
		if p.Port == osdMgmtDefaultPort {
			return p.TargetPort.IntVal
		}
	}
	return osdMgmtDefaultPort // default
}

// getPortworxService returns the portworx cluster service from the API server
func getPortworxService(host volume.VolumeHost) (*v1.Service, error) {
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		err := fmt.Errorf("Failed to get kubeclient when creating portworx client")
		klog.Errorf(err.Error())
		return nil, err
	}

	opts := metav1.GetOptions{}
	svc, err := kubeClient.CoreV1().Services(api.NamespaceSystem).Get(pxServiceName, opts)
	if err != nil {
		klog.Errorf("Failed to get service. Err: %v", err)
		return nil, err
	}

	if svc == nil {
		err = fmt.Errorf("Service: %v not found. Consult Portworx docs to deploy it.", pxServiceName)
		klog.Errorf(err.Error())
		return nil, err
	}

	return svc, nil
}
