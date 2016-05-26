package kubelet

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
)

// AddMountErrorHint performs some basic analysis
// on the current mount error returned and will
// add a user hint or resolution tip for enhanced UXP
func (kl *Kubelet) AddMountErrorHint(volpath string, volname string, inerr error) error{
	glog.V(4).Infof("Analyzing mount error and adding user resolution hint")

	//General - applies to all
	if strings.Contains(inerr.Error(), "lstat") && strings.Contains(inerr.Error(), "permission denied"){
		return fmt.Errorf("%v\nResolution hint: (%s) The pod is running, and the mount succeeded, however the mount is not accessbile due to permissions.\nCheck the POSIX based permissions (owner, groups and others) on your mounted directory.\nIf needed containers and pods can utilize and pass in a securityContext specifying runAsUser (uid/owner), or additional linux groups such as fsGroup (for block) or SupplementalGroups (for shared).\nWork with the storage adminstrator to properly set up access\n", inerr, volname)
	}
	if strings.Contains(inerr.Error(), "endpoints") && strings.Contains(inerr.Error(), "not found") {
		return fmt.Errorf("%v\nResolution hint: (%s) Make sure the above endpoint exists. To persist endpoints, they should be created as a service.\n", inerr, volname)
	}
	if strings.Contains(inerr.Error(), "secrets") && strings.Contains(inerr.Error(), "missing") {
		return fmt.Errorf("%v\nResolution hint: (%s) Make sure the above secret exists.  Secret is needed for rbd mount and access\n", inerr, volname)
	}


	// NFS
	if strings.Contains(volpath, "kubernetes.io~nfs"){
		if strings.Contains(inerr.Error(), "access denied by server") {
			return fmt.Errorf("%v\nResolution hint: (%s) Check the NFS Server exports, likely that the host/node was not added. (/etc/exports).  Rerun exportfs -ra on NFS Server after updated.\n", inerr, volname)
		}
		if strings.Contains(inerr.Error(), "Connection timed out"){
			return fmt.Errorf("%v\nResolution hint: (%s) Check and make sure the NFS Server exists (ensure that correct IPAddress/Hostname was given) and is available/reachable.\n Also make sure firewall ports are open on both client and NFS Server (2049 v4 and 2049, 20048 and 111 for v3)\ntelnet <nfs server> <port> and showmount <nfs server> are useful commands to try.\n", inerr, volname)
		}
		if strings.Contains(inerr.Error(), "wrong fs type, bad option, bad superblock"){
			return fmt.Errorf("%v\nResolution hint: (%s) This typically means that the nfs client packages are not installed on the host/node (nfsutils and rpcbind).  Make sure they are installed and running on your host client\n", inerr, volname)
		}
	}

	// Glusterfs (this depends on PR 24808 to read proper root cause error from log)
	if strings.Contains(volpath, "kubernetes.io~glusterfs"){
		if strings.Contains(inerr.Error(), "Connection timed out") || strings.Contains(inerr.Error(), "Transport endpoint is not connected"){
			return fmt.Errorf("%v\nResolution hint: (%s) Check and make sure the Gluster Server exists (ensure that correct IPAddress/Hostname was given in the endpoints) and is available/reachable.\n", inerr, volname)
		}
		if strings.Contains(inerr.Error(), "mount: unknown filesystem type"){
			return fmt.Errorf("%v\nResolution hint: (%s) Check and make sure the glusterfs-client package is installed (rpm -qa 'gluster*') on your nodes.\nIf not, install the client package on your nodes (i.e. yum install glusterfs-client -y).\n", inerr, volname)
		}
		// this should only get hit if 24808 doesn't merge or an undefined error happens
		if strings.Contains(inerr.Error(), "Mount failed. Please check the log"){
			return fmt.Errorf("%v\nResolution hint: (%s) Open the gluster log file, you can see this above in the mount arguments from the error (i.e. --log-file=<path>.\n", inerr, volname)
		}
	}

	// AWS
	if strings.Contains(volpath, "kubernetes.io~aws-ebs"){
		if strings.Contains(inerr.Error(), "InvalidVolume.NotFound") {
			return fmt.Errorf("%v\nResolution hint: (%s) Check AWS available volumes for the appropriate availability zone, and make sure the specified volumeID exists and is spelled correctly.\n", inerr, volname)
		}
		if strings.Contains(inerr.Error(), "InvalidParameterValue") && strings.Contains(inerr.Error(), "volume is invalid. Expected: 'vol-") {
			return fmt.Errorf("%v\nResolution hint: (%s) Check AWS available volumes, make sure the specified volumeID exists, is spelled and typed correctly and is valid format.\n", inerr, volname)
		}
		if strings.Contains(inerr.Error(), "VolumeInUse:") || strings.Contains(inerr.Error(), "is already attached to an instance") {
			return fmt.Errorf("%v\nResolution hint: (%s) The AWS volume is already attached to another instance and only one node per volume is allowed for EBS block devices (can not share across nodes). Another volume will need to be provisioned for use with this pod\n", inerr, volname)
		}
	}


	//GCE



	//RBD



	// TODO: rest of plugins as errors are reported
	return inerr
}
