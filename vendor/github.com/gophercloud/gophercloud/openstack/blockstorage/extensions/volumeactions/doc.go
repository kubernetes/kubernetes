/*
Package volumeactions provides information and interaction with volumes in the
OpenStack Block Storage service. A volume is a detachable block storage
device, akin to a USB hard drive.

Example of Attaching a Volume to an Instance

	attachOpts := volumeactions.AttachOpts{
		MountPoint:   "/mnt",
		Mode:         "rw",
		InstanceUUID: server.ID,
	}

	err := volumeactions.Attach(client, volume.ID, attachOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

	detachOpts := volumeactions.DetachOpts{
		AttachmentID: volume.Attachments[0].AttachmentID,
	}

	err = volumeactions.Detach(client, volume.ID, detachOpts).ExtractErr()
	if err != nil {
		panic(err)
	}


Example of Creating an Image from a Volume

	uploadImageOpts := volumeactions.UploadImageOpts{
		ImageName: "my_vol",
		Force:     true,
	}

	volumeImage, err := volumeactions.UploadImage(client, volume.ID, uploadImageOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", volumeImage)

Example of Extending a Volume's Size

	extendOpts := volumeactions.ExtendSizeOpts{
		NewSize: 100,
	}

	err := volumeactions.ExtendSize(client, volume.ID, extendOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

Example of Initializing a Volume Connection

	connectOpts := &volumeactions.InitializeConnectionOpts{
		IP:        "127.0.0.1",
		Host:      "stack",
		Initiator: "iqn.1994-05.com.redhat:17cf566367d2",
		Multipath: gophercloud.Disabled,
		Platform:  "x86_64",
		OSType:    "linux2",
	}

	connectionInfo, err := volumeactions.InitializeConnection(client, volume.ID, connectOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", connectionInfo["data"])

	terminateOpts := &volumeactions.InitializeConnectionOpts{
		IP:        "127.0.0.1",
		Host:      "stack",
		Initiator: "iqn.1994-05.com.redhat:17cf566367d2",
		Multipath: gophercloud.Disabled,
		Platform:  "x86_64",
		OSType:    "linux2",
	}

	err = volumeactions.TerminateConnection(client, volume.ID, terminateOpts).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package volumeactions
