/*
Package volumetenants provides the ability to extend a volume result with
tenant/project information. Example:

	type VolumeWithTenant struct {
		volumes.Volume
		volumetenants.VolumeTenantExt
	}

	var allVolumes []VolumeWithTenant

	allPages, err := volumes.List(client, nil).AllPages()
	if err != nil {
		panic("Unable to retrieve volumes: %s", err)
	}

	err = volumes.ExtractVolumesInto(allPages, &allVolumes)
	if err != nil {
		panic("Unable to extract volumes: %s", err)
	}

	for _, volume := range allVolumes {
		fmt.Println(volume.TenantID)
	}
*/
package volumetenants
