/*
Package schedulerhints extends the volume create request with the ability to
specify additional parameters which determine where the volume will be
created in the OpenStack cloud.

Example to Place Volume B on a Different Host than Volume A

	schedulerHints := schedulerhints.SchedulerHints{
		DifferentHost: []string{
			"volume-a-uuid",
		}
	}

	volumeCreateOpts := volumes.CreateOpts{
		Name:   "volume_b",
		Size:   10,
	}

	createOpts := schedulerhints.CreateOptsExt{
		VolumeCreateOptsBuilder: volumeCreateOpts,
		SchedulerHints:    schedulerHints,
	}

	volume, err := volumes.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Place Volume B on the Same Host as Volume A

	schedulerHints := schedulerhints.SchedulerHints{
		SameHost: []string{
			"volume-a-uuid",
		}
	}

	volumeCreateOpts := volumes.CreateOpts{
		Name:   "volume_b",
		Size:   10
	}

	createOpts := schedulerhints.CreateOptsExt{
		VolumeCreateOptsBuilder: volumeCreateOpts,
		SchedulerHints:    schedulerHints,
	}

	volume, err := volumes.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package schedulerhints
