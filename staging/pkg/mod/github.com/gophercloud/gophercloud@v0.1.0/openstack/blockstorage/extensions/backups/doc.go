/*
Package backups provides information and interaction with backups in the
OpenStack Block Storage service. A backup is a point in time copy of the
data contained in an external storage volume, and can be controlled
programmatically.

Example to List Backups

	listOpts := backups.ListOpts{
		VolumeID: "uuid",
	}

	allPages, err := backups.List(client, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allBackups, err := backups.ExtractBackups(allPages)
	if err != nil {
		panic(err)
	}

	for _, backup := range allBackups {
		fmt.Println(backup)
	}

Example to Create a Backup

	createOpts := backups.CreateOpts{
		VolumeID: "uuid",
		Name:     "my-backup",
	}

	backup, err := backups.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Println(backup)

Example to Update a Backup

	updateOpts := backups.UpdateOpts{
		Name: "new-name",
	}

	backup, err := backups.Update(client, "uuid", updateOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Println(backup)

Example to Delete a Backup

	err := backups.Delete(client, "uuid").ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package backups
