package local

import (
	// load the config
	_ "github.com/emccode/libstorage/imports/config"

	// load the libStorage storage driver
	_ "github.com/emccode/libstorage/drivers/storage/libstorage"

	// load the os drivers
	_ "github.com/emccode/libstorage/drivers/os/darwin"
	_ "github.com/emccode/libstorage/drivers/os/linux"

	// load the integration drivers
	_ "github.com/emccode/libstorage/drivers/integration/docker"

	// load the client drivers
	//_ "github.com/emccode/libstorage/drivers/storage/ec2/client"
	//_ "github.com/emccode/libstorage/drivers/storage/gce/client"
	//_ "github.com/emccode/libstorage/drivers/storage/isilon/client"
	// _ "github.com/emccode/libstorage/drivers/storage/mock/client"
	//_ "github.com/emccode/libstorage/drivers/storage/openstack/client"
	//_ "github.com/emccode/libstorage/drivers/storage/rackspace/client"
	// _ "github.com/emccode/libstorage/drivers/storage/scaleio"
	//_ "github.com/emccode/libstorage/drivers/storage/vbox/client"
	//_ "github.com/emccode/libstorage/drivers/storage/scaleio/client"
	_ "github.com/emccode/libstorage/drivers/storage/vfs/client"
	//_ "github.com/emccode/libstorage/drivers/storage/virtualbox"
	//_ "github.com/emccode/libstorage/drivers/storage/vmax/client"
	//_ "github.com/emccode/libstorage/drivers/storage/xtremio/client"
)
