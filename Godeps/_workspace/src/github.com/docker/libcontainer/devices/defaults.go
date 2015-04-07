package devices

var (
	// These are devices that are to be both allowed and created.

	DefaultSimpleDevices = []*Device{
		// /dev/null and zero
		{
			Path:              "/dev/null",
			Type:              'c',
			MajorNumber:       1,
			MinorNumber:       3,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},
		{
			Path:              "/dev/zero",
			Type:              'c',
			MajorNumber:       1,
			MinorNumber:       5,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},

		{
			Path:              "/dev/full",
			Type:              'c',
			MajorNumber:       1,
			MinorNumber:       7,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},

		// consoles and ttys
		{
			Path:              "/dev/tty",
			Type:              'c',
			MajorNumber:       5,
			MinorNumber:       0,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},

		// /dev/urandom,/dev/random
		{
			Path:              "/dev/urandom",
			Type:              'c',
			MajorNumber:       1,
			MinorNumber:       9,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},
		{
			Path:              "/dev/random",
			Type:              'c',
			MajorNumber:       1,
			MinorNumber:       8,
			CgroupPermissions: "rwm",
			FileMode:          0666,
		},
	}

	DefaultAllowedDevices = append([]*Device{
		// allow mknod for any device
		{
			Type:              'c',
			MajorNumber:       Wildcard,
			MinorNumber:       Wildcard,
			CgroupPermissions: "m",
		},
		{
			Type:              'b',
			MajorNumber:       Wildcard,
			MinorNumber:       Wildcard,
			CgroupPermissions: "m",
		},

		{
			Path:              "/dev/console",
			Type:              'c',
			MajorNumber:       5,
			MinorNumber:       1,
			CgroupPermissions: "rwm",
		},
		{
			Path:              "/dev/tty0",
			Type:              'c',
			MajorNumber:       4,
			MinorNumber:       0,
			CgroupPermissions: "rwm",
		},
		{
			Path:              "/dev/tty1",
			Type:              'c',
			MajorNumber:       4,
			MinorNumber:       1,
			CgroupPermissions: "rwm",
		},
		// /dev/pts/ - pts namespaces are "coming soon"
		{
			Path:              "",
			Type:              'c',
			MajorNumber:       136,
			MinorNumber:       Wildcard,
			CgroupPermissions: "rwm",
		},
		{
			Path:              "",
			Type:              'c',
			MajorNumber:       5,
			MinorNumber:       2,
			CgroupPermissions: "rwm",
		},

		// tuntap
		{
			Path:              "",
			Type:              'c',
			MajorNumber:       10,
			MinorNumber:       200,
			CgroupPermissions: "rwm",
		},

		/*// fuse
		   {
		    Path: "",
		    Type: 'c',
		    MajorNumber: 10,
		    MinorNumber: 229,
		    CgroupPermissions: "rwm",
		   },

		// rtc
		   {
		    Path: "",
		    Type: 'c',
		    MajorNumber: 254,
		    MinorNumber: 0,
		    CgroupPermissions: "rwm",
		   },
		*/
	}, DefaultSimpleDevices...)

	DefaultAutoCreatedDevices = append([]*Device{
		{
			// /dev/fuse is created but not allowed.
			// This is to allow java to work.  Because java
			// Insists on there being a /dev/fuse
			// https://github.com/docker/docker/issues/514
			// https://github.com/docker/docker/issues/2393
			//
			Path:              "/dev/fuse",
			Type:              'c',
			MajorNumber:       10,
			MinorNumber:       229,
			CgroupPermissions: "rwm",
		},
	}, DefaultSimpleDevices...)
)
