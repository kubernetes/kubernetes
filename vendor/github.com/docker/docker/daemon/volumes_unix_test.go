// +build !windows

package daemon

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	containertypes "github.com/docker/docker/api/types/container"
	mounttypes "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/container"
	"github.com/docker/docker/volume"
)

func TestBackportMountSpec(t *testing.T) {
	d := Daemon{containers: container.NewMemoryStore()}

	c := &container.Container{
		State: &container.State{},
		MountPoints: map[string]*volume.MountPoint{
			"/apple":      {Destination: "/apple", Source: "/var/lib/docker/volumes/12345678", Name: "12345678", RW: true, CopyData: true}, // anonymous volume
			"/banana":     {Destination: "/banana", Source: "/var/lib/docker/volumes/data", Name: "data", RW: true, CopyData: true},        // named volume
			"/cherry":     {Destination: "/cherry", Source: "/var/lib/docker/volumes/data", Name: "data", CopyData: true},                  // RO named volume
			"/dates":      {Destination: "/dates", Source: "/var/lib/docker/volumes/data", Name: "data"},                                   // named volume nocopy
			"/elderberry": {Destination: "/elderberry", Source: "/var/lib/docker/volumes/data", Name: "data"},                              // masks anon vol
			"/fig":        {Destination: "/fig", Source: "/data", RW: true},                                                                // RW bind
			"/guava":      {Destination: "/guava", Source: "/data", RW: false, Propagation: "shared"},                                      // RO bind + propagation
			"/kumquat":    {Destination: "/kumquat", Name: "data", RW: false, CopyData: true},                                              // volumes-from

			// partially configured mountpoint due to #32613
			// specifically, `mp.Spec.Source` is not set
			"/honeydew": {
				Type:        mounttypes.TypeVolume,
				Destination: "/honeydew",
				Name:        "data",
				Source:      "/var/lib/docker/volumes/data",
				Spec:        mounttypes.Mount{Type: mounttypes.TypeVolume, Target: "/honeydew", VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true}},
			},

			// from hostconfig.Mounts
			"/jambolan": {
				Type:        mounttypes.TypeVolume,
				Destination: "/jambolan",
				Source:      "/var/lib/docker/volumes/data",
				RW:          true,
				Name:        "data",
				Spec:        mounttypes.Mount{Type: mounttypes.TypeVolume, Target: "/jambolan", Source: "data"},
			},
		},
		HostConfig: &containertypes.HostConfig{
			Binds: []string{
				"data:/banana",
				"data:/cherry:ro",
				"data:/dates:ro,nocopy",
				"data:/elderberry:ro,nocopy",
				"/data:/fig",
				"/data:/guava:ro,shared",
				"data:/honeydew:nocopy",
			},
			VolumesFrom: []string{"1:ro"},
			Mounts: []mounttypes.Mount{
				{Type: mounttypes.TypeVolume, Target: "/jambolan"},
			},
		},
		Config: &containertypes.Config{Volumes: map[string]struct{}{
			"/apple":      {},
			"/elderberry": {},
		}},
	}

	d.containers.Add("1", &container.Container{
		State: &container.State{},
		ID:    "1",
		MountPoints: map[string]*volume.MountPoint{
			"/kumquat": {Destination: "/kumquat", Name: "data", RW: false, CopyData: true},
		},
		HostConfig: &containertypes.HostConfig{
			Binds: []string{
				"data:/kumquat:ro",
			},
		},
	})

	type expected struct {
		mp      *volume.MountPoint
		comment string
	}

	pretty := func(mp *volume.MountPoint) string {
		b, err := json.MarshalIndent(mp, "\t", "    ")
		if err != nil {
			return fmt.Sprintf("%#v", mp)
		}
		return string(b)
	}

	for _, x := range []expected{
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/apple",
				RW:          true,
				Name:        "12345678",
				Source:      "/var/lib/docker/volumes/12345678",
				CopyData:    true,
				Spec: mounttypes.Mount{
					Type:   mounttypes.TypeVolume,
					Source: "",
					Target: "/apple",
				},
			},
			comment: "anonymous volume",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/banana",
				RW:          true,
				Name:        "data",
				Source:      "/var/lib/docker/volumes/data",
				CopyData:    true,
				Spec: mounttypes.Mount{
					Type:   mounttypes.TypeVolume,
					Source: "data",
					Target: "/banana",
				},
			},
			comment: "named volume",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/cherry",
				Name:        "data",
				Source:      "/var/lib/docker/volumes/data",
				CopyData:    true,
				Spec: mounttypes.Mount{
					Type:     mounttypes.TypeVolume,
					Source:   "data",
					Target:   "/cherry",
					ReadOnly: true,
				},
			},
			comment: "read-only named volume",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/dates",
				Name:        "data",
				Source:      "/var/lib/docker/volumes/data",
				Spec: mounttypes.Mount{
					Type:          mounttypes.TypeVolume,
					Source:        "data",
					Target:        "/dates",
					ReadOnly:      true,
					VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true},
				},
			},
			comment: "named volume with nocopy",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/elderberry",
				Name:        "data",
				Source:      "/var/lib/docker/volumes/data",
				Spec: mounttypes.Mount{
					Type:          mounttypes.TypeVolume,
					Source:        "data",
					Target:        "/elderberry",
					ReadOnly:      true,
					VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true},
				},
			},
			comment: "masks an anonymous volume",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeBind,
				Destination: "/fig",
				Source:      "/data",
				RW:          true,
				Spec: mounttypes.Mount{
					Type:   mounttypes.TypeBind,
					Source: "/data",
					Target: "/fig",
				},
			},
			comment: "bind mount with read/write",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeBind,
				Destination: "/guava",
				Source:      "/data",
				RW:          false,
				Propagation: "shared",
				Spec: mounttypes.Mount{
					Type:        mounttypes.TypeBind,
					Source:      "/data",
					Target:      "/guava",
					ReadOnly:    true,
					BindOptions: &mounttypes.BindOptions{Propagation: "shared"},
				},
			},
			comment: "bind mount with read/write + shared propagation",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/honeydew",
				Source:      "/var/lib/docker/volumes/data",
				RW:          true,
				Propagation: "shared",
				Spec: mounttypes.Mount{
					Type:          mounttypes.TypeVolume,
					Source:        "data",
					Target:        "/honeydew",
					VolumeOptions: &mounttypes.VolumeOptions{NoCopy: true},
				},
			},
			comment: "partially configured named volume caused by #32613",
		},
		{
			mp:      &(*c.MountPoints["/jambolan"]), // copy the mountpoint, expect no changes
			comment: "volume defined in mounts API",
		},
		{
			mp: &volume.MountPoint{
				Type:        mounttypes.TypeVolume,
				Destination: "/kumquat",
				Source:      "/var/lib/docker/volumes/data",
				RW:          false,
				Name:        "data",
				Spec: mounttypes.Mount{
					Type:     mounttypes.TypeVolume,
					Source:   "data",
					Target:   "/kumquat",
					ReadOnly: true,
				},
			},
			comment: "partially configured named volume caused by #32613",
		},
	} {

		mp := c.MountPoints[x.mp.Destination]
		d.backportMountSpec(c)

		if !reflect.DeepEqual(mp.Spec, x.mp.Spec) {
			t.Fatalf("%s\nexpected:\n\t%s\n\ngot:\n\t%s", x.comment, pretty(x.mp), pretty(mp))
		}
	}
}
