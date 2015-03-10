package fs

import "github.com/docker/libcontainer/cgroups"

type DevicesGroup struct {
}

func (s *DevicesGroup) Set(d *data) error {
	dir, err := d.join("devices")
	if err != nil {
		return err
	}

	if !d.c.AllowAllDevices {
		if err := writeFile(dir, "devices.deny", "a"); err != nil {
			return err
		}

		for _, dev := range d.c.AllowedDevices {
			if err := writeFile(dir, "devices.allow", dev.GetCgroupAllowString()); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *DevicesGroup) Remove(d *data) error {
	return removePath(d.path("devices"))
}

func (s *DevicesGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
