package extendedserverattributes

import (
	"github.com/gophercloud/gophercloud"
)

// ExtractReservationID will extract the reservation_id attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractReservationID(r gophercloud.Result) (string, error) {
	var s struct {
		ReservationID string `json:"OS-EXT-SRV-ATTR:reservation_id"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.ReservationID, err
}

// ExtractLaunchIndex will extract the launch_index attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractLaunchIndex(r gophercloud.Result) (int, error) {
	var s struct {
		LaunchIndex int `json:"OS-EXT-SRV-ATTR:launch_index"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.LaunchIndex, err
}

// ExtractRamdiskID will extract the ramdisk_id attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractRamdiskID(r gophercloud.Result) (string, error) {
	var s struct {
		RamdiskID string `json:"OS-EXT-SRV-ATTR:ramdisk_id"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.RamdiskID, err
}

// ExtractKernelID will extract the kernel_id attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractKernelID(r gophercloud.Result) (string, error) {
	var s struct {
		KernelID string `json:"OS-EXT-SRV-ATTR:kernel_id"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.KernelID, err
}

// ExtractHostname will extract the hostname attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractHostname(r gophercloud.Result) (string, error) {
	var s struct {
		Hostname string `json:"OS-EXT-SRV-ATTR:hostname"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.Hostname, err
}

// ExtractRootDeviceName will extract the root_device_name attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractRootDeviceName(r gophercloud.Result) (string, error) {
	var s struct {
		RootDeviceName string `json:"OS-EXT-SRV-ATTR:root_device_name"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.RootDeviceName, err
}

// ExtractUserData will extract the userdata attribute.
// This requires the client to be set to microversion 2.3 or later.
func ExtractUserData(r gophercloud.Result) (string, error) {
	var s struct {
		Userdata string `json:"OS-EXT-SRV-ATTR:userdata"`
	}
	err := r.ExtractIntoStructPtr(&s, "server")

	return s.Userdata, err
}
