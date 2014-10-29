package servers

import "github.com/rackspace/gophercloud"

// WaitForStatus will continually poll a server until it successfully transitions to a specified
// status. It will do this for at most the number of seconds specified.
func WaitForStatus(c *gophercloud.ServiceClient, id, status string, secs int) error {
	return gophercloud.WaitFor(secs, func() (bool, error) {
		current, err := Get(c, id).Extract()
		if err != nil {
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		return false, nil
	})
}
