package evacuate

import (
	"github.com/gophercloud/gophercloud"
)

// EvacuateResult is the response from an Evacuate operation.
//Call its ExtractAdminPass method to retrieve the admin password of the instance.
//The admin password will be an empty string if the cloud is not configured to inject admin passwords..
type EvacuateResult struct {
	gophercloud.Result
}

func (r EvacuateResult) ExtractAdminPass() (string, error) {
	var s struct {
		AdminPass string `json:"adminPass"`
	}
	err := r.ExtractInto(&s)
	if err != nil && err.Error() == "EOF" {
		return "", nil
	}
	return s.AdminPass, err
}
