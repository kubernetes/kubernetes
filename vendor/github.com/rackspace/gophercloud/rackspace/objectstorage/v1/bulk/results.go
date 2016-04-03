package bulk

import (
	"github.com/rackspace/gophercloud"

	"github.com/mitchellh/mapstructure"
)

// DeleteResult represents the result of a bulk delete operation.
type DeleteResult struct {
	gophercloud.Result
}

// DeleteRespBody is the form of the response body returned by a bulk delete request.
type DeleteRespBody struct {
	NumberNotFound int      `mapstructure:"Number Not Found"`
	ResponseStatus string   `mapstructure:"Response Status"`
	Errors         []string `mapstructure:"Errors"`
	NumberDeleted  int      `mapstructure:"Number Deleted"`
	ResponseBody   string   `mapstructure:"Response Body"`
}

// ExtractBody will extract the body returned by the bulk extract request.
func (dr DeleteResult) ExtractBody() (DeleteRespBody, error) {
	var resp DeleteRespBody
	err := mapstructure.Decode(dr.Body, &resp)
	return resp, err
}
