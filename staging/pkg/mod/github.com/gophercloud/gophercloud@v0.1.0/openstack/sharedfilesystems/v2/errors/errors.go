package errors

import (
	"encoding/json"
	"fmt"

	"github.com/gophercloud/gophercloud"
)

type ManilaError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Details string `json:"details"`
}

type ErrorDetails map[string]ManilaError

// error types from provider_client.go
func ExtractErrorInto(rawError error, errorDetails *ErrorDetails) (err error) {
	switch e := rawError.(type) {
	case gophercloud.ErrDefault400:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault401:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault403:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault404:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault405:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault408:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault429:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault500:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	case gophercloud.ErrDefault503:
		err = json.Unmarshal(e.ErrUnexpectedResponseCode.Body, errorDetails)
	default:
		err = fmt.Errorf("Unable to extract detailed error message")
	}

	return err
}
