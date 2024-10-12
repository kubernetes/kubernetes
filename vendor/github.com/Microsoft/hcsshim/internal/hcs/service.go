//go:build windows

package hcs

import (
	"context"
	"encoding/json"

	hcsschema "github.com/Microsoft/hcsshim/internal/hcs/schema2"
	"github.com/Microsoft/hcsshim/internal/vmcompute"
)

// GetServiceProperties returns properties of the host compute service.
func GetServiceProperties(ctx context.Context, q hcsschema.PropertyQuery) (*hcsschema.ServiceProperties, error) {
	operation := "hcs::GetServiceProperties"

	queryb, err := json.Marshal(q)
	if err != nil {
		return nil, err
	}
	propertiesJSON, resultJSON, err := vmcompute.HcsGetServiceProperties(ctx, string(queryb))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, &HcsError{Op: operation, Err: err, Events: events}
	}

	if propertiesJSON == "" {
		return nil, ErrUnexpectedValue
	}
	properties := &hcsschema.ServiceProperties{}
	if err := json.Unmarshal([]byte(propertiesJSON), properties); err != nil {
		return nil, err
	}
	return properties, nil
}

// ModifyServiceSettings modifies settings of the host compute service.
func ModifyServiceSettings(ctx context.Context, settings hcsschema.ModificationRequest) error {
	operation := "hcs::ModifyServiceSettings"

	settingsJSON, err := json.Marshal(settings)
	if err != nil {
		return err
	}
	resultJSON, err := vmcompute.HcsModifyServiceSettings(ctx, string(settingsJSON))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return &HcsError{Op: operation, Err: err, Events: events}
	}
	return nil
}
