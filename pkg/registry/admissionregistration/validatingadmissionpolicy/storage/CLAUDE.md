# Package: storage

This package provides REST storage for ValidatingAdmissionPolicy and its status subresource.

## Key Types

- `REST` - Main storage for ValidatingAdmissionPolicy
- `StatusREST` - Storage for the /status subresource

## Key Functions

- `NewREST()` - Creates both main and status storage
- `Categories()` - Returns ["api-extensions"]
- `Get()` - StatusREST method for retrieving policy
- `Update()` - StatusREST method for updating status

## Design Notes

- Returns two storage objects: one for spec, one for status
- Uses different strategies for spec vs status updates
- Status storage shares underlying store with main storage
- ResetFieldsStrategy ensures spec/status isolation
- Resource: "validatingadmissionpolicies"
- Status subresource: "validatingadmissionpolicies/status"
