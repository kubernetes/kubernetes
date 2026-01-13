# Package: fuzzer

## Purpose
Provides fuzzer functions for the rbac API group to generate valid random test data for fuzz testing.

## Key Functions

### Funcs
Returns fuzzer functions for RBAC types:

- **RoleRef fuzzer**: Defaults empty APIGroup to "rbac.authorization.k8s.io" to match defaulter behavior

- **Subject fuzzer**: Randomly generates one of three subject types:
  - ServiceAccount: Kind=ServiceAccount, APIGroup="", with random name and namespace
  - User: Kind=User, APIGroup=rbac.GroupName, avoids "*" name (special case)
  - Group: Kind=Group, APIGroup=rbac.GroupName

## Notes
- User "*" is explicitly avoided because it converts to system:authenticated group during round-trip
- Ensures fuzzed objects have correct APIGroup values for their subject kind
- Critical for testing RBAC API stability with random inputs
