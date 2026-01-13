# Package: install

This package provides factory functions for creating resource quota configurations for admission control and controllers.

## Key Functions

- `NewQuotaConfigurationForAdmission()` - Creates quota configuration for admission webhooks
- `NewQuotaConfigurationForControllers()` - Creates quota configuration for the quota controller
- `DefaultIgnoredResources()` - Returns resources that should not be quota-controlled

## Ignored Resources

The following resources are ignored by default:
- `bindings` - Virtual resource for pod binding
- `componentstatuses` - Virtual resource
- `tokenreviews`, `selfsubjectreviews` - Authentication virtual resources
- `subjectaccessreviews` and variants - Authorization virtual resources
- `events` - Not traditionally quota-controlled

## Design Notes

- Admission configuration uses nil lister (evaluates objects directly)
- Controller configuration uses informer listers for efficiency
- Both configurations share the same set of core evaluators
- Extensible via the generic quota framework
