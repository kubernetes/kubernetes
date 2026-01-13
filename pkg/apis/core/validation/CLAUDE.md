# Package: validation

## Purpose
Comprehensive validation logic for internal core API types. This is the primary validation package for all core Kubernetes resources.

## Key Functions

### Common Validation
- **ValidateAnnotations(annotations, fldPath)**: Validates annotation key/value format.
- **ValidateHasLabel(meta, fldPath, key, expectedValue)**: Ensures required label exists.
- **ValidateQualifiedName(value, fldPath)**: Validates qualified names (DNS subdomain with optional prefix).
- **ValidateImmutableField(newVal, oldVal, fldPath)**: Ensures field hasn't changed.

### Pod Validation
- **ValidatePod(pod)**: Full pod validation.
- **ValidatePodSpec(spec, fldPath, opts)**: Validates PodSpec.
- **ValidatePodUpdate(newPod, oldPod)**: Validates pod updates with immutability checks.
- **ValidateContainers(containers, fldPath, opts)**: Validates container specs.

### Service Validation
- **ValidateService(service)**: Full service validation.
- **ValidateServiceUpdate(newService, oldService)**: Service update validation.

### Node Validation
- **ValidateNode(node)**: Full node validation.
- **ValidateNodeUpdate(newNode, oldNode)**: Node update validation.

### Storage Validation
- **ValidatePersistentVolume(pv)**: PV validation.
- **ValidatePersistentVolumeClaim(pvc)**: PVC validation.

### Resource Validation
- **ValidateResourceQuota(resourceQuota)**: Quota validation.
- **ValidateLimitRange(limitRange)**: LimitRange validation.
- **ValidateConfigMap(configMap)**: ConfigMap validation.
- **ValidateSecret(secret)**: Secret validation with type-specific rules.

## Design Notes

- Uses field.ErrorList for error accumulation.
- Extensive use of feature gates for conditional validation.
- BannedOwners prevents certain owner references.
- Supports create vs update validation modes.
- Heavy use of regex for format validation (IQN, NAA, EUI for iSCSI, etc.).
