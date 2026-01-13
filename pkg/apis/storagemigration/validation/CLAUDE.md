# Package: storagemigration/validation

## Purpose
Provides validation logic for StorageVersionMigration API types.

## Key Functions
- `ValidateStorageVersionMigration(svm)`: Validates StorageVersionMigration creation including resource name and group
- `ValidateStorageVersionMigrationUpdate(new, old)`: Validates updates, ensuring spec is immutable after creation
- `ValidateStorageVersionMigrationStatusUpdate(new, old)`: Validates status updates with complex condition rules

## Key Validation Rules
- Resource field is required and must be a valid DNS1035 label
- Group must be a valid DNS1123 subdomain if specified
- Spec is immutable after creation
- ResourceVersion in status is immutable once set
- Success and Failed conditions are mutually exclusive
- Running cannot be true when Success or Failed is true
- Success/Failed conditions cannot be set back to false once true

## Design Notes
- Uses field.ErrorList pattern for accumulating validation errors
- Validates condition state machine transitions to ensure valid migration lifecycle
- Leverages metav1validation for condition validation
