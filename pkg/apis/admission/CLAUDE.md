# Package: admission

## Purpose
Defines the internal types for admission webhook request/response communication between the API server and admission webhooks.

## Key Types

### AdmissionReview
Top-level request/response wrapper containing:
- `Request *AdmissionRequest` - The admission request attributes
- `Response *AdmissionResponse` - The admission response

### AdmissionRequest
Contains all context about the request being reviewed:
- `UID` - Unique identifier for request/response correlation
- `Kind` / `Resource` / `SubResource` - What is being operated on
- `RequestKind` / `RequestResource` / `RequestSubResource` - Original request info (may differ if converted)
- `Name` / `Namespace` - Object identifiers
- `Operation` - CREATE, UPDATE, DELETE, or CONNECT
- `UserInfo` - Information about the requesting user
- `Object` / `OldObject` - The new and existing objects
- `DryRun` - If true, webhook must have no side effects
- `Options` - Operation options (CreateOptions, DeleteOptions, etc.)

### AdmissionResponse
Contains the webhook's decision:
- `UID` - Must match request UID
- `Allowed` - Whether the request is permitted
- `Result` - Details for denied requests
- `Patch` / `PatchType` - JSONPatch mutations (for mutating webhooks)
- `AuditAnnotations` - Key-value pairs added to audit log
- `Warnings` - Warning messages returned to client

## Constants
- `Operation`: Create, Update, Delete, Connect
- `PatchType`: PatchTypeJSONPatch
