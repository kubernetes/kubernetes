# Package: protectionutil

## Purpose
Provides utility functions for determining whether objects need finalizer protection and whether they are candidates for deletion.

## Key Functions

- **IsDeletionCandidate(obj, finalizer)**: Returns true if the object has a deletion timestamp set AND contains the specified finalizer. Used to identify objects that are being deleted but still protected by a finalizer.

- **NeedToAddFinalizer(obj, finalizer)**: Returns true if the object does NOT have a deletion timestamp AND does NOT already have the specified finalizer. Used to identify objects that should have protection added.

## Design Notes

- Used by PVC/PV protection controllers to manage deletion protection finalizers.
- Works with any metav1.Object (PVC, PV, or other resource types).
- Simple stateless utility functions - no state or caching.
