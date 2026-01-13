# Package: storage/util

## Purpose
Provides utility functions and constants for working with storage API types, particularly StorageClass annotations.

## Key Constants
- `IsDefaultStorageClassAnnotation`: "storageclass.kubernetes.io/is-default-class" - marks a StorageClass as default
- `BetaIsDefaultStorageClassAnnotation`: "storageclass.beta.kubernetes.io/is-default-class" - beta version of the default annotation

## Key Functions
- `IsDefaultAnnotation(obj metav1.ObjectMeta) bool`: Returns true if the object has either the GA or beta default storage class annotation set to "true"

## Design Notes
- Supports both GA and beta annotations for backwards compatibility
- The beta annotation is deprecated but still supported
- Used by admission controllers and storage provisioners to identify default storage classes
