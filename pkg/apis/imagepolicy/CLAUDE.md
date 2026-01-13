# Package: imagepolicy

## Purpose
Internal (unversioned) API types for the imagepolicy.k8s.io API group, used by the Image Policy Webhook admission controller to validate container images.

## Key Types

- **ImageReview**: Request/response object for image policy webhook.
- **ImageReviewSpec**: Contains containers (images), annotations (*.image-policy.k8s.io/*), and namespace.
- **ImageReviewContainerSpec**: Contains image reference (image:tag or image@SHA:...).
- **ImageReviewStatus**: Contains Allowed boolean, Reason string, and AuditAnnotations.

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers ImageReview type.

## Key Constants

- **GroupName**: "imagepolicy.k8s.io"

## Design Notes

- Used by ImagePolicyWebhook admission controller.
- Webhook receives ImageReviewSpec and returns ImageReviewStatus.
- Annotations with *.image-policy.k8s.io/* prefix are passed to webhook.
- AuditAnnotations in response are added to admission audit log.
