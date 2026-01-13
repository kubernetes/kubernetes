# Package: util

## Purpose
Provides utility functions for the networking API group.

## Key Functions
- `HasDefaultAnnotation(obj metav1.ObjectMeta) bool`: Checks if an object has the `ingressclass.kubernetes.io/is-default-class` annotation set to "true"

## Notes
- Used to identify the default IngressClass when multiple IngressClasses exist
- The annotation `networkingv1.AnnotationIsDefaultIngressClass` determines which IngressClass should be used when an Ingress doesn't specify a class
