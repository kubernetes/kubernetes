# Clock

This package provides an interface for time-based operations.  It allows
mocking time for testing.

This is a copy of k8s.io/utils/clock. We have to copy it to avoid a circular
dependency (k8s.io/klog -> k8s.io/utils -> k8s.io/klog).
