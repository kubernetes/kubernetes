Heapster Overview
===================

Heapster is a monitoring merics and events processing tool designed to work inside Kubernetes clusters. It consists of 2 components:

* Heapster core that reads [metrics](storage-schema.md) from Kubernetes cluster nodes (see [sources](source-configuration.md)), 
do some processing and writes them to permanent storage (see [sinks](sink-configuration.md)). 
It also provides metrics for other Kubernetes components through [Model API](model.md).

* Eventer that reads events from Kubernetes master (see [sources](source-configuration.md)) and writes them to permanent storage
(see [sinks](sink-configuration.md)).

