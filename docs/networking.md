# Networking
Kubernetes gives every pod its own IP address allocated from an internal network, so you do not need to explicitly create links between communicating pods.
However, since pods can fail and be scheduled to different nodes, we do not recommend having a pod directly talk to the IP address of another Pod.  Instead, if a pod, or collection of pods, provide some service, then you should create a `service` object spanning those pods, and clients should connect to the IP of the service object.  See [services](services.md).

The networking model and its rationale, and our future plans are described in more detail in the [networking design document](design/networking.md).
