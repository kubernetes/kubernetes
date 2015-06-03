# Service Accounts
A serviceAccount provides an identity for processes that run in a Pod.
The behavior of the the serviceAccount object is implemented via a plugin
called an [Admission Controller]( admission_controllers.md). When this plugin is active
(and it is by default on most distributions), then it does the following when a pod is created or modified:
  1. If the pod does not have a ```ServiceAccount```, it modifies the pod's ```ServiceAccount``` to "default".
  2. It ensures that the ```ServiceAccount``` referenced by a pod exists.
  3. If ```LimitSecretReferences``` is true, it rejects the pod if the pod references ```Secret``` objects which the pods
```ServiceAccount``` does not reference.
  4. If the pod does not contain any ```ImagePullSecrets```, the ```ImagePullSecrets``` of the
```ServiceAccount``` are added to the pod.
  5. If ```MountServiceAccountToken``` is true, it adds a ```VolumeMount``` with the pod's ```ServiceAccount``` API token secret to containers in the pod.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/service_accounts.md?pixel)]()
