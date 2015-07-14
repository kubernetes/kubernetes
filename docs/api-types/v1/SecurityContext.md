###SecurityContext###

---
* capabilities: 
  * **_type_**: [Capabilities](Capabilities.md)
  * **_description_**: the linux capabilites that should be added or removed; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* privileged: 
  * **_type_**: boolean
  * **_description_**: run the container in privileged mode; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* runAsUser: 
  * **_type_**: integer
  * **_description_**: the user id that runs the first process in the container; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* seLinuxOptions: 
  * **_type_**: [SELinuxOptions](SELinuxOptions.md)
  * **_description_**: options that control the SELinux labels applied; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
