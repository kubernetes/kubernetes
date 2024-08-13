// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

// LINT.IfChange
const (
	nameReferenceFieldSpecs = `
nameReference:
- kind: Deployment
  fieldSpecs:
  - path: spec/scaleTargetRef/name
    kind: HorizontalPodAutoscaler

- kind: ReplicationController
  fieldSpecs:
  - path: spec/scaleTargetRef/name
    kind: HorizontalPodAutoscaler

- kind: ReplicaSet
  fieldSpecs:
  - path: spec/scaleTargetRef/name
    kind: HorizontalPodAutoscaler

- kind: StatefulSet
  fieldSpecs:
  - path: spec/scaleTargetRef/name
    kind: HorizontalPodAutoscaler

- kind: ConfigMap
  version: v1
  fieldSpecs:
  - path: spec/volumes/configMap/name
    version: v1
    kind: Pod
  - path: spec/containers/env/valueFrom/configMapKeyRef/name
    version: v1
    kind: Pod
  - path: spec/initContainers/env/valueFrom/configMapKeyRef/name
    version: v1
    kind: Pod
  - path: spec/containers/envFrom/configMapRef/name
    version: v1
    kind: Pod
  - path: spec/initContainers/envFrom/configMapRef/name
    version: v1
    kind: Pod
  - path: spec/volumes/projected/sources/configMap/name
    version: v1
    kind: Pod
  - path: template/spec/volumes/configMap/name
    kind: PodTemplate
  - path: template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: PodTemplate
  - path: template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: PodTemplate
  - path: template/spec/containers/envFrom/configMapRef/name
    kind: PodTemplate
  - path: template/spec/initContainers/envFrom/configMapRef/name
    kind: PodTemplate
  - path: template/spec/volumes/projected/sources/configMap/name
    kind: PodTemplate
  - path: spec/template/spec/volumes/configMap/name
    kind: Deployment
  - path: spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: Deployment
  - path: spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: Deployment
  - path: spec/template/spec/containers/envFrom/configMapRef/name
    kind: Deployment
  - path: spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: Deployment
  - path: spec/template/spec/volumes/projected/sources/configMap/name
    kind: Deployment
  - path: spec/template/spec/volumes/configMap/name
    kind: ReplicaSet
  - path: spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: ReplicaSet
  - path: spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: ReplicaSet
  - path: spec/template/spec/containers/envFrom/configMapRef/name
    kind: ReplicaSet
  - path: spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: ReplicaSet
  - path: spec/template/spec/volumes/projected/sources/configMap/name
    kind: ReplicaSet
  - path: spec/template/spec/volumes/configMap/name
    kind: DaemonSet
  - path: spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: DaemonSet
  - path: spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: DaemonSet
  - path: spec/template/spec/containers/envFrom/configMapRef/name
    kind: DaemonSet
  - path: spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: DaemonSet
  - path: spec/template/spec/volumes/projected/sources/configMap/name
    kind: DaemonSet
  - path: spec/template/spec/volumes/configMap/name
    kind: StatefulSet
  - path: spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: StatefulSet
  - path: spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: StatefulSet
  - path: spec/template/spec/containers/envFrom/configMapRef/name
    kind: StatefulSet
  - path: spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: StatefulSet
  - path: spec/template/spec/volumes/projected/sources/configMap/name
    kind: StatefulSet
  - path: spec/template/spec/volumes/configMap/name
    kind: Job
  - path: spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: Job
  - path: spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: Job
  - path: spec/template/spec/containers/envFrom/configMapRef/name
    kind: Job
  - path: spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: Job
  - path: spec/template/spec/volumes/projected/sources/configMap/name
    kind: Job
  - path: spec/jobTemplate/spec/template/spec/volumes/configMap/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/volumes/projected/sources/configMap/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/containers/env/valueFrom/configMapKeyRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/initContainers/env/valueFrom/configMapKeyRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/containers/envFrom/configMapRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/initContainers/envFrom/configMapRef/name
    kind: CronJob
  - path: spec/configSource/configMap
    kind: Node
  - path: rules/resourceNames
    kind: Role
  - path: rules/resourceNames
    kind: ClusterRole
  - path: metadata/annotations/nginx.ingress.kubernetes.io\/fastcgi-params-configmap
    kind: Ingress

- kind: Secret
  version: v1
  fieldSpecs:
  - path: spec/volumes/secret/secretName
    version: v1
    kind: Pod
  - path: spec/containers/env/valueFrom/secretKeyRef/name
    version: v1
    kind: Pod
  - path: spec/initContainers/env/valueFrom/secretKeyRef/name
    version: v1
    kind: Pod
  - path: spec/containers/envFrom/secretRef/name
    version: v1
    kind: Pod
  - path: spec/initContainers/envFrom/secretRef/name
    version: v1
    kind: Pod
  - path: spec/imagePullSecrets/name
    version: v1
    kind: Pod
  - path: spec/volumes/projected/sources/secret/name
    version: v1
    kind: Pod
  - path: template/spec/volumes/secret/secretName
    kind: PodTemplate
  - path: template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: PodTemplate
  - path: template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: PodTemplate
  - path: template/spec/containers/envFrom/secretRef/name
    kind: PodTemplate
  - path: template/spec/initContainers/envFrom/secretRef/name
    kind: PodTemplate
  - path: template/spec/imagePullSecrets/name
    kind: PodTemplate
  - path: template/spec/volumes/projected/sources/secret/name
    kind: PodTemplate
  - path: spec/template/spec/volumes/secret/secretName
    kind: Deployment
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: Deployment
  - path: spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: Deployment
  - path: spec/template/spec/containers/envFrom/secretRef/name
    kind: Deployment
  - path: spec/template/spec/initContainers/envFrom/secretRef/name
    kind: Deployment
  - path: spec/template/spec/imagePullSecrets/name
    kind: Deployment
  - path: spec/template/spec/volumes/projected/sources/secret/name
    kind: Deployment
  - path: spec/template/spec/volumes/secret/secretName
    kind: ReplicaSet
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: ReplicaSet
  - path: spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: ReplicaSet
  - path: spec/template/spec/containers/envFrom/secretRef/name
    kind: ReplicaSet
  - path: spec/template/spec/initContainers/envFrom/secretRef/name
    kind: ReplicaSet
  - path: spec/template/spec/imagePullSecrets/name
    kind: ReplicaSet
  - path: spec/template/spec/volumes/projected/sources/secret/name
    kind: ReplicaSet
  - path: spec/template/spec/volumes/secret/secretName
    kind: DaemonSet
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: DaemonSet
  - path: spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: DaemonSet
  - path: spec/template/spec/containers/envFrom/secretRef/name
    kind: DaemonSet
  - path: spec/template/spec/initContainers/envFrom/secretRef/name
    kind: DaemonSet
  - path: spec/template/spec/imagePullSecrets/name
    kind: DaemonSet
  - path: spec/template/spec/volumes/projected/sources/secret/name
    kind: DaemonSet
  - path: spec/template/spec/volumes/secret/secretName
    kind: StatefulSet
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: StatefulSet
  - path: spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: StatefulSet
  - path: spec/template/spec/containers/envFrom/secretRef/name
    kind: StatefulSet
  - path: spec/template/spec/initContainers/envFrom/secretRef/name
    kind: StatefulSet
  - path: spec/template/spec/imagePullSecrets/name
    kind: StatefulSet
  - path: spec/template/spec/volumes/projected/sources/secret/name
    kind: StatefulSet
  - path: spec/template/spec/volumes/secret/secretName
    kind: Job
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: Job
  - path: spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: Job
  - path: spec/template/spec/containers/envFrom/secretRef/name
    kind: Job
  - path: spec/template/spec/initContainers/envFrom/secretRef/name
    kind: Job
  - path: spec/template/spec/imagePullSecrets/name
    kind: Job
  - path: spec/template/spec/volumes/projected/sources/secret/name
    kind: Job
  - path: spec/jobTemplate/spec/template/spec/volumes/secret/secretName
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/volumes/projected/sources/secret/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/initContainers/env/valueFrom/secretKeyRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/containers/envFrom/secretRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/initContainers/envFrom/secretRef/name
    kind: CronJob
  - path: spec/jobTemplate/spec/template/spec/imagePullSecrets/name
    kind: CronJob
  - path: spec/tls/secretName
    kind: Ingress
  - path: metadata/annotations/ingress.kubernetes.io\/auth-secret
    kind: Ingress
  - path: metadata/annotations/nginx.ingress.kubernetes.io\/auth-secret
    kind: Ingress
  - path: metadata/annotations/nginx.ingress.kubernetes.io\/auth-tls-secret
    kind: Ingress
  - path: spec/tls/secretName
    kind: Ingress
  - path: imagePullSecrets/name
    kind: ServiceAccount
  - path: parameters/secretName
    kind: StorageClass
  - path: parameters/adminSecretName
    kind: StorageClass
  - path: parameters/userSecretName
    kind: StorageClass
  - path: parameters/secretRef
    kind: StorageClass
  - path: rules/resourceNames
    kind: Role
  - path: rules/resourceNames
    kind: ClusterRole
  - path: spec/template/spec/containers/env/valueFrom/secretKeyRef/name
    kind: Service
    group: serving.knative.dev
    version: v1
  - path: spec/azureFile/secretName
    kind: PersistentVolume

- kind: Service
  version: v1
  fieldSpecs:
  - path: spec/serviceName
    kind: StatefulSet
    group: apps
  - path: spec/rules/http/paths/backend/serviceName
    kind: Ingress
  - path: spec/backend/serviceName
    kind: Ingress
  - path: spec/rules/http/paths/backend/service/name
    kind: Ingress
  - path: spec/defaultBackend/service/name
    kind: Ingress
  - path: spec/service/name
    kind: APIService
    group: apiregistration.k8s.io
  - path: webhooks/clientConfig/service
    kind: ValidatingWebhookConfiguration
    group: admissionregistration.k8s.io
  - path: webhooks/clientConfig/service
    kind: MutatingWebhookConfiguration
    group: admissionregistration.k8s.io

- kind: Role
  group: rbac.authorization.k8s.io
  fieldSpecs:
  - path: roleRef/name
    kind: RoleBinding
    group: rbac.authorization.k8s.io

- kind: ClusterRole
  group: rbac.authorization.k8s.io
  fieldSpecs:
  - path: roleRef/name
    kind: RoleBinding
    group: rbac.authorization.k8s.io
  - path: roleRef/name
    kind: ClusterRoleBinding
    group: rbac.authorization.k8s.io

- kind: ServiceAccount
  version: v1
  fieldSpecs:
  - path: subjects
    kind: RoleBinding
    group: rbac.authorization.k8s.io
  - path: subjects
    kind: ClusterRoleBinding
    group: rbac.authorization.k8s.io
  - path: spec/serviceAccountName
    kind: Pod
  - path: spec/template/spec/serviceAccountName
    kind: StatefulSet
  - path: spec/template/spec/serviceAccountName
    kind: Deployment
  - path: spec/template/spec/serviceAccountName
    kind: ReplicationController
  - path: spec/jobTemplate/spec/template/spec/serviceAccountName
    kind: CronJob
  - path: spec/template/spec/serviceAccountName
    kind: Job
  - path: spec/template/spec/serviceAccountName
    kind: DaemonSet

- kind: PersistentVolumeClaim
  version: v1
  fieldSpecs:
  - path: spec/volumes/persistentVolumeClaim/claimName
    kind: Pod
  - path: spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: StatefulSet
  - path: spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: Deployment
  - path: spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: ReplicationController
  - path: spec/jobTemplate/spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: CronJob
  - path: spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: Job
  - path: spec/template/spec/volumes/persistentVolumeClaim/claimName
    kind: DaemonSet

- kind: PersistentVolume
  version: v1
  fieldSpecs:
  - path: spec/volumeName
    kind: PersistentVolumeClaim
  - path: rules/resourceNames
    kind: ClusterRole

- kind: StorageClass
  version: v1
  group: storage.k8s.io
  fieldSpecs:
  - path: spec/storageClassName
    kind: PersistentVolume
  - path: spec/storageClassName
    kind: PersistentVolumeClaim
  - path: spec/volumeClaimTemplates/spec/storageClassName
    kind: StatefulSet

- kind: PriorityClass
  version: v1
  group: scheduling.k8s.io
  fieldSpecs:
  - path: spec/priorityClassName
    kind: Pod
  - path: spec/template/spec/priorityClassName
    kind: StatefulSet
  - path: spec/template/spec/priorityClassName
    kind: Deployment
  - path: spec/template/spec/priorityClassName
    kind: ReplicationController
  - path: spec/jobTemplate/spec/template/spec/priorityClassName
    kind: CronJob
  - path: spec/template/spec/priorityClassName
    kind: Job
  - path: spec/template/spec/priorityClassName
    kind: DaemonSet

- kind: IngressClass
  version: v1
  group: networking.k8s.io/v1
  fieldSpecs:
  - path: spec/ingressClassName
    kind: Ingress
`
)

// LINT.ThenChange(/examples/transformerconfigs/README.md)
