package v1

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// +genclient
// +genclient:method=Instantiate,verb=create,subresource=instantiate,input=DeploymentRequest
// +genclient:method=Rollback,verb=create,subresource=rollback,input=DeploymentConfigRollback
// +genclient:method=GetScale,verb=get,subresource=scale,result=k8s.io/api/extensions/v1beta1.Scale
// +genclient:method=UpdateScale,verb=update,subresource=scale,input=k8s.io/api/extensions/v1beta1.Scale,result=k8s.io/api/extensions/v1beta1.Scale
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// Deployment Configs define the template for a pod and manages deploying new images or configuration changes.
// A single deployment configuration is usually analogous to a single micro-service. Can support many different
// deployment patterns, including full restart, customizable rolling updates, and  fully custom behaviors, as
// well as pre- and post- deployment hooks. Each individual deployment is represented as a replication controller.
//
// A deployment is "triggered" when its configuration is changed or a tag in an Image Stream is changed.
// Triggers can be disabled to allow manual control over a deployment. The "strategy" determines how the deployment
// is carried out and may be changed at any time. The `latestVersion` field is updated when a new deployment
// is triggered by any means.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// Deprecated: Use deployments or other means for declarative updates for pods instead.
// +openshift:compatibility-gen:level=1
type DeploymentConfig struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec represents a desired deployment state and how to deploy to it.
	Spec DeploymentConfigSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status represents the current deployment state.
	// +optional
	Status DeploymentConfigStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// DeploymentConfigSpec represents the desired state of the deployment.
type DeploymentConfigSpec struct {
	// Strategy describes how a deployment is executed.
	// +optional
	Strategy DeploymentStrategy `json:"strategy" protobuf:"bytes,1,opt,name=strategy"`

	// MinReadySeconds is the minimum number of seconds for which a newly created pod should
	// be ready without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int32 `json:"minReadySeconds,omitempty" protobuf:"varint,9,opt,name=minReadySeconds"`

	// Triggers determine how updates to a DeploymentConfig result in new deployments. If no triggers
	// are defined, a new deployment can only occur as a result of an explicit client update to the
	// DeploymentConfig with a new LatestVersion. If null, defaults to having a config change trigger.
	// +optional
	Triggers DeploymentTriggerPolicies `json:"triggers" protobuf:"bytes,2,rep,name=triggers"`

	// Replicas is the number of desired replicas.
	// +optional
	Replicas int32 `json:"replicas" protobuf:"varint,3,opt,name=replicas"`

	// RevisionHistoryLimit is the number of old ReplicationControllers to retain to allow for rollbacks.
	// This field is a pointer to allow for differentiation between an explicit zero and not specified.
	// Defaults to 10. (This only applies to DeploymentConfigs created via the new group API resource, not the legacy resource.)
	RevisionHistoryLimit *int32 `json:"revisionHistoryLimit,omitempty" protobuf:"varint,4,opt,name=revisionHistoryLimit"`

	// Test ensures that this deployment config will have zero replicas except while a deployment is running. This allows the
	// deployment config to be used as a continuous deployment test - triggering on images, running the deployment, and then succeeding
	// or failing. Post strategy hooks and After actions can be used to integrate successful deployment with an action.
	// +optional
	Test bool `json:"test" protobuf:"varint,5,opt,name=test"`

	// Paused indicates that the deployment config is paused resulting in no new deployments on template
	// changes or changes in the template caused by other triggers.
	Paused bool `json:"paused,omitempty" protobuf:"varint,6,opt,name=paused"`

	// Selector is a label query over pods that should match the Replicas count.
	Selector map[string]string `json:"selector,omitempty" protobuf:"bytes,7,rep,name=selector"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected.
	Template *corev1.PodTemplateSpec `json:"template,omitempty" protobuf:"bytes,8,opt,name=template"`
}

// DeploymentStrategy describes how to perform a deployment.
type DeploymentStrategy struct {
	// Type is the name of a deployment strategy.
	// +optional
	Type DeploymentStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=DeploymentStrategyType"`

	// CustomParams are the input to the Custom deployment strategy, and may also
	// be specified for the Recreate and Rolling strategies to customize the execution
	// process that runs the deployment.
	CustomParams *CustomDeploymentStrategyParams `json:"customParams,omitempty" protobuf:"bytes,2,opt,name=customParams"`
	// RecreateParams are the input to the Recreate deployment strategy.
	RecreateParams *RecreateDeploymentStrategyParams `json:"recreateParams,omitempty" protobuf:"bytes,3,opt,name=recreateParams"`
	// RollingParams are the input to the Rolling deployment strategy.
	RollingParams *RollingDeploymentStrategyParams `json:"rollingParams,omitempty" protobuf:"bytes,4,opt,name=rollingParams"`

	// Resources contains resource requirements to execute the deployment and any hooks.
	Resources corev1.ResourceRequirements `json:"resources,omitempty" protobuf:"bytes,5,opt,name=resources"`
	// Labels is a set of key, value pairs added to custom deployer and lifecycle pre/post hook pods.
	Labels map[string]string `json:"labels,omitempty" protobuf:"bytes,6,rep,name=labels"`
	// Annotations is a set of key, value pairs added to custom deployer and lifecycle pre/post hook pods.
	Annotations map[string]string `json:"annotations,omitempty" protobuf:"bytes,7,rep,name=annotations"`

	// ActiveDeadlineSeconds is the duration in seconds that the deployer pods for this deployment
	// config may be active on a node before the system actively tries to terminate them.
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty" protobuf:"varint,8,opt,name=activeDeadlineSeconds"`
}

// DeploymentStrategyType refers to a specific DeploymentStrategy implementation.
type DeploymentStrategyType string

const (
	// DeploymentStrategyTypeRecreate is a simple strategy suitable as a default.
	DeploymentStrategyTypeRecreate DeploymentStrategyType = "Recreate"
	// DeploymentStrategyTypeCustom is a user defined strategy.
	DeploymentStrategyTypeCustom DeploymentStrategyType = "Custom"
	// DeploymentStrategyTypeRolling uses the Kubernetes RollingUpdater.
	DeploymentStrategyTypeRolling DeploymentStrategyType = "Rolling"
)

// CustomDeploymentStrategyParams are the input to the Custom deployment strategy.
type CustomDeploymentStrategyParams struct {
	// Image specifies a container image which can carry out a deployment.
	Image string `json:"image,omitempty" protobuf:"bytes,1,opt,name=image"`
	// Environment holds the environment which will be given to the container for Image.
	Environment []corev1.EnvVar `json:"environment,omitempty" protobuf:"bytes,2,rep,name=environment"`
	// Command is optional and overrides CMD in the container Image.
	Command []string `json:"command,omitempty" protobuf:"bytes,3,rep,name=command"`
}

// RecreateDeploymentStrategyParams are the input to the Recreate deployment
// strategy.
type RecreateDeploymentStrategyParams struct {
	// TimeoutSeconds is the time to wait for updates before giving up. If the
	// value is nil, a default will be used.
	TimeoutSeconds *int64 `json:"timeoutSeconds,omitempty" protobuf:"varint,1,opt,name=timeoutSeconds"`
	// Pre is a lifecycle hook which is executed before the strategy manipulates
	// the deployment. All LifecycleHookFailurePolicy values are supported.
	Pre *LifecycleHook `json:"pre,omitempty" protobuf:"bytes,2,opt,name=pre"`
	// Mid is a lifecycle hook which is executed while the deployment is scaled down to zero before the first new
	// pod is created. All LifecycleHookFailurePolicy values are supported.
	Mid *LifecycleHook `json:"mid,omitempty" protobuf:"bytes,3,opt,name=mid"`
	// Post is a lifecycle hook which is executed after the strategy has
	// finished all deployment logic. All LifecycleHookFailurePolicy values are supported.
	Post *LifecycleHook `json:"post,omitempty" protobuf:"bytes,4,opt,name=post"`
}

// RollingDeploymentStrategyParams are the input to the Rolling deployment
// strategy.
type RollingDeploymentStrategyParams struct {
	// UpdatePeriodSeconds is the time to wait between individual pod updates.
	// If the value is nil, a default will be used.
	UpdatePeriodSeconds *int64 `json:"updatePeriodSeconds,omitempty" protobuf:"varint,1,opt,name=updatePeriodSeconds"`
	// IntervalSeconds is the time to wait between polling deployment status
	// after update. If the value is nil, a default will be used.
	IntervalSeconds *int64 `json:"intervalSeconds,omitempty" protobuf:"varint,2,opt,name=intervalSeconds"`
	// TimeoutSeconds is the time to wait for updates before giving up. If the
	// value is nil, a default will be used.
	TimeoutSeconds *int64 `json:"timeoutSeconds,omitempty" protobuf:"varint,3,opt,name=timeoutSeconds"`
	// MaxUnavailable is the maximum number of pods that can be unavailable
	// during the update. Value can be an absolute number (ex: 5) or a
	// percentage of total pods at the start of update (ex: 10%). Absolute
	// number is calculated from percentage by rounding down.
	//
	// This cannot be 0 if MaxSurge is 0. By default, 25% is used.
	//
	// Example: when this is set to 30%, the old RC can be scaled down by 30%
	// immediately when the rolling update starts. Once new pods are ready, old
	// RC can be scaled down further, followed by scaling up the new RC,
	// ensuring that at least 70% of original number of pods are available at
	// all times during the update.
	MaxUnavailable *intstr.IntOrString `json:"maxUnavailable,omitempty" protobuf:"bytes,4,opt,name=maxUnavailable"`
	// MaxSurge is the maximum number of pods that can be scheduled above the
	// original number of pods. Value can be an absolute number (ex: 5) or a
	// percentage of total pods at the start of the update (ex: 10%). Absolute
	// number is calculated from percentage by rounding up.
	//
	// This cannot be 0 if MaxUnavailable is 0. By default, 25% is used.
	//
	// Example: when this is set to 30%, the new RC can be scaled up by 30%
	// immediately when the rolling update starts. Once old pods have been
	// killed, new RC can be scaled up further, ensuring that total number of
	// pods running at any time during the update is atmost 130% of original
	// pods.
	MaxSurge *intstr.IntOrString `json:"maxSurge,omitempty" protobuf:"bytes,5,opt,name=maxSurge"`
	// Pre is a lifecycle hook which is executed before the deployment process
	// begins. All LifecycleHookFailurePolicy values are supported.
	Pre *LifecycleHook `json:"pre,omitempty" protobuf:"bytes,7,opt,name=pre"`
	// Post is a lifecycle hook which is executed after the strategy has
	// finished all deployment logic. All LifecycleHookFailurePolicy values
	// are supported.
	Post *LifecycleHook `json:"post,omitempty" protobuf:"bytes,8,opt,name=post"`
}

// LifecycleHook defines a specific deployment lifecycle action. Only one type of action may be specified at any time.
type LifecycleHook struct {
	// FailurePolicy specifies what action to take if the hook fails.
	FailurePolicy LifecycleHookFailurePolicy `json:"failurePolicy" protobuf:"bytes,1,opt,name=failurePolicy,casttype=LifecycleHookFailurePolicy"`

	// ExecNewPod specifies the options for a lifecycle hook backed by a pod.
	ExecNewPod *ExecNewPodHook `json:"execNewPod,omitempty" protobuf:"bytes,2,opt,name=execNewPod"`

	// TagImages instructs the deployer to tag the current image referenced under a container onto an image stream tag.
	TagImages []TagImageHook `json:"tagImages,omitempty" protobuf:"bytes,3,rep,name=tagImages"`
}

// LifecycleHookFailurePolicy describes possibles actions to take if a hook fails.
type LifecycleHookFailurePolicy string

const (
	// LifecycleHookFailurePolicyRetry means retry the hook until it succeeds.
	LifecycleHookFailurePolicyRetry LifecycleHookFailurePolicy = "Retry"
	// LifecycleHookFailurePolicyAbort means abort the deployment.
	LifecycleHookFailurePolicyAbort LifecycleHookFailurePolicy = "Abort"
	// LifecycleHookFailurePolicyIgnore means ignore failure and continue the deployment.
	LifecycleHookFailurePolicyIgnore LifecycleHookFailurePolicy = "Ignore"
)

// ExecNewPodHook is a hook implementation which runs a command in a new pod
// based on the specified container which is assumed to be part of the
// deployment template.
type ExecNewPodHook struct {
	// Command is the action command and its arguments.
	Command []string `json:"command" protobuf:"bytes,1,rep,name=command"`
	// Env is a set of environment variables to supply to the hook pod's container.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,2,rep,name=env"`
	// ContainerName is the name of a container in the deployment pod template
	// whose container image will be used for the hook pod's container.
	ContainerName string `json:"containerName" protobuf:"bytes,3,opt,name=containerName"`
	// Volumes is a list of named volumes from the pod template which should be
	// copied to the hook pod. Volumes names not found in pod spec are ignored.
	// An empty list means no volumes will be copied.
	Volumes []string `json:"volumes,omitempty" protobuf:"bytes,4,rep,name=volumes"`
}

// TagImageHook is a request to tag the image in a particular container onto an ImageStreamTag.
type TagImageHook struct {
	// ContainerName is the name of a container in the deployment config whose image value will be used as the source of the tag. If there is only a single
	// container this value will be defaulted to the name of that container.
	ContainerName string `json:"containerName" protobuf:"bytes,1,opt,name=containerName"`
	// To is the target ImageStreamTag to set the container's image onto.
	To corev1.ObjectReference `json:"to" protobuf:"bytes,2,opt,name=to"`
}

// DeploymentTriggerPolicies is a list of policies where nil values and different from empty arrays.
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type DeploymentTriggerPolicies []DeploymentTriggerPolicy

func (t DeploymentTriggerPolicies) String() string {
	return fmt.Sprintf("%v", []DeploymentTriggerPolicy(t))
}

// DeploymentTriggerPolicy describes a policy for a single trigger that results in a new deployment.
type DeploymentTriggerPolicy struct {
	// Type of the trigger
	Type DeploymentTriggerType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=DeploymentTriggerType"`
	// ImageChangeParams represents the parameters for the ImageChange trigger.
	ImageChangeParams *DeploymentTriggerImageChangeParams `json:"imageChangeParams,omitempty" protobuf:"bytes,2,opt,name=imageChangeParams"`
}

// DeploymentTriggerType refers to a specific DeploymentTriggerPolicy implementation.
type DeploymentTriggerType string

const (
	// DeploymentTriggerOnImageChange will create new deployments in response to updated tags from
	// a container image repository.
	DeploymentTriggerOnImageChange DeploymentTriggerType = "ImageChange"
	// DeploymentTriggerOnConfigChange will create new deployments in response to changes to
	// the ControllerTemplate of a DeploymentConfig.
	DeploymentTriggerOnConfigChange DeploymentTriggerType = "ConfigChange"
)

// DeploymentTriggerImageChangeParams represents the parameters to the ImageChange trigger.
type DeploymentTriggerImageChangeParams struct {
	// Automatic means that the detection of a new tag value should result in an image update
	// inside the pod template.
	Automatic bool `json:"automatic,omitempty" protobuf:"varint,1,opt,name=automatic"`
	// ContainerNames is used to restrict tag updates to the specified set of container names in a pod.
	// If multiple triggers point to the same containers, the resulting behavior is undefined. Future
	// API versions will make this a validation error. If ContainerNames does not point to a valid container,
	// the trigger will be ignored. Future API versions will make this a validation error.
	ContainerNames []string `json:"containerNames,omitempty" protobuf:"bytes,2,rep,name=containerNames"`
	// From is a reference to an image stream tag to watch for changes. From.Name is the only
	// required subfield - if From.Namespace is blank, the namespace of the current deployment
	// trigger will be used.
	From corev1.ObjectReference `json:"from" protobuf:"bytes,3,opt,name=from"`
	// LastTriggeredImage is the last image to be triggered.
	LastTriggeredImage string `json:"lastTriggeredImage,omitempty" protobuf:"bytes,4,opt,name=lastTriggeredImage"`
}

// DeploymentConfigStatus represents the current deployment state.
type DeploymentConfigStatus struct {
	// LatestVersion is used to determine whether the current deployment associated with a deployment
	// config is out of sync.
	LatestVersion int64 `json:"latestVersion" protobuf:"varint,1,opt,name=latestVersion"`
	// ObservedGeneration is the most recent generation observed by the deployment config controller.
	ObservedGeneration int64 `json:"observedGeneration" protobuf:"varint,2,opt,name=observedGeneration"`
	// Replicas is the total number of pods targeted by this deployment config.
	Replicas int32 `json:"replicas" protobuf:"varint,3,opt,name=replicas"`
	// UpdatedReplicas is the total number of non-terminated pods targeted by this deployment config
	// that have the desired template spec.
	UpdatedReplicas int32 `json:"updatedReplicas" protobuf:"varint,4,opt,name=updatedReplicas"`
	// AvailableReplicas is the total number of available pods targeted by this deployment config.
	AvailableReplicas int32 `json:"availableReplicas" protobuf:"varint,5,opt,name=availableReplicas"`
	// UnavailableReplicas is the total number of unavailable pods targeted by this deployment config.
	UnavailableReplicas int32 `json:"unavailableReplicas" protobuf:"varint,6,opt,name=unavailableReplicas"`
	// Details are the reasons for the update to this deployment config.
	// This could be based on a change made by the user or caused by an automatic trigger
	Details *DeploymentDetails `json:"details,omitempty" protobuf:"bytes,7,opt,name=details"`
	// Conditions represents the latest available observations of a deployment config's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []DeploymentCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,8,rep,name=conditions"`
	// Total number of ready pods targeted by this deployment.
	ReadyReplicas int32 `json:"readyReplicas,omitempty" protobuf:"varint,9,opt,name=readyReplicas"`
}

// DeploymentDetails captures information about the causes of a deployment.
type DeploymentDetails struct {
	// Message is the user specified change message, if this deployment was triggered manually by the user
	Message string `json:"message,omitempty" protobuf:"bytes,1,opt,name=message"`
	// Causes are extended data associated with all the causes for creating a new deployment
	Causes []DeploymentCause `json:"causes" protobuf:"bytes,2,rep,name=causes"`
}

// DeploymentCause captures information about a particular cause of a deployment.
type DeploymentCause struct {
	// Type of the trigger that resulted in the creation of a new deployment
	Type DeploymentTriggerType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=DeploymentTriggerType"`
	// ImageTrigger contains the image trigger details, if this trigger was fired based on an image change
	ImageTrigger *DeploymentCauseImageTrigger `json:"imageTrigger,omitempty" protobuf:"bytes,2,opt,name=imageTrigger"`
}

// DeploymentCauseImageTrigger represents details about the cause of a deployment originating
// from an image change trigger
type DeploymentCauseImageTrigger struct {
	// From is a reference to the changed object which triggered a deployment. The field may have
	// the kinds DockerImage, ImageStreamTag, or ImageStreamImage.
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`
}

type DeploymentConditionType string

// These are valid conditions of a DeploymentConfig.
const (
	// DeploymentAvailable means the DeploymentConfig is available, ie. at least the minimum available
	// replicas required (dc.spec.replicas in case the DeploymentConfig is of Recreate type,
	// dc.spec.replicas - dc.spec.strategy.rollingParams.maxUnavailable in case it's Rolling) are up and
	// running for at least dc.spec.minReadySeconds.
	DeploymentAvailable DeploymentConditionType = "Available"
	// DeploymentProgressing is:
	// * True: the DeploymentConfig has been successfully deployed or is amidst getting deployed.
	//   The two different states can be determined by looking at the Reason of the Condition.
	//   For example, a complete DC will have {Status: True, Reason: NewReplicationControllerAvailable}
	//   and a DC in the middle of a rollout {Status: True, Reason: ReplicationControllerUpdated}.
	//   TODO: Represent a successfully deployed DC by using something else for Status like Unknown?
	// * False: the DeploymentConfig has failed to deploy its latest version.
	//
	// This condition is purely informational and depends on the dc.spec.strategy.*params.timeoutSeconds
	// field, which is responsible for the time in seconds to wait for a rollout before deciding that
	// no progress can be made, thus the rollout is aborted.
	//
	// Progress for a DeploymentConfig is considered when new pods scale up or old pods scale down.
	DeploymentProgressing DeploymentConditionType = "Progressing"
	// DeploymentReplicaFailure is added in a deployment config when one of its pods
	// fails to be created or deleted.
	DeploymentReplicaFailure DeploymentConditionType = "ReplicaFailure"
)

// DeploymentCondition describes the state of a deployment config at a certain point.
type DeploymentCondition struct {
	// Type of deployment condition.
	Type DeploymentConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=DeploymentConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/kubernetes/pkg/api/v1.ConditionStatus"`
	// The last time this condition was updated.
	LastUpdateTime metav1.Time `json:"lastUpdateTime,omitempty" protobuf:"bytes,6,opt,name=lastUpdateTime"`
	// The last time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// DeploymentConfigList is a collection of deployment configs.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DeploymentConfigList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of deployment configs
	Items []DeploymentConfig `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// DeploymentConfigRollback provides the input to rollback generation.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DeploymentConfigRollback struct {
	metav1.TypeMeta `json:",inline"`
	// Name of the deployment config that will be rolled back.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// UpdatedAnnotations is a set of new annotations that will be added in the deployment config.
	UpdatedAnnotations map[string]string `json:"updatedAnnotations,omitempty" protobuf:"bytes,2,rep,name=updatedAnnotations"`
	// Spec defines the options to rollback generation.
	Spec DeploymentConfigRollbackSpec `json:"spec" protobuf:"bytes,3,opt,name=spec"`
}

// DeploymentConfigRollbackSpec represents the options for rollback generation.
type DeploymentConfigRollbackSpec struct {
	// From points to a ReplicationController which is a deployment.
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`
	// Revision to rollback to. If set to 0, rollback to the last revision.
	Revision int64 `json:"revision,omitempty" protobuf:"varint,2,opt,name=revision"`
	// IncludeTriggers specifies whether to include config Triggers.
	IncludeTriggers bool `json:"includeTriggers" protobuf:"varint,3,opt,name=includeTriggers"`
	// IncludeTemplate specifies whether to include the PodTemplateSpec.
	IncludeTemplate bool `json:"includeTemplate" protobuf:"varint,4,opt,name=includeTemplate"`
	// IncludeReplicationMeta specifies whether to include the replica count and selector.
	IncludeReplicationMeta bool `json:"includeReplicationMeta" protobuf:"varint,5,opt,name=includeReplicationMeta"`
	// IncludeStrategy specifies whether to include the deployment Strategy.
	IncludeStrategy bool `json:"includeStrategy" protobuf:"varint,6,opt,name=includeStrategy"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// DeploymentRequest is a request to a deployment config for a new deployment.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DeploymentRequest struct {
	metav1.TypeMeta `json:",inline"`
	// Name of the deployment config for requesting a new deployment.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// Latest will update the deployment config with the latest state from all triggers.
	Latest bool `json:"latest" protobuf:"varint,2,opt,name=latest"`
	// Force will try to force a new deployment to run. If the deployment config is paused,
	// then setting this to true will return an Invalid error.
	Force bool `json:"force" protobuf:"varint,3,opt,name=force"`
	// ExcludeTriggers instructs the instantiator to avoid processing the specified triggers.
	// This field overrides the triggers from latest and allows clients to control specific
	// logic. This field is ignored if not specified.
	ExcludeTriggers []DeploymentTriggerType `json:"excludeTriggers,omitempty" protobuf:"bytes,4,rep,name=excludeTriggers,casttype=DeploymentTriggerType"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// DeploymentLog represents the logs for a deployment
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DeploymentLog struct {
	metav1.TypeMeta `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=3.0
// +k8s:prerelease-lifecycle-gen:deprecated=4.14
// +k8s:prerelease-lifecycle-gen:removed=4.10000

// DeploymentLogOptions is the REST options for a deployment log
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DeploymentLogOptions struct {
	metav1.TypeMeta `json:",inline"`

	// The container for which to stream logs. Defaults to only container if there is one container in the pod.
	Container string `json:"container,omitempty" protobuf:"bytes,1,opt,name=container"`
	// Follow if true indicates that the build log should be streamed until
	// the build terminates.
	Follow bool `json:"follow,omitempty" protobuf:"varint,2,opt,name=follow"`
	// Return previous deployment logs. Defaults to false.
	Previous bool `json:"previous,omitempty" protobuf:"varint,3,opt,name=previous"`
	// A relative time in seconds before the current time from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceSeconds *int64 `json:"sinceSeconds,omitempty" protobuf:"varint,4,opt,name=sinceSeconds"`
	// An RFC3339 timestamp from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceTime *metav1.Time `json:"sinceTime,omitempty" protobuf:"bytes,5,opt,name=sinceTime"`
	// If true, add an RFC3339 or RFC3339Nano timestamp at the beginning of every line
	// of log output. Defaults to false.
	Timestamps bool `json:"timestamps,omitempty" protobuf:"varint,6,opt,name=timestamps"`
	// If set, the number of lines from the end of the logs to show. If not specified,
	// logs are shown from the creation of the container or sinceSeconds or sinceTime
	TailLines *int64 `json:"tailLines,omitempty" protobuf:"varint,7,opt,name=tailLines"`
	// If set, the number of bytes to read from the server before terminating the
	// log output. This may not display a complete final line of logging, and may return
	// slightly more or slightly less than the specified limit.
	LimitBytes *int64 `json:"limitBytes,omitempty" protobuf:"varint,8,opt,name=limitBytes"`

	// NoWait if true causes the call to return immediately even if the deployment
	// is not available yet. Otherwise the server will wait until the deployment has started.
	// TODO: Fix the tag to 'noWait' in v2
	NoWait bool `json:"nowait,omitempty" protobuf:"varint,9,opt,name=nowait"`

	// Version of the deployment for which to view logs.
	Version *int64 `json:"version,omitempty" protobuf:"varint,10,opt,name=version"`
}
