package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=machineconfigurations,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1453
// +openshift:file-pattern=cvoRunLevel=0000_80,operatorName=machine-config,operatorOrdering=01

// MachineConfiguration provides information to configure an operator to manage Machine Configuration.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type MachineConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// spec is the specification of the desired behavior of the Machine Config Operator
	// +required
	Spec MachineConfigurationSpec `json:"spec"`

	// status is the most recently observed status of the Machine Config Operator
	// +optional
	Status MachineConfigurationStatus `json:"status"`
}

type MachineConfigurationSpec struct {
	StaticPodOperatorSpec `json:",inline"`

	// TODO(jkyros): This is where we put our knobs and dials

	// managedBootImages allows configuration for the management of boot images for machine
	// resources within the cluster. This configuration allows users to select resources that should
	// be updated to the latest boot images during cluster upgrades, ensuring that new machines
	// always boot with the current cluster version's boot image. When omitted, no boot images
	// will be updated.
	// +openshift:enable:FeatureGate=ManagedBootImages
	// +optional
	ManagedBootImages ManagedBootImages `json:"managedBootImages"`

	// nodeDisruptionPolicy allows an admin to set granular node disruption actions for
	// MachineConfig-based updates, such as drains, service reloads, etc. Specifying this will allow
	// for less downtime when doing small configuration updates to the cluster. This configuration
	// has no effect on cluster upgrades which will still incur node disruption where required.
	// +openshift:enable:FeatureGate=NodeDisruptionPolicy
	// +optional
	NodeDisruptionPolicy NodeDisruptionPolicyConfig `json:"nodeDisruptionPolicy"`
}

type MachineConfigurationStatus struct {
	// observedGeneration is the last generation change you've dealt with
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// conditions is a list of conditions and their status
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// Previously there was a StaticPodOperatorStatus here for legacy reasons. Many of the fields within
	// it are no longer relevant for the MachineConfiguration CRD's functions. The following remainder
	// fields were tombstoned after lifting out StaticPodOperatorStatus. To avoid conflicts with
	// serialisation, the following field names may never be used again.

	// Tombstone: legacy field from StaticPodOperatorStatus
	// Version string `json:"version,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// ReadyReplicas int32 `json:"readyReplicas"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// Generations []GenerationStatus `json:"generations,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// LatestAvailableRevision int32 `json:"latestAvailableRevision,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// LatestAvailableRevisionReason string `json:"latestAvailableRevisionReason,omitempty"`

	// Tombstone: legacy field from StaticPodOperatorStatus
	// NodeStatuses []NodeStatus `json:"nodeStatuses,omitempty"`

	// nodeDisruptionPolicyStatus status reflects what the latest cluster-validated policies are,
	// and will be used by the Machine Config Daemon during future node updates.
	// +openshift:enable:FeatureGate=NodeDisruptionPolicy
	// +optional
	NodeDisruptionPolicyStatus NodeDisruptionPolicyStatus `json:"nodeDisruptionPolicyStatus"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MachineConfigurationList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type MachineConfigurationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// items contains the items
	Items []MachineConfiguration `json:"items"`
}

type ManagedBootImages struct {
	// machineManagers can be used to register machine management resources for boot image updates. The Machine Config Operator
	// will watch for changes to this list. Only one entry is permitted per type of machine management resource.
	// +optional
	// +listType=map
	// +listMapKey=resource
	// +listMapKey=apiGroup
	MachineManagers []MachineManager `json:"machineManagers"`
}

// MachineManager describes a target machine resource that is registered for boot image updates. It stores identifying information
// such as the resource type and the API Group of the resource. It also provides granular control via the selection field.
type MachineManager struct {
	// resource is the machine management resource's type.
	// The only current valid value is machinesets.
	// machinesets means that the machine manager will only register resources of the kind MachineSet.
	// +required
	Resource MachineManagerMachineSetsResourceType `json:"resource"`

	// apiGroup is name of the APIGroup that the machine management resource belongs to.
	// The only current valid value is machine.openshift.io.
	// machine.openshift.io means that the machine manager will only register resources that belong to OpenShift machine API group.
	// +required
	APIGroup MachineManagerMachineSetsAPIGroupType `json:"apiGroup"`

	// selection allows granular control of the machine management resources that will be registered for boot image updates.
	// +required
	Selection MachineManagerSelector `json:"selection"`
}

// +kubebuilder:validation:XValidation:rule="has(self.mode) && self.mode == 'Partial' ?  has(self.partial) : !has(self.partial)",message="Partial is required when type is partial, and forbidden otherwise"
// +union
type MachineManagerSelector struct {
	// mode determines how machine managers will be selected for updates.
	// Valid values are All and Partial.
	// All means that every resource matched by the machine manager will be updated.
	// Partial requires specified selector(s) and allows customisation of which resources matched by the machine manager will be updated.
	// +unionDiscriminator
	// +required
	Mode MachineManagerSelectorMode `json:"mode"`

	// partial provides label selector(s) that can be used to match machine management resources.
	// Only permitted when mode is set to "Partial".
	// +optional
	Partial *PartialSelector `json:"partial,omitempty"`
}

// PartialSelector provides label selector(s) that can be used to match machine management resources.
type PartialSelector struct {
	// machineResourceSelector is a label selector that can be used to select machine resources like MachineSets.
	// +required
	MachineResourceSelector *metav1.LabelSelector `json:"machineResourceSelector,omitempty"`
}

// MachineManagerSelectorMode is a string enum used in the MachineManagerSelector union discriminator.
// +kubebuilder:validation:Enum:="All";"Partial"
type MachineManagerSelectorMode string

const (
	// All represents a configuration mode that registers all resources specified by the parent MachineManager for boot image updates.
	All MachineManagerSelectorMode = "All"

	// Partial represents a configuration mode that will register resources specified by the parent MachineManager only
	// if they match with the label selector.
	Partial MachineManagerSelectorMode = "Partial"
)

// MachineManagerManagedResourceType is a string enum used in the MachineManager type to describe the resource
// type to be registered.
// +kubebuilder:validation:Enum:="machinesets"
type MachineManagerMachineSetsResourceType string

const (
	// MachineSets represent the MachineSet resource type, which manage a group of machines and belong to the Openshift machine API group.
	MachineSets MachineManagerMachineSetsResourceType = "machinesets"
)

// MachineManagerManagedAPIGroupType is a string enum used in in the MachineManager type to describe the APIGroup
// of the resource type being registered.
// +kubebuilder:validation:Enum:="machine.openshift.io"
type MachineManagerMachineSetsAPIGroupType string

const (
	// MachineAPI represent the traditional MAPI Group that a machineset may belong to.
	// This feature only supports MAPI machinesets at this time.
	MachineAPI MachineManagerMachineSetsAPIGroupType = "machine.openshift.io"
)

type NodeDisruptionPolicyStatus struct {
	// clusterPolicies is a merge of cluster default and user provided node disruption policies.
	// +optional
	ClusterPolicies NodeDisruptionPolicyClusterStatus `json:"clusterPolicies"`
}

// NodeDisruptionPolicyConfig is the overall spec definition for files/units/sshkeys
type NodeDisruptionPolicyConfig struct {
	// files is a list of MachineConfig file definitions and actions to take to changes on those paths
	// This list supports a maximum of 50 entries.
	// +optional
	// +listType=map
	// +listMapKey=path
	// +kubebuilder:validation:MaxItems=50
	Files []NodeDisruptionPolicySpecFile `json:"files"`
	// units is a list MachineConfig unit definitions and actions to take on changes to those services
	// This list supports a maximum of 50 entries.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=50
	Units []NodeDisruptionPolicySpecUnit `json:"units"`
	// sshkey maps to the ignition.sshkeys field in the MachineConfig object, definition an action for this
	// will apply to all sshkey changes in the cluster
	// +optional
	SSHKey NodeDisruptionPolicySpecSSHKey `json:"sshkey"`
}

// NodeDisruptionPolicyClusterStatus is the type for the status object, rendered by the controller as a
// merge of cluster defaults and user provided policies
type NodeDisruptionPolicyClusterStatus struct {
	// files is a list of MachineConfig file definitions and actions to take to changes on those paths
	// +optional
	// +listType=map
	// +listMapKey=path
	// +kubebuilder:validation:MaxItems=100
	Files []NodeDisruptionPolicyStatusFile `json:"files,omitempty"`
	// units is a list MachineConfig unit definitions and actions to take on changes to those services
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=100
	Units []NodeDisruptionPolicyStatusUnit `json:"units,omitempty"`
	// sshkey is the overall sshkey MachineConfig definition
	// +optional
	SSHKey NodeDisruptionPolicyStatusSSHKey `json:"sshkey,omitempty"`
}

// NodeDisruptionPolicySpecFile is a file entry and corresponding actions to take and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecFile struct {
	// path is the location of a file being managed through a MachineConfig.
	// The Actions in the policy will apply to changes to the file at this path.
	// +required
	Path string `json:"path"`
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusFile is a file entry and corresponding actions to take and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusFile struct {
	// path is the location of a file being managed through a MachineConfig.
	// The Actions in the policy will apply to changes to the file at this path.
	// +required
	Path string `json:"path"`
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// NodeDisruptionPolicySpecUnit is a systemd unit name and corresponding actions to take and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecUnit struct {
	// name represents the service name of a systemd service managed through a MachineConfig
	// Actions specified will be applied for changes to the named service.
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	Name NodeDisruptionPolicyServiceName `json:"name"`

	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusUnit is a systemd unit name and corresponding actions to take and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusUnit struct {
	// name represents the service name of a systemd service managed through a MachineConfig
	// Actions specified will be applied for changes to the named service.
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	Name NodeDisruptionPolicyServiceName `json:"name"`

	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// NodeDisruptionPolicySpecSSHKey is actions to take for any SSHKey change and is used in the NodeDisruptionPolicyConfig object
type NodeDisruptionPolicySpecSSHKey struct {
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicySpecAction `json:"actions"`
}

// NodeDisruptionPolicyStatusSSHKey is actions to take for any SSHKey change and is used in the NodeDisruptionPolicyClusterStatus object
type NodeDisruptionPolicyStatusSSHKey struct {
	// actions represents the series of commands to be executed on changes to the file at
	// the corresponding file path. Actions will be applied in the order that
	// they are set in this list. If there are other incoming changes to other MachineConfig
	// entries in the same update that require a reboot, the reboot will supercede these actions.
	// Valid actions are Reboot, Drain, Reload, DaemonReload and None.
	// The Reboot action and the None action cannot be used in conjunction with any of the other actions.
	// This list supports a maximum of 10 entries.
	// +required
	// +listType=atomic
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='Reboot') ? size(self) == 1 : true", message="Reboot action can only be specified standalone, as it will override any other actions"
	// +kubebuilder:validation:XValidation:rule="self.exists(x, x.type=='None') ? size(self) == 1 : true", message="None action can only be specified standalone, as it will override any other actions"
	Actions []NodeDisruptionPolicyStatusAction `json:"actions"`
}

// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Reload' ? has(self.reload) : !has(self.reload)",message="reload is required when type is Reload, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Restart' ? has(self.restart) : !has(self.restart)",message="restart is required when type is Restart, and forbidden otherwise"
// +union
type NodeDisruptionPolicySpecAction struct {
	// type represents the commands that will be carried out if this NodeDisruptionPolicySpecActionType is executed
	// Valid values are Reboot, Drain, Reload, Restart, DaemonReload and None.
	// reload/restart requires a corresponding service target specified in the reload/restart field.
	// Other values require no further configuration
	// +unionDiscriminator
	// +required
	Type NodeDisruptionPolicySpecActionType `json:"type"`
	// reload specifies the service to reload, only valid if type is reload
	// +optional
	Reload *ReloadService `json:"reload,omitempty"`
	// restart specifies the service to restart, only valid if type is restart
	// +optional
	Restart *RestartService `json:"restart,omitempty"`
}

// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Reload' ? has(self.reload) : !has(self.reload)",message="reload is required when type is Reload, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Restart' ? has(self.restart) : !has(self.restart)",message="restart is required when type is Restart, and forbidden otherwise"
// +union
type NodeDisruptionPolicyStatusAction struct {
	// type represents the commands that will be carried out if this NodeDisruptionPolicyStatusActionType is executed
	// Valid values are Reboot, Drain, Reload, Restart, DaemonReload, None and Special.
	// reload/restart requires a corresponding service target specified in the reload/restart field.
	// Other values require no further configuration
	// +unionDiscriminator
	// +required
	Type NodeDisruptionPolicyStatusActionType `json:"type"`
	// reload specifies the service to reload, only valid if type is reload
	// +optional
	Reload *ReloadService `json:"reload,omitempty"`
	// restart specifies the service to restart, only valid if type is restart
	// +optional
	Restart *RestartService `json:"restart,omitempty"`
}

// ReloadService allows the user to specify the services to be reloaded
type ReloadService struct {
	// serviceName is the full name (e.g. crio.service) of the service to be reloaded
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	ServiceName NodeDisruptionPolicyServiceName `json:"serviceName"`
}

// RestartService allows the user to specify the services to be restarted
type RestartService struct {
	// serviceName is the full name (e.g. crio.service) of the service to be restarted
	// Service names should be of the format ${NAME}${SERVICETYPE} and can up to 255 characters long.
	// ${NAME} must be atleast 1 character long and can only consist of alphabets, digits, ":", "-", "_", ".", and "\".
	// ${SERVICETYPE} must be one of ".service", ".socket", ".device", ".mount", ".automount", ".swap", ".target", ".path", ".timer", ".snapshot", ".slice" or ".scope".
	// +required
	ServiceName NodeDisruptionPolicyServiceName `json:"serviceName"`
}

// NodeDisruptionPolicySpecActionType is a string enum used in a NodeDisruptionPolicySpecAction object. They describe an action to be performed.
// +kubebuilder:validation:Enum:="Reboot";"Drain";"Reload";"Restart";"DaemonReload";"None"
type NodeDisruptionPolicySpecActionType string

// +kubebuilder:validation:XValidation:rule=`self.matches('\\.(service|socket|device|mount|automount|swap|target|path|timer|snapshot|slice|scope)$')`, message="Invalid ${SERVICETYPE} in service name. Expected format is ${NAME}${SERVICETYPE}, where ${SERVICETYPE} must be one of \".service\", \".socket\", \".device\", \".mount\", \".automount\", \".swap\", \".target\", \".path\", \".timer\",\".snapshot\", \".slice\" or \".scope\"."
// +kubebuilder:validation:XValidation:rule=`self.matches('^[a-zA-Z0-9:._\\\\-]+\\..')`, message="Invalid ${NAME} in service name. Expected format is ${NAME}${SERVICETYPE}, where {NAME} must be atleast 1 character long and can only consist of alphabets, digits, \":\", \"-\", \"_\", \".\", and \"\\\""
// +kubebuilder:validation:MaxLength=255
type NodeDisruptionPolicyServiceName string

const (
	// Reboot represents an action that will cause nodes to be rebooted. This is the default action by the MCO
	// if a reboot policy is not found for a change/update being performed by the MCO.
	RebootSpecAction NodeDisruptionPolicySpecActionType = "Reboot"

	// Drain represents an action that will cause nodes to be drained of their workloads.
	DrainSpecAction NodeDisruptionPolicySpecActionType = "Drain"

	// Reload represents an action that will cause nodes to reload the service described by the Target field.
	ReloadSpecAction NodeDisruptionPolicySpecActionType = "Reload"

	// Restart represents an action that will cause nodes to restart the service described by the Target field.
	RestartSpecAction NodeDisruptionPolicySpecActionType = "Restart"

	// DaemonReload represents an action that TBD
	DaemonReloadSpecAction NodeDisruptionPolicySpecActionType = "DaemonReload"

	// None represents an action that no handling is required by the MCO.
	NoneSpecAction NodeDisruptionPolicySpecActionType = "None"
)

// NodeDisruptionPolicyStatusActionType is a string enum used in a NodeDisruptionPolicyStatusAction object. They describe an action to be performed.
// The key difference of this object from NodeDisruptionPolicySpecActionType is that there is a additional SpecialStatusAction value in this enum. This will only be
// used by the MCO's controller to indicate some internal actions. They are not part of the NodeDisruptionPolicyConfig object and cannot be set by the user.
// +kubebuilder:validation:Enum:="Reboot";"Drain";"Reload";"Restart";"DaemonReload";"None";"Special"
type NodeDisruptionPolicyStatusActionType string

const (
	// Reboot represents an action that will cause nodes to be rebooted. This is the default action by the MCO
	// if a reboot policy is not found for a change/update being performed by the MCO.
	RebootStatusAction NodeDisruptionPolicyStatusActionType = "Reboot"

	// Drain represents an action that will cause nodes to be drained of their workloads.
	DrainStatusAction NodeDisruptionPolicyStatusActionType = "Drain"

	// Reload represents an action that will cause nodes to reload the service described by the Target field.
	ReloadStatusAction NodeDisruptionPolicyStatusActionType = "Reload"

	// Restart represents an action that will cause nodes to restart the service described by the Target field.
	RestartStatusAction NodeDisruptionPolicyStatusActionType = "Restart"

	// DaemonReload represents an action that TBD
	DaemonReloadStatusAction NodeDisruptionPolicyStatusActionType = "DaemonReload"

	// None represents an action that no handling is required by the MCO.
	NoneStatusAction NodeDisruptionPolicyStatusActionType = "None"

	// Special represents an action that is internal to the MCO, and is not allowed in user defined NodeDisruption policies.
	SpecialStatusAction NodeDisruptionPolicyStatusActionType = "Special"
)

// These strings will be used for MachineConfiguration Status conditions.
const (
	// MachineConfigurationBootImageUpdateDegraded means that the MCO ran into an error while reconciling boot images. This
	// will cause the clusteroperators.config.openshift.io/machine-config to degrade. This  condition will indicate the cause
	// of the degrade, the progress of the update and the generation of the boot images configmap that it degraded on.
	MachineConfigurationBootImageUpdateDegraded string = "BootImageUpdateDegraded"

	// MachineConfigurationBootImageUpdateProgressing means that the MCO is in the process of reconciling boot images. This
	// will cause the clusteroperators.config.openshift.io/machine-config to be in a Progressing state. This condition will
	// indicate the progress of the update and the generation of the boot images configmap that triggered this update.
	MachineConfigurationBootImageUpdateProgressing string = "BootImageUpdateProgressing"
)
