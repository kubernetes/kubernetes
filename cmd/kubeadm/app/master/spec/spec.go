/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This has been temporarily taken from https://github.com/coreos/etcd-operator
// and will be properly vendored when kubeadm moves to its own repo.

package spec

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	defaultVersion = "3.1.4"

	TPRKind        = "cluster"
	TPRKindPlural  = "clusters"
	TPRGroup       = "etcd.coreos.com"
	TPRVersion     = "v1beta1"
	TPRDescription = "Managed etcd clusters"
)

var (
	ErrBackupUnsetRestoreSet = errors.New("spec: backup policy must be set if restore policy is set")
)

func TPRName() string {
	return fmt.Sprintf("%s.%s", TPRKind, TPRGroup)
}

type Cluster struct {
	metav1.TypeMeta `json:",inline"`
	Metadata        metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec            ClusterSpec       `json:"spec"`
	Status          ClusterStatus     `json:"status"`
}

func (c *Cluster) AsOwner() metav1.OwnerReference {
	trueVar := true
	// TODO: In 1.6 this is gonna be "k8s.io/kubernetes/pkg/apis/meta/v1"
	// Both api.OwnerReference and metatypes.OwnerReference are combined into that.
	return metav1.OwnerReference{
		APIVersion: c.APIVersion,
		Kind:       c.Kind,
		Name:       c.Metadata.Name,
		UID:        c.Metadata.UID,
		Controller: &trueVar,
	}
}

type ClusterSpec struct {
	// Size is the expected size of the etcd cluster.
	// The etcd-operator will eventually make the size of the running
	// cluster equal to the expected size.
	// The vaild range of the size is from 1 to 7.
	Size int `json:"size"`

	// Version is the expected version of the etcd cluster.
	// The etcd-operator will eventually make the etcd cluster version
	// equal to the expected version.
	//
	// The version must follow the [semver]( http://semver.org) format, for example "3.1.4".
	// Only etcd released versions are supported: https://github.com/coreos/etcd/releases
	//
	// If version is not set, default is "3.1.4".
	Version string `json:"version"`

	// Paused is to pause the control of the operator for the etcd cluster.
	Paused bool `json:"paused,omitempty"`

	// Pod defines the policy to create pod for the etcd pod.
	//
	// Updating Pod does not take effect on any existing etcd pods.
	Pod *PodPolicy `json:"pod,omitempty"`

	// Backup defines the policy to backup data of etcd cluster if not nil.
	// If backup policy is set but restore policy not, and if a previous backup exists,
	// this cluster would face conflict and fail to start.
	Backup *BackupPolicy `json:"backup,omitempty"`

	// Restore defines the policy to restore cluster form existing backup if not nil.
	// It's not allowed if restore policy is set and backup policy not.
	//
	// Restore is a cluster initialization configuration. It cannot be updated.
	Restore *RestorePolicy `json:"restore,omitempty"`

	// SelfHosted determines if the etcd cluster is used for a self-hosted
	// Kubernetes cluster.
	//
	// SelfHosted is a cluster initialization configuration. It cannot be updated.
	SelfHosted *SelfHostedPolicy `json:"selfHosted,omitempty"`

	// NOTE: This field is half finished. It will be ignored.
	// etcd cluster TLS configuration
	TLS *TLSPolicy `json:"TLS,omitempty"`
}

// RestorePolicy defines the policy to restore cluster form existing backup if not nil.
type RestorePolicy struct {
	// BackupClusterName is the cluster name of the backup to recover from.
	BackupClusterName string `json:"backupClusterName"`

	// StorageType specifies the type of storage device to store backup files.
	// If not set, the default is "PersistentVolume".
	StorageType BackupStorageType `json:"storageType"`
}

// PodPolicy defines the policy to create pod for the etcd container.
type PodPolicy struct {
	// Labels specifies the labels to attach to pods the operator creates for the
	// etcd cluster.
	// "app" and "etcd_*" labels are reserved for the internal use of the etcd operator.
	// Do not overwrite them.
	Labels map[string]string `json:"labels,omitempty"`

	// NodeSelector specifies a map of key-value pairs. For the pod to be eligible
	// to run on a node, the node must have each of the indicated key-value pairs as
	// labels.
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// AntiAffinity determines if the etcd-operator tries to avoid putting
	// the etcd members in the same cluster onto the same node.
	AntiAffinity bool `json:"antiAffinity"`

	// Resources is the resource requirements for the etcd container.
	// This field cannot be updated once the cluster is created.
	Resources v1.ResourceRequirements `json:"resources"`

	// Tolerations specifies the pod's tolerations.
	Tolerations []v1.Toleration `json:"tolerations"`

	// List of environment variables to set in the etcd container.
	// This is used to configure etcd process. etcd cluster cannot be created, when
	// bad environement variables are provided. Do not overwrite any flags used to
	// bootstrap the cluster (for example `--initial-cluster` flag).
	// This field cannot be updated.
	EtcdEnv []v1.EnvVar `json:"etcdEnv"`
}

func (c *ClusterSpec) Validate() error {
	if c.Backup == nil && c.Restore != nil {
		return ErrBackupUnsetRestoreSet
	}
	if c.Backup != nil && c.Restore != nil {
		if c.Backup.StorageType != c.Restore.StorageType {
			return errors.New("spec: backup and restore storage types are different")
		}
	}
	if c.Backup != nil {
		if err := c.Backup.Validate(); err != nil {
			return err
		}
	}
	if c.TLS != nil {
		if err := c.TLS.Validate(); err != nil {
			return err
		}
	}

	if c.Pod != nil {
		for k := range c.Pod.Labels {
			if k == "app" || strings.HasPrefix(k, "etcd_") {
				return errors.New("spec: pod labels contains reserved label")
			}
		}
	}
	return nil
}

// Cleanup cleans up user passed spec, e.g. defaulting, transforming fields.
// TODO: move this to admission controller
func (c *ClusterSpec) Cleanup() {
	if len(c.Version) == 0 {
		c.Version = defaultVersion
	}
	c.Version = strings.TrimLeft(c.Version, "v")
}

type ClusterPhase string

const (
	ClusterPhaseNone     ClusterPhase = ""
	ClusterPhaseCreating              = "Creating"
	ClusterPhaseRunning               = "Running"
	ClusterPhaseFailed                = "Failed"
)

type ClusterCondition struct {
	Type ClusterConditionType `json:"type"`

	Reason string `json:"reason"`

	TransitionTime string `json:"transitionTime"`
}

type ClusterConditionType string

const (
	ClusterConditionReady = "Ready"

	ClusterConditionRemovingDeadMember = "RemovingDeadMember"

	ClusterConditionRecovering = "Recovering"

	ClusterConditionScalingUp   = "ScalingUp"
	ClusterConditionScalingDown = "ScalingDown"

	ClusterConditionUpgrading = "Upgrading"
)

type ClusterStatus struct {
	// Phase is the cluster running phase
	Phase  ClusterPhase `json:"phase"`
	Reason string       `json:"reason"`

	// ControlPuased indicates the operator pauses the control of the cluster.
	ControlPaused bool `json:"controlPaused"`

	// Condition keeps ten most recent cluster conditions
	Conditions []ClusterCondition `json:"conditions"`

	// Size is the current size of the cluster
	Size int `json:"size"`
	// Members are the etcd members in the cluster
	Members MembersStatus `json:"members"`
	// CurrentVersion is the current cluster version
	CurrentVersion string `json:"currentVersion"`
	// TargetVersion is the version the cluster upgrading to.
	// If the cluster is not upgrading, TargetVersion is empty.
	TargetVersion string `json:"targetVersion"`

	// BackupServiceStatus is the status of the backup service.
	// BackupServiceStatus only exists when backup is enabled in the
	// cluster spec.
	BackupServiceStatus *BackupServiceStatus `json:"backupServiceStatus,omitempty"`
}

type MembersStatus struct {
	// Ready are the etcd members that are ready to serve requests
	// The member names are the same as the etcd pod names
	Ready []string `json:"ready,omitempty"`
	// Unready are the etcd members not ready to serve requests
	Unready []string `json:"unready,omitempty"`
}

func (cs ClusterStatus) Copy() ClusterStatus {
	newCS := ClusterStatus{}
	b, err := json.Marshal(cs)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(b, &newCS)
	if err != nil {
		panic(err)
	}
	return newCS
}

func (cs *ClusterStatus) IsFailed() bool {
	if cs == nil {
		return false
	}
	return cs.Phase == ClusterPhaseFailed
}

func (cs *ClusterStatus) SetPhase(p ClusterPhase) {
	cs.Phase = p
}

func (cs *ClusterStatus) PauseControl() {
	cs.ControlPaused = true
}

func (cs *ClusterStatus) Control() {
	cs.ControlPaused = false
}

func (cs *ClusterStatus) UpgradeVersionTo(v string) {
	cs.TargetVersion = v
}

func (cs *ClusterStatus) SetVersion(v string) {
	cs.TargetVersion = ""
	cs.CurrentVersion = v
}

func (cs *ClusterStatus) SetReason(r string) {
	cs.Reason = r
}

func (cs *ClusterStatus) AppendScalingUpCondition(from, to int) {
	c := ClusterCondition{
		Type:           ClusterConditionScalingUp,
		Reason:         scalingReason(from, to),
		TransitionTime: time.Now().Format(time.RFC3339),
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) AppendScalingDownCondition(from, to int) {
	c := ClusterCondition{
		Type:           ClusterConditionScalingDown,
		Reason:         scalingReason(from, to),
		TransitionTime: time.Now().Format(time.RFC3339),
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) AppendRecoveringCondition() {
	c := ClusterCondition{
		Type:           ClusterConditionRecovering,
		TransitionTime: time.Now().Format(time.RFC3339),
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) AppendUpgradingCondition(to string, member string) {
	reason := fmt.Sprintf("upgrading cluster member %s version to %v", member, to)

	c := ClusterCondition{
		Type:           ClusterConditionUpgrading,
		Reason:         reason,
		TransitionTime: time.Now().Format(time.RFC3339),
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) AppendRemovingDeadMember(name string) {
	reason := fmt.Sprintf("removing dead member %s", name)

	c := ClusterCondition{
		Type:           ClusterConditionRemovingDeadMember,
		Reason:         reason,
		TransitionTime: time.Now().Format(time.RFC3339),
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) SetReadyCondition() {
	c := ClusterCondition{
		Type:           ClusterConditionReady,
		TransitionTime: time.Now().Format(time.RFC3339),
	}

	if len(cs.Conditions) == 0 {
		cs.appendCondition(c)
		return
	}

	lastc := cs.Conditions[len(cs.Conditions)-1]
	if lastc.Type == ClusterConditionReady {
		return
	}
	cs.appendCondition(c)
}

func (cs *ClusterStatus) appendCondition(c ClusterCondition) {
	cs.Conditions = append(cs.Conditions, c)
	if len(cs.Conditions) > 10 {
		cs.Conditions = cs.Conditions[1:]
	}
}

func scalingReason(from, to int) string {
	return fmt.Sprintf("Current cluster size: %d, desired cluster size: %d", from, to)
}

type BackupStorageType string

const (
	BackupStorageTypeDefault          = ""
	BackupStorageTypePersistentVolume = "PersistentVolume"
	BackupStorageTypeS3               = "S3"
)

var errPVZeroSize = errors.New("PV backup should not have 0 size volume")

type BackupPolicy struct {
	// Pod defines the policy to create the backup pod.
	Pod *PodPolicy `json:"pod,omitempty"`

	// StorageType specifies the type of storage device to store backup files.
	// If it's not set by user, the default is "PersistentVolume".
	StorageType BackupStorageType `json:"storageType"`

	StorageSource `json:",inline"`

	// BackupIntervalInSecond specifies the interval between two backups.
	// The default interval is 1800 seconds.
	BackupIntervalInSecond int `json:"backupIntervalInSecond"`

	// If greater than 0, MaxBackups is the maximum number of backup files to retain.
	// If equal to 0, it means unlimited backups.
	// Otherwise, it is invalid.
	MaxBackups int `json:"maxBackups"`

	// CleanupBackupsOnClusterDelete tells whether to cleanup backup data if cluster is deleted.
	// By default, operator will keep the backup data.
	CleanupBackupsOnClusterDelete bool `json:"cleanupBackupsOnClusterDelete"`
}

func (bp *BackupPolicy) Validate() error {
	if bp.MaxBackups < 0 {
		return errors.New("MaxBackups value should be >= 0")
	}
	if bp.StorageType == BackupStorageTypePersistentVolume {
		if pv := bp.StorageSource.PV; pv == nil || pv.VolumeSizeInMB <= 0 {
			return errPVZeroSize
		}
	}
	return nil
}

type StorageSource struct {
	PV *PVSource `json:"pv,omitempty"`
	S3 *S3Source `json:"s3,omitempty"`
}

type PVSource struct {
	// VolumeSizeInMB specifies the required volume size to perform backups.
	// Operator will claim the required size before creating the etcd cluster for backup
	// purpose.
	// If the snapshot size is larger than the size specified, backup fails.
	VolumeSizeInMB int `json:"volumeSizeInMB"`
}

// TODO: support per cluster S3 Source configuration.
type S3Source struct {
	// The name of the AWS S3 bucket to store backups in.
	//
	// S3Bucket overwrites the default etcd operator wide bucket.
	S3Bucket string `json:"s3Bucket,omitempty"`

	// The name of the secret object that stores the AWS credential and config files.
	// The file name of the credential MUST be 'credentials'.
	// The file name of the config MUST be 'config'
	//
	// AWSCredentialSecret overwrites the default etcd operator wide AWS credential and
	// config.
	AWSSecret string `json:"awsSecret,omitempty"`
}

type BackupServiceStatus struct {
	// RecentBackup is status of the most recent backup created by
	// the backup service
	RecentBackup *BackupStatus `json:"recentBackup,omitempty"`

	// Backups is the totoal number of existing backups
	Backups int `json:"backups"`

	// BackupSize is the total size of existing backups in MB.
	BackupSize float64 `json:"backupSize"`
}

type BackupStatus struct {
	// Creation time of the backup.
	CreationTime string `json:"creationTime"`

	// Size is the size of the backup in MB.
	Size float64 `json:"size"`

	// Version is the version of the backup cluster.
	Version string `json:"version"`

	// TimeTookInSecond is the total time took to create the backup.
	TimeTookInSecond int `json:"timeTookInSecond"`
}

// TLSPolicy defines the TLS policy of an etcd cluster
type TLSPolicy struct {
	// StaticTLS enables user to generate static x509 certificates and keys,
	// put them into Kubernetes secrets, and specify them into here.
	Static *StaticTLS `json:"static"`
}

type StaticTLS struct {
	// Member contains secrets containing TLS certs used by each etcd member pod.
	Member *MemberSecret `json:"member"`
	// OperatorSecret is the secret containing TLS certs used by operator to
	// talk securely to this cluster.
	OperatorSecret string `json:"operatorSecret"`
}

type MemberSecret struct {
	// PeerSecret is the secret containing TLS certs used by each etcd member pod
	// for the communication between etcd peers.
	PeerSecret string `json:"peerSecret"`
	// ClientSecret is the secret containing TLS certs used by each etcd member pod
	// for the communication between etcd and its clients.
	ClientSecret string `json:"clientSecret"`
}

func (tp *TLSPolicy) Validate() error {
	if tp.Static == nil {
		return nil
	}
	st := tp.Static

	if len(st.OperatorSecret) != 0 {
		if len(st.Member.ClientSecret) == 0 {
			return errors.New("operator secret set but member clientSecret not set")
		}
	} else if st.Member != nil && len(st.Member.ClientSecret) != 0 {
		return errors.New("member clientSecret set but operator secret not set")
	}
	return nil
}

func (tp *TLSPolicy) IsSecureClient() bool {
	if tp == nil || tp.Static == nil {
		return false
	}
	return len(tp.Static.OperatorSecret) != 0
}

func (tp *TLSPolicy) IsSecurePeer() bool {
	if tp == nil || tp.Static == nil || tp.Static.Member == nil {
		return false
	}
	return len(tp.Static.Member.PeerSecret) != 0
}

// ClusterList is a list of etcd clusters.
type ClusterList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Metadata metav1.ListMeta `json:"metadata,omitempty"`
	// Items is a list of third party objects
	Items []Cluster `json:"items"`
}

// There is known issue with TPR in client-go:
//   https://github.com/kubernetes/client-go/issues/8
// Workarounds:
// - We include `Metadata` field in object explicitly.
// - we have the code below to work around a known problem with third-party resources and ugorji.

type ClusterListCopy ClusterList
type ClusterCopy Cluster

func (c *Cluster) UnmarshalJSON(data []byte) error {
	tmp := ClusterCopy{}
	err := json.Unmarshal(data, &tmp)
	if err != nil {
		return err
	}
	tmp2 := Cluster(tmp)
	*c = tmp2
	return nil
}

func (cl *ClusterList) UnmarshalJSON(data []byte) error {
	tmp := ClusterListCopy{}
	err := json.Unmarshal(data, &tmp)
	if err != nil {
		return err
	}
	tmp2 := ClusterList(tmp)
	*cl = tmp2
	return nil
}

type SelfHostedPolicy struct {
	// BootMemberClientEndpoint specifies a bootstrap member for the cluster.
	// If there is no bootstrap member, a completely new cluster will be created.
	// The boot member will be removed from the cluster once the self-hosted cluster
	// setup successfully.
	BootMemberClientEndpoint string `json:"bootMemberClientEndpoint"`
}
