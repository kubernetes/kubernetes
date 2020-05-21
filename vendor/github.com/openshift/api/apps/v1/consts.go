package v1

const (
	// DeploymentStatusReasonAnnotation represents the reason for deployment being in a given state
	// Used for specifying the reason for cancellation or failure of a deployment
	// This is on replication controller set by deployer controller.
	DeploymentStatusReasonAnnotation = "openshift.io/deployment.status-reason"

	// DeploymentPodAnnotation is an annotation on a deployment (a ReplicationController). The
	// annotation value is the name of the deployer Pod which will act upon the ReplicationController
	// to implement the deployment behavior.
	// This is set on replication controller by deployer controller.
	DeploymentPodAnnotation = "openshift.io/deployer-pod.name"

	// DeploymentConfigAnnotation is an annotation name used to correlate a deployment with the
	// DeploymentConfig on which the deployment is based.
	// This is set on replication controller pod template by deployer controller.
	DeploymentConfigAnnotation = "openshift.io/deployment-config.name"

	// DeploymentCancelledAnnotation indicates that the deployment has been cancelled
	// The annotation value does not matter and its mere presence indicates cancellation.
	// This is set on replication controller by deployment config controller or oc rollout cancel command.
	DeploymentCancelledAnnotation = "openshift.io/deployment.cancelled"

	// DeploymentEncodedConfigAnnotation is an annotation name used to retrieve specific encoded
	// DeploymentConfig on which a given deployment is based.
	// This is set on replication controller by deployer controller.
	DeploymentEncodedConfigAnnotation = "openshift.io/encoded-deployment-config"

	// DeploymentVersionAnnotation is an annotation on a deployment (a ReplicationController). The
	// annotation value is the LatestVersion value of the DeploymentConfig which was the basis for
	// the deployment.
	// This is set on replication controller pod template by deployment config controller.
	DeploymentVersionAnnotation = "openshift.io/deployment-config.latest-version"

	// DeployerPodForDeploymentLabel is a label which groups pods related to a
	// deployment. The value is a deployment name. The deployer pod and hook pods
	// created by the internal strategies will have this label. Custom
	// strategies can apply this label to any pods they create, enabling
	// platform-provided cancellation and garbage collection support.
	// This is set on deployer pod by deployer controller.
	DeployerPodForDeploymentLabel = "openshift.io/deployer-pod-for.name"

	// DeploymentStatusAnnotation is an annotation name used to retrieve the DeploymentPhase of
	// a deployment.
	// This is set on replication controller by deployer controller.
	DeploymentStatusAnnotation = "openshift.io/deployment.phase"
)

type DeploymentConditionReason string

var (
	// ReplicationControllerUpdatedReason is added in a deployment config when one of its replication
	// controllers is updated as part of the rollout process.
	ReplicationControllerUpdatedReason DeploymentConditionReason = "ReplicationControllerUpdated"

	// ReplicationControllerCreateError is added in a deployment config when it cannot create a new replication
	// controller.
	ReplicationControllerCreateErrorReason DeploymentConditionReason = "ReplicationControllerCreateError"

	// ReplicationControllerCreatedReason is added in a deployment config when it creates a new replication
	// controller.
	NewReplicationControllerCreatedReason DeploymentConditionReason = "NewReplicationControllerCreated"

	// NewReplicationControllerAvailableReason is added in a deployment config when its newest replication controller is made
	// available ie. the number of new pods that have passed readiness checks and run for at least
	// minReadySeconds is at least the minimum available pods that need to run for the deployment config.
	NewReplicationControllerAvailableReason DeploymentConditionReason = "NewReplicationControllerAvailable"

	// ProgressDeadlineExceededReason is added in a deployment config when its newest replication controller fails to show
	// any progress within the given deadline (progressDeadlineSeconds).
	ProgressDeadlineExceededReason DeploymentConditionReason = "ProgressDeadlineExceeded"

	// DeploymentConfigPausedReason is added in a deployment config when it is paused. Lack of progress shouldn't be
	// estimated once a deployment config is paused.
	DeploymentConfigPausedReason DeploymentConditionReason = "DeploymentConfigPaused"

	// DeploymentConfigResumedReason is added in a deployment config when it is resumed. Useful for not failing accidentally
	// deployment configs that paused amidst a rollout.
	DeploymentConfigResumedReason DeploymentConditionReason = "DeploymentConfigResumed"

	// RolloutCancelledReason is added in a deployment config when its newest rollout was
	// interrupted by cancellation.
	RolloutCancelledReason DeploymentConditionReason = "RolloutCancelled"
)

// DeploymentStatus describes the possible states a deployment can be in.
type DeploymentStatus string

var (

	// DeploymentStatusNew means the deployment has been accepted but not yet acted upon.
	DeploymentStatusNew DeploymentStatus = "New"

	// DeploymentStatusPending means the deployment been handed over to a deployment strategy,
	// but the strategy has not yet declared the deployment to be running.
	DeploymentStatusPending DeploymentStatus = "Pending"

	// DeploymentStatusRunning means the deployment strategy has reported the deployment as
	// being in-progress.
	DeploymentStatusRunning DeploymentStatus = "Running"

	// DeploymentStatusComplete means the deployment finished without an error.
	DeploymentStatusComplete DeploymentStatus = "Complete"

	// DeploymentStatusFailed means the deployment finished with an error.
	DeploymentStatusFailed DeploymentStatus = "Failed"
)
