package v1

import (
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:method=UpdateDetails,verb=update,subresource=details
// +genclient:method=Clone,verb=create,subresource=clone,input=BuildRequest
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Build encapsulates the inputs needed to produce a new deployable image, as well as
// the status of the execution and a reference to the Pod which executed the build.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Build struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is all the inputs used to execute the build.
	Spec BuildSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status is the current status of the build.
	// +optional
	Status BuildStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// BuildSpec has the information to represent a build and also additional
// information about a build
type BuildSpec struct {
	// CommonSpec is the information that represents a build
	CommonSpec `json:",inline" protobuf:"bytes,1,opt,name=commonSpec"`

	// triggeredBy describes which triggers started the most recent update to the
	// build configuration and contains information about those triggers.
	TriggeredBy []BuildTriggerCause `json:"triggeredBy,omitempty" protobuf:"bytes,2,rep,name=triggeredBy"`
}

// OptionalNodeSelector is a map that may also be left nil to distinguish between set and unset.
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type OptionalNodeSelector map[string]string

func (t OptionalNodeSelector) String() string {
	return fmt.Sprintf("%v", map[string]string(t))
}

// CommonSpec encapsulates all the inputs necessary to represent a build.
type CommonSpec struct {
	// serviceAccount is the name of the ServiceAccount to use to run the pod
	// created by this build.
	// The pod will be allowed to use secrets referenced by the ServiceAccount
	ServiceAccount string `json:"serviceAccount,omitempty" protobuf:"bytes,1,opt,name=serviceAccount"`

	// source describes the SCM in use.
	Source BuildSource `json:"source,omitempty" protobuf:"bytes,2,opt,name=source"`

	// revision is the information from the source for a specific repo snapshot.
	// This is optional.
	Revision *SourceRevision `json:"revision,omitempty" protobuf:"bytes,3,opt,name=revision"`

	// strategy defines how to perform a build.
	Strategy BuildStrategy `json:"strategy" protobuf:"bytes,4,opt,name=strategy"`

	// output describes the container image the Strategy should produce.
	Output BuildOutput `json:"output,omitempty" protobuf:"bytes,5,opt,name=output"`

	// resources computes resource requirements to execute the build.
	Resources corev1.ResourceRequirements `json:"resources,omitempty" protobuf:"bytes,6,opt,name=resources"`

	// postCommit is a build hook executed after the build output image is
	// committed, before it is pushed to a registry.
	PostCommit BuildPostCommitSpec `json:"postCommit,omitempty" protobuf:"bytes,7,opt,name=postCommit"`

	// completionDeadlineSeconds is an optional duration in seconds, counted from
	// the time when a build pod gets scheduled in the system, that the build may
	// be active on a node before the system actively tries to terminate the
	// build; value must be positive integer
	CompletionDeadlineSeconds *int64 `json:"completionDeadlineSeconds,omitempty" protobuf:"varint,8,opt,name=completionDeadlineSeconds"`

	// nodeSelector is a selector which must be true for the build pod to fit on a node
	// If nil, it can be overridden by default build nodeselector values for the cluster.
	// If set to an empty map or a map with any values, default build nodeselector values
	// are ignored.
	// +optional
	NodeSelector OptionalNodeSelector `json:"nodeSelector" protobuf:"bytes,9,name=nodeSelector"`

	// mountTrustedCA bind mounts the cluster's trusted certificate authorities, as defined in
	// the cluster's proxy configuration, into the build. This lets processes within a build trust
	// components signed by custom PKI certificate authorities, such as private artifact
	// repositories and HTTPS proxies.
	//
	// When this field is set to true, the contents of `/etc/pki/ca-trust` within the build are
	// managed by the build container, and any changes to this directory or its subdirectories (for
	// example - within a Dockerfile `RUN` instruction) are not persisted in the build's output image.
	MountTrustedCA *bool `json:"mountTrustedCA,omitempty" protobuf:"varint,10,opt,name=mountTrustedCA"`
}

// BuildTriggerCause holds information about a triggered build. It is used for
// displaying build trigger data for each build and build configuration in oc
// describe. It is also used to describe which triggers led to the most recent
// update in the build configuration.
type BuildTriggerCause struct {
	// message is used to store a human readable message for why the build was
	// triggered. E.g.: "Manually triggered by user", "Configuration change",etc.
	Message string `json:"message,omitempty" protobuf:"bytes,1,opt,name=message"`

	// genericWebHook holds data about a builds generic webhook trigger.
	GenericWebHook *GenericWebHookCause `json:"genericWebHook,omitempty" protobuf:"bytes,2,opt,name=genericWebHook"`

	// gitHubWebHook represents data for a GitHub webhook that fired a
	//specific build.
	GitHubWebHook *GitHubWebHookCause `json:"githubWebHook,omitempty" protobuf:"bytes,3,opt,name=githubWebHook"`

	// imageChangeBuild stores information about an imagechange event
	// that triggered a new build.
	ImageChangeBuild *ImageChangeCause `json:"imageChangeBuild,omitempty" protobuf:"bytes,4,opt,name=imageChangeBuild"`

	// GitLabWebHook represents data for a GitLab webhook that fired a specific
	// build.
	GitLabWebHook *GitLabWebHookCause `json:"gitlabWebHook,omitempty" protobuf:"bytes,5,opt,name=gitlabWebHook"`

	// BitbucketWebHook represents data for a Bitbucket webhook that fired a
	// specific build.
	BitbucketWebHook *BitbucketWebHookCause `json:"bitbucketWebHook,omitempty" protobuf:"bytes,6,opt,name=bitbucketWebHook"`
}

// GenericWebHookCause holds information about a generic WebHook that
// triggered a build.
type GenericWebHookCause struct {
	// revision is an optional field that stores the git source revision
	// information of the generic webhook trigger when it is available.
	Revision *SourceRevision `json:"revision,omitempty" protobuf:"bytes,1,opt,name=revision"`

	// secret is the obfuscated webhook secret that triggered a build.
	Secret string `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`
}

// GitHubWebHookCause has information about a GitHub webhook that triggered a
// build.
type GitHubWebHookCause struct {
	// revision is the git revision information of the trigger.
	Revision *SourceRevision `json:"revision,omitempty" protobuf:"bytes,1,opt,name=revision"`

	// secret is the obfuscated webhook secret that triggered a build.
	Secret string `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`
}

// CommonWebHookCause factors out the identical format of these webhook
// causes into struct so we can share it in the specific causes;  it is too late for
// GitHub and Generic but we can leverage this pattern with GitLab and Bitbucket.
type CommonWebHookCause struct {
	// Revision is the git source revision information of the trigger.
	Revision *SourceRevision `json:"revision,omitempty" protobuf:"bytes,1,opt,name=revision"`

	// Secret is the obfuscated webhook secret that triggered a build.
	Secret string `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`
}

// GitLabWebHookCause has information about a GitLab webhook that triggered a
// build.
type GitLabWebHookCause struct {
	CommonWebHookCause `json:",inline" protobuf:"bytes,1,opt,name=commonSpec"`
}

// BitbucketWebHookCause has information about a Bitbucket webhook that triggered a
// build.
type BitbucketWebHookCause struct {
	CommonWebHookCause `json:",inline" protobuf:"bytes,1,opt,name=commonSpec"`
}

// ImageChangeCause contains information about the image that triggered a
// build
type ImageChangeCause struct {
	// imageID is the ID of the image that triggered a new build.
	ImageID string `json:"imageID,omitempty" protobuf:"bytes,1,opt,name=imageID"`

	// fromRef contains detailed information about an image that triggered a
	// build.
	FromRef *corev1.ObjectReference `json:"fromRef,omitempty" protobuf:"bytes,2,opt,name=fromRef"`
}

// BuildStatus contains the status of a build
type BuildStatus struct {
	// phase is the point in the build lifecycle. Possible values are
	// "New", "Pending", "Running", "Complete", "Failed", "Error", and "Cancelled".
	Phase BuildPhase `json:"phase" protobuf:"bytes,1,opt,name=phase,casttype=BuildPhase"`

	// cancelled describes if a cancel event was triggered for the build.
	Cancelled bool `json:"cancelled,omitempty" protobuf:"varint,2,opt,name=cancelled"`

	// reason is a brief CamelCase string that describes any failure and is meant for machine parsing and tidy display in the CLI.
	Reason StatusReason `json:"reason,omitempty" protobuf:"bytes,3,opt,name=reason,casttype=StatusReason"`

	// message is a human-readable message indicating details about why the build has this status.
	Message string `json:"message,omitempty" protobuf:"bytes,4,opt,name=message"`

	// startTimestamp is a timestamp representing the server time when this Build started
	// running in a Pod.
	// It is represented in RFC3339 form and is in UTC.
	StartTimestamp *metav1.Time `json:"startTimestamp,omitempty" protobuf:"bytes,5,opt,name=startTimestamp"`

	// completionTimestamp is a timestamp representing the server time when this Build was
	// finished, whether that build failed or succeeded.  It reflects the time at which
	// the Pod running the Build terminated.
	// It is represented in RFC3339 form and is in UTC.
	CompletionTimestamp *metav1.Time `json:"completionTimestamp,omitempty" protobuf:"bytes,6,opt,name=completionTimestamp"`

	// duration contains time.Duration object describing build time.
	Duration time.Duration `json:"duration,omitempty" protobuf:"varint,7,opt,name=duration,casttype=time.Duration"`

	// outputDockerImageReference contains a reference to the container image that
	// will be built by this build. Its value is computed from
	// Build.Spec.Output.To, and should include the registry address, so that
	// it can be used to push and pull the image.
	OutputDockerImageReference string `json:"outputDockerImageReference,omitempty" protobuf:"bytes,8,opt,name=outputDockerImageReference"`

	// config is an ObjectReference to the BuildConfig this Build is based on.
	Config *corev1.ObjectReference `json:"config,omitempty" protobuf:"bytes,9,opt,name=config"`

	// output describes the container image the build has produced.
	Output BuildStatusOutput `json:"output,omitempty" protobuf:"bytes,10,opt,name=output"`

	// stages contains details about each stage that occurs during the build
	// including start time, duration (in milliseconds), and the steps that
	// occured within each stage.
	Stages []StageInfo `json:"stages,omitempty" protobuf:"bytes,11,opt,name=stages"`

	// logSnippet is the last few lines of the build log.  This value is only set for builds that failed.
	LogSnippet string `json:"logSnippet,omitempty" protobuf:"bytes,12,opt,name=logSnippet"`

	// Conditions represents the latest available observations of a build's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []BuildCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,13,rep,name=conditions"`
}

// StageInfo contains details about a build stage.
type StageInfo struct {
	// name is a unique identifier for each build stage that occurs.
	Name StageName `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`

	// startTime is a timestamp representing the server time when this Stage started.
	// It is represented in RFC3339 form and is in UTC.
	StartTime metav1.Time `json:"startTime,omitempty" protobuf:"bytes,2,opt,name=startTime"`

	// durationMilliseconds identifies how long the stage took
	// to complete in milliseconds.
	// Note: the duration of a stage can exceed the sum of the duration of the steps within
	// the stage as not all actions are accounted for in explicit build steps.
	DurationMilliseconds int64 `json:"durationMilliseconds,omitempty" protobuf:"varint,3,opt,name=durationMilliseconds"`

	// steps contains details about each step that occurs during a build stage
	// including start time and duration in milliseconds.
	Steps []StepInfo `json:"steps,omitempty" protobuf:"bytes,4,opt,name=steps"`
}

// StageName is the unique identifier for each build stage.
type StageName string

// Valid values for StageName
const (
	// StageFetchInputs fetches any inputs such as source code.
	StageFetchInputs StageName = "FetchInputs"

	// StagePullImages pulls any images that are needed such as
	// base images or input images.
	StagePullImages StageName = "PullImages"

	// StageBuild performs the steps necessary to build the image.
	StageBuild StageName = "Build"

	// StagePostCommit executes any post commit steps.
	StagePostCommit StageName = "PostCommit"

	// StagePushImage pushes the image to the node.
	StagePushImage StageName = "PushImage"
)

// StepInfo contains details about a build step.
type StepInfo struct {
	// name is a unique identifier for each build step.
	Name StepName `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`

	// startTime is a timestamp representing the server time when this Step started.
	// it is represented in RFC3339 form and is in UTC.
	StartTime metav1.Time `json:"startTime,omitempty" protobuf:"bytes,2,opt,name=startTime"`

	// durationMilliseconds identifies how long the step took
	// to complete in milliseconds.
	DurationMilliseconds int64 `json:"durationMilliseconds,omitempty" protobuf:"varint,3,opt,name=durationMilliseconds"`
}

// StepName is a unique identifier for each build step.
type StepName string

// Valid values for StepName
const (
	// StepExecPostCommitHook executes the buildconfigs post commit hook.
	StepExecPostCommitHook StepName = "RunPostCommitHook"

	// StepFetchGitSource fetches source code for the build.
	StepFetchGitSource StepName = "FetchGitSource"

	// StepPullBaseImage pulls a base image for the build.
	StepPullBaseImage StepName = "PullBaseImage"

	// StepPullInputImage pulls an input image for the build.
	StepPullInputImage StepName = "PullInputImage"

	// StepPushImage pushes an image to the registry.
	StepPushImage StepName = "PushImage"

	// StepPushDockerImage pushes a container image to the registry.
	StepPushDockerImage StepName = "PushDockerImage"

	//StepDockerBuild performs the container image build
	StepDockerBuild StepName = "DockerBuild"
)

// BuildPhase represents the status of a build at a point in time.
type BuildPhase string

// Valid values for BuildPhase.
const (
	// BuildPhaseNew is automatically assigned to a newly created build.
	BuildPhaseNew BuildPhase = "New"

	// BuildPhasePending indicates that a pod name has been assigned and a build is
	// about to start running.
	BuildPhasePending BuildPhase = "Pending"

	// BuildPhaseRunning indicates that a pod has been created and a build is running.
	BuildPhaseRunning BuildPhase = "Running"

	// BuildPhaseComplete indicates that a build has been successful.
	BuildPhaseComplete BuildPhase = "Complete"

	// BuildPhaseFailed indicates that a build has executed and failed.
	BuildPhaseFailed BuildPhase = "Failed"

	// BuildPhaseError indicates that an error prevented the build from executing.
	BuildPhaseError BuildPhase = "Error"

	// BuildPhaseCancelled indicates that a running/pending build was stopped from executing.
	BuildPhaseCancelled BuildPhase = "Cancelled"
)

type BuildConditionType string

// BuildCondition describes the state of a build at a certain point.
type BuildCondition struct {
	// Type of build condition.
	Type BuildConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=BuildConditionType"`
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

// StatusReason is a brief CamelCase string that describes a temporary or
// permanent build error condition, meant for machine parsing and tidy display
// in the CLI.
type StatusReason string

// BuildStatusOutput contains the status of the built image.
type BuildStatusOutput struct {
	// to describes the status of the built image being pushed to a registry.
	To *BuildStatusOutputTo `json:"to,omitempty" protobuf:"bytes,1,opt,name=to"`
}

// BuildStatusOutputTo describes the status of the built image with regards to
// image registry to which it was supposed to be pushed.
type BuildStatusOutputTo struct {
	// imageDigest is the digest of the built container image. The digest uniquely
	// identifies the image in the registry to which it was pushed.
	//
	// Please note that this field may not always be set even if the push
	// completes successfully - e.g. when the registry returns no digest or
	// returns it in a format that the builder doesn't understand.
	ImageDigest string `json:"imageDigest,omitempty" protobuf:"bytes,1,opt,name=imageDigest"`
}

// BuildSourceType is the type of SCM used.
type BuildSourceType string

// Valid values for BuildSourceType.
const (
	//BuildSourceGit instructs a build to use a Git source control repository as the build input.
	BuildSourceGit BuildSourceType = "Git"
	// BuildSourceDockerfile uses a Dockerfile as the start of a build
	BuildSourceDockerfile BuildSourceType = "Dockerfile"
	// BuildSourceBinary indicates the build will accept a Binary file as input.
	BuildSourceBinary BuildSourceType = "Binary"
	// BuildSourceImage indicates the build will accept an image as input
	BuildSourceImage BuildSourceType = "Image"
	// BuildSourceNone indicates the build has no predefined input (only valid for Source and Custom Strategies)
	BuildSourceNone BuildSourceType = "None"
)

// BuildSource is the SCM used for the build.
type BuildSource struct {
	// type of build input to accept
	// +k8s:conversion-gen=false
	// +optional
	Type BuildSourceType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=BuildSourceType"`

	// binary builds accept a binary as their input. The binary is generally assumed to be a tar,
	// gzipped tar, or zip file depending on the strategy. For container image builds, this is the build
	// context and an optional Dockerfile may be specified to override any Dockerfile in the
	// build context. For Source builds, this is assumed to be an archive as described above. For
	// Source and container image builds, if binary.asFile is set the build will receive a directory with
	// a single file. contextDir may be used when an archive is provided. Custom builds will
	// receive this binary as input on STDIN.
	Binary *BinaryBuildSource `json:"binary,omitempty" protobuf:"bytes,2,opt,name=binary"`

	// dockerfile is the raw contents of a Dockerfile which should be built. When this option is
	// specified, the FROM may be modified based on your strategy base image and additional ENV
	// stanzas from your strategy environment will be added after the FROM, but before the rest
	// of your Dockerfile stanzas. The Dockerfile source type may be used with other options like
	// git - in those cases the Git repo will have any innate Dockerfile replaced in the context
	// dir.
	Dockerfile *string `json:"dockerfile,omitempty" protobuf:"bytes,3,opt,name=dockerfile"`

	// git contains optional information about git build source
	Git *GitBuildSource `json:"git,omitempty" protobuf:"bytes,4,opt,name=git"`

	// images describes a set of images to be used to provide source for the build
	Images []ImageSource `json:"images,omitempty" protobuf:"bytes,5,rep,name=images"`

	// contextDir specifies the sub-directory where the source code for the application exists.
	// This allows to have buildable sources in directory other than root of
	// repository.
	ContextDir string `json:"contextDir,omitempty" protobuf:"bytes,6,opt,name=contextDir"`

	// sourceSecret is the name of a Secret that would be used for setting
	// up the authentication for cloning private repository.
	// The secret contains valid credentials for remote repository, where the
	// data's key represent the authentication method to be used and value is
	// the base64 encoded credentials. Supported auth methods are: ssh-privatekey.
	SourceSecret *corev1.LocalObjectReference `json:"sourceSecret,omitempty" protobuf:"bytes,7,opt,name=sourceSecret"`

	// secrets represents a list of secrets and their destinations that will
	// be used only for the build.
	Secrets []SecretBuildSource `json:"secrets,omitempty" protobuf:"bytes,8,rep,name=secrets"`

	// configMaps represents a list of configMaps and their destinations that will
	// be used for the build.
	ConfigMaps []ConfigMapBuildSource `json:"configMaps,omitempty" protobuf:"bytes,9,rep,name=configMaps"`
}

// ImageSource is used to describe build source that will be extracted from an image or used during a
// multi stage build. A reference of type ImageStreamTag, ImageStreamImage or DockerImage may be used.
// A pull secret can be specified to pull the image from an external registry or override the default
// service account secret if pulling from the internal registry. Image sources can either be used to
// extract content from an image and place it into the build context along with the repository source,
// or used directly during a multi-stage container image build to allow content to be copied without overwriting
// the contents of the repository source (see the 'paths' and 'as' fields).
type ImageSource struct {
	// from is a reference to an ImageStreamTag, ImageStreamImage, or DockerImage to
	// copy source from.
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`

	// A list of image names that this source will be used in place of during a multi-stage container image
	// build. For instance, a Dockerfile that uses "COPY --from=nginx:latest" will first check for an image
	// source that has "nginx:latest" in this field before attempting to pull directly. If the Dockerfile
	// does not reference an image source it is ignored. This field and paths may both be set, in which case
	// the contents will be used twice.
	// +optional
	As []string `json:"as,omitempty" protobuf:"bytes,4,rep,name=as"`

	// paths is a list of source and destination paths to copy from the image. This content will be copied
	// into the build context prior to starting the build. If no paths are set, the build context will
	// not be altered.
	// +optional
	Paths []ImageSourcePath `json:"paths,omitempty" protobuf:"bytes,2,rep,name=paths"`

	// pullSecret is a reference to a secret to be used to pull the image from a registry
	// If the image is pulled from the OpenShift registry, this field does not need to be set.
	PullSecret *corev1.LocalObjectReference `json:"pullSecret,omitempty" protobuf:"bytes,3,opt,name=pullSecret"`
}

// ImageSourcePath describes a path to be copied from a source image and its destination within the build directory.
type ImageSourcePath struct {
	// sourcePath is the absolute path of the file or directory inside the image to
	// copy to the build directory.  If the source path ends in /. then the content of
	// the directory will be copied, but the directory itself will not be created at the
	// destination.
	SourcePath string `json:"sourcePath" protobuf:"bytes,1,opt,name=sourcePath"`

	// destinationDir is the relative directory within the build directory
	// where files copied from the image are placed.
	DestinationDir string `json:"destinationDir" protobuf:"bytes,2,opt,name=destinationDir"`
}

// SecretBuildSource describes a secret and its destination directory that will be
// used only at the build time. The content of the secret referenced here will
// be copied into the destination directory instead of mounting.
type SecretBuildSource struct {
	// secret is a reference to an existing secret that you want to use in your
	// build.
	Secret corev1.LocalObjectReference `json:"secret" protobuf:"bytes,1,opt,name=secret"`

	// destinationDir is the directory where the files from the secret should be
	// available for the build time.
	// For the Source build strategy, these will be injected into a container
	// where the assemble script runs. Later, when the script finishes, all files
	// injected will be truncated to zero length.
	// For the container image build strategy, these will be copied into the build
	// directory, where the Dockerfile is located, so users can ADD or COPY them
	// during container image build.
	DestinationDir string `json:"destinationDir,omitempty" protobuf:"bytes,2,opt,name=destinationDir"`
}

// ConfigMapBuildSource describes a configmap and its destination directory that will be
// used only at the build time. The content of the configmap referenced here will
// be copied into the destination directory instead of mounting.
type ConfigMapBuildSource struct {
	// configMap is a reference to an existing configmap that you want to use in your
	// build.
	ConfigMap corev1.LocalObjectReference `json:"configMap" protobuf:"bytes,1,opt,name=configMap"`

	// destinationDir is the directory where the files from the configmap should be
	// available for the build time.
	// For the Source build strategy, these will be injected into a container
	// where the assemble script runs.
	// For the container image build strategy, these will be copied into the build
	// directory, where the Dockerfile is located, so users can ADD or COPY them
	// during container image build.
	DestinationDir string `json:"destinationDir,omitempty" protobuf:"bytes,2,opt,name=destinationDir"`
}

// BinaryBuildSource describes a binary file to be used for the Docker and Source build strategies,
// where the file will be extracted and used as the build source.
type BinaryBuildSource struct {
	// asFile indicates that the provided binary input should be considered a single file
	// within the build input. For example, specifying "webapp.war" would place the provided
	// binary as `/webapp.war` for the builder. If left empty, the Docker and Source build
	// strategies assume this file is a zip, tar, or tar.gz file and extract it as the source.
	// The custom strategy receives this binary as standard input. This filename may not
	// contain slashes or be '..' or '.'.
	AsFile string `json:"asFile,omitempty" protobuf:"bytes,1,opt,name=asFile"`
}

// SourceRevision is the revision or commit information from the source for the build
type SourceRevision struct {
	// type of the build source, may be one of 'Source', 'Dockerfile', 'Binary', or 'Images'
	// +k8s:conversion-gen=false
	Type BuildSourceType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=BuildSourceType"`

	// Git contains information about git-based build source
	Git *GitSourceRevision `json:"git,omitempty" protobuf:"bytes,2,opt,name=git"`
}

// GitSourceRevision is the commit information from a git source for a build
type GitSourceRevision struct {
	// commit is the commit hash identifying a specific commit
	Commit string `json:"commit,omitempty" protobuf:"bytes,1,opt,name=commit"`

	// author is the author of a specific commit
	Author SourceControlUser `json:"author,omitempty" protobuf:"bytes,2,opt,name=author"`

	// committer is the committer of a specific commit
	Committer SourceControlUser `json:"committer,omitempty" protobuf:"bytes,3,opt,name=committer"`

	// message is the description of a specific commit
	Message string `json:"message,omitempty" protobuf:"bytes,4,opt,name=message"`
}

// ProxyConfig defines what proxies to use for an operation
type ProxyConfig struct {
	// httpProxy is a proxy used to reach the git repository over http
	HTTPProxy *string `json:"httpProxy,omitempty" protobuf:"bytes,3,opt,name=httpProxy"`

	// httpsProxy is a proxy used to reach the git repository over https
	HTTPSProxy *string `json:"httpsProxy,omitempty" protobuf:"bytes,4,opt,name=httpsProxy"`

	// noProxy is the list of domains for which the proxy should not be used
	NoProxy *string `json:"noProxy,omitempty" protobuf:"bytes,5,opt,name=noProxy"`
}

// GitBuildSource defines the parameters of a Git SCM
type GitBuildSource struct {
	// uri points to the source that will be built. The structure of the source
	// will depend on the type of build to run
	URI string `json:"uri" protobuf:"bytes,1,opt,name=uri"`

	// ref is the branch/tag/ref to build.
	Ref string `json:"ref,omitempty" protobuf:"bytes,2,opt,name=ref"`

	// proxyConfig defines the proxies to use for the git clone operation. Values
	// not set here are inherited from cluster-wide build git proxy settings.
	ProxyConfig `json:",inline" protobuf:"bytes,3,opt,name=proxyConfig"`
}

// SourceControlUser defines the identity of a user of source control
type SourceControlUser struct {
	// name of the source control user
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`

	// email of the source control user
	Email string `json:"email,omitempty" protobuf:"bytes,2,opt,name=email"`
}

// BuildStrategy contains the details of how to perform a build.
type BuildStrategy struct {
	// type is the kind of build strategy.
	// +k8s:conversion-gen=false
	// +optional
	Type BuildStrategyType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=BuildStrategyType"`

	// dockerStrategy holds the parameters to the container image build strategy.
	DockerStrategy *DockerBuildStrategy `json:"dockerStrategy,omitempty" protobuf:"bytes,2,opt,name=dockerStrategy"`

	// sourceStrategy holds the parameters to the Source build strategy.
	SourceStrategy *SourceBuildStrategy `json:"sourceStrategy,omitempty" protobuf:"bytes,3,opt,name=sourceStrategy"`

	// customStrategy holds the parameters to the Custom build strategy
	CustomStrategy *CustomBuildStrategy `json:"customStrategy,omitempty" protobuf:"bytes,4,opt,name=customStrategy"`

	// JenkinsPipelineStrategy holds the parameters to the Jenkins Pipeline build strategy.
	// Deprecated: use OpenShift Pipelines
	JenkinsPipelineStrategy *JenkinsPipelineBuildStrategy `json:"jenkinsPipelineStrategy,omitempty" protobuf:"bytes,5,opt,name=jenkinsPipelineStrategy"`
}

// BuildStrategyType describes a particular way of performing a build.
type BuildStrategyType string

// Valid values for BuildStrategyType.
const (
	// DockerBuildStrategyType performs builds using a Dockerfile.
	DockerBuildStrategyType BuildStrategyType = "Docker"

	// SourceBuildStrategyType performs builds build using Source To Images with a Git repository
	// and a builder image.
	SourceBuildStrategyType BuildStrategyType = "Source"

	// CustomBuildStrategyType performs builds using custom builder container image.
	CustomBuildStrategyType BuildStrategyType = "Custom"

	// JenkinsPipelineBuildStrategyType indicates the build will run via Jenkine Pipeline.
	JenkinsPipelineBuildStrategyType BuildStrategyType = "JenkinsPipeline"
)

// CustomBuildStrategy defines input parameters specific to Custom build.
type CustomBuildStrategy struct {
	// from is reference to an DockerImage, ImageStreamTag, or ImageStreamImage from which
	// the container image should be pulled
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`

	// pullSecret is the name of a Secret that would be used for setting up
	// the authentication for pulling the container images from the private Docker
	// registries
	PullSecret *corev1.LocalObjectReference `json:"pullSecret,omitempty" protobuf:"bytes,2,opt,name=pullSecret"`

	// env contains additional environment variables you want to pass into a builder container.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,3,rep,name=env"`

	// exposeDockerSocket will allow running Docker commands (and build container images) from
	// inside the container.
	// TODO: Allow admins to enforce 'false' for this option
	ExposeDockerSocket bool `json:"exposeDockerSocket,omitempty" protobuf:"varint,4,opt,name=exposeDockerSocket"`

	// forcePull describes if the controller should configure the build pod to always pull the images
	// for the builder or only pull if it is not present locally
	ForcePull bool `json:"forcePull,omitempty" protobuf:"varint,5,opt,name=forcePull"`

	// secrets is a list of additional secrets that will be included in the build pod
	Secrets []SecretSpec `json:"secrets,omitempty" protobuf:"bytes,6,rep,name=secrets"`

	// buildAPIVersion is the requested API version for the Build object serialized and passed to the custom builder
	BuildAPIVersion string `json:"buildAPIVersion,omitempty" protobuf:"bytes,7,opt,name=buildAPIVersion"`
}

// ImageOptimizationPolicy describes what optimizations the builder can perform when building images.
type ImageOptimizationPolicy string

const (
	// ImageOptimizationNone will generate a canonical container image as produced by the
	// `container image build` command.
	ImageOptimizationNone ImageOptimizationPolicy = "None"

	// ImageOptimizationSkipLayers is an experimental policy and will avoid creating
	// unique layers for each dockerfile line, resulting in smaller images and saving time
	// during creation. Some Dockerfile syntax is not fully supported - content added to
	// a VOLUME by an earlier layer may have incorrect uid, gid, and filesystem permissions.
	// If an unsupported setting is detected, the build will fail.
	ImageOptimizationSkipLayers ImageOptimizationPolicy = "SkipLayers"

	// ImageOptimizationSkipLayersAndWarn is the same as SkipLayers, but will only
	// warn to the build output instead of failing when unsupported syntax is detected. This
	// policy is experimental.
	ImageOptimizationSkipLayersAndWarn ImageOptimizationPolicy = "SkipLayersAndWarn"
)

// DockerBuildStrategy defines input parameters specific to container image build.
type DockerBuildStrategy struct {
	// from is a reference to an DockerImage, ImageStreamTag, or ImageStreamImage which overrides
	// the FROM image in the Dockerfile for the build. If the Dockerfile uses multi-stage builds,
	// this will replace the image in the last FROM directive of the file.
	From *corev1.ObjectReference `json:"from,omitempty" protobuf:"bytes,1,opt,name=from"`

	// pullSecret is the name of a Secret that would be used for setting up
	// the authentication for pulling the container images from the private Docker
	// registries
	PullSecret *corev1.LocalObjectReference `json:"pullSecret,omitempty" protobuf:"bytes,2,opt,name=pullSecret"`

	// noCache if set to true indicates that the container image build must be executed with the
	// --no-cache=true flag
	NoCache bool `json:"noCache,omitempty" protobuf:"varint,3,opt,name=noCache"`

	// env contains additional environment variables you want to pass into a builder container.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,4,rep,name=env"`

	// forcePull describes if the builder should pull the images from registry prior to building.
	ForcePull bool `json:"forcePull,omitempty" protobuf:"varint,5,opt,name=forcePull"`

	// dockerfilePath is the path of the Dockerfile that will be used to build the container image,
	// relative to the root of the context (contextDir).
	// Defaults to `Dockerfile` if unset.
	DockerfilePath string `json:"dockerfilePath,omitempty" protobuf:"bytes,6,opt,name=dockerfilePath"`

	// buildArgs contains build arguments that will be resolved in the Dockerfile.  See
	// https://docs.docker.com/engine/reference/builder/#/arg for more details.
	// NOTE: Only the 'name' and 'value' fields are supported. Any settings on the 'valueFrom' field
	// are ignored.
	BuildArgs []corev1.EnvVar `json:"buildArgs,omitempty" protobuf:"bytes,7,rep,name=buildArgs"`

	// imageOptimizationPolicy describes what optimizations the system can use when building images
	// to reduce the final size or time spent building the image. The default policy is 'None' which
	// means the final build image will be equivalent to an image created by the container image build API.
	// The experimental policy 'SkipLayers' will avoid commiting new layers in between each
	// image step, and will fail if the Dockerfile cannot provide compatibility with the 'None'
	// policy. An additional experimental policy 'SkipLayersAndWarn' is the same as
	// 'SkipLayers' but simply warns if compatibility cannot be preserved.
	ImageOptimizationPolicy *ImageOptimizationPolicy `json:"imageOptimizationPolicy,omitempty" protobuf:"bytes,8,opt,name=imageOptimizationPolicy,casttype=ImageOptimizationPolicy"`

	// volumes is a list of input volumes that can be mounted into the builds runtime environment.
	// Only a subset of Kubernetes Volume sources are supported by builds.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes
	// +listType=map
	// +listMapKey=name
	// +patchMergeKey=name
	// +patchStrategy=merge
	Volumes []BuildVolume `json:"volumes,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,9,opt,name=volumes"`
}

// SourceBuildStrategy defines input parameters specific to an Source build.
type SourceBuildStrategy struct {
	// from is reference to an DockerImage, ImageStreamTag, or ImageStreamImage from which
	// the container image should be pulled
	From corev1.ObjectReference `json:"from" protobuf:"bytes,1,opt,name=from"`

	// pullSecret is the name of a Secret that would be used for setting up
	// the authentication for pulling the container images from the private Docker
	// registries
	PullSecret *corev1.LocalObjectReference `json:"pullSecret,omitempty" protobuf:"bytes,2,opt,name=pullSecret"`

	// env contains additional environment variables you want to pass into a builder container.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,3,rep,name=env"`

	// scripts is the location of Source scripts
	Scripts string `json:"scripts,omitempty" protobuf:"bytes,4,opt,name=scripts"`

	// incremental flag forces the Source build to do incremental builds if true.
	Incremental *bool `json:"incremental,omitempty" protobuf:"varint,5,opt,name=incremental"`

	// forcePull describes if the builder should pull the images from registry prior to building.
	ForcePull bool `json:"forcePull,omitempty" protobuf:"varint,6,opt,name=forcePull"`

	// deprecated json field, do not reuse: runtimeImage
	// +k8s:protobuf-deprecated=runtimeImage,7

	// deprecated json field, do not reuse: runtimeArtifacts
	// +k8s:protobuf-deprecated=runtimeArtifacts,8

	// volumes is a list of input volumes that can be mounted into the builds runtime environment.
	// Only a subset of Kubernetes Volume sources are supported by builds.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes
	// +listType=map
	// +listMapKey=name
	// +patchMergeKey=name
	// +patchStrategy=merge
	Volumes []BuildVolume `json:"volumes,omitempty" patchStrategy:"merge" patchMergeKey:"name" protobuf:"bytes,9,opt,name=volumes"`
}

// JenkinsPipelineBuildStrategy holds parameters specific to a Jenkins Pipeline build.
// Deprecated: use OpenShift Pipelines
type JenkinsPipelineBuildStrategy struct {
	// JenkinsfilePath is the optional path of the Jenkinsfile that will be used to configure the pipeline
	// relative to the root of the context (contextDir). If both JenkinsfilePath & Jenkinsfile are
	// both not specified, this defaults to Jenkinsfile in the root of the specified contextDir.
	JenkinsfilePath string `json:"jenkinsfilePath,omitempty" protobuf:"bytes,1,opt,name=jenkinsfilePath"`

	// Jenkinsfile defines the optional raw contents of a Jenkinsfile which defines a Jenkins pipeline build.
	Jenkinsfile string `json:"jenkinsfile,omitempty" protobuf:"bytes,2,opt,name=jenkinsfile"`

	// env contains additional environment variables you want to pass into a build pipeline.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,3,rep,name=env"`
}

// A BuildPostCommitSpec holds a build post commit hook specification. The hook
// executes a command in a temporary container running the build output image,
// immediately after the last layer of the image is committed and before the
// image is pushed to a registry. The command is executed with the current
// working directory ($PWD) set to the image's WORKDIR.
//
// The build will be marked as failed if the hook execution fails. It will fail
// if the script or command return a non-zero exit code, or if there is any
// other error related to starting the temporary container.
//
// There are five different ways to configure the hook. As an example, all forms
// below are equivalent and will execute `rake test --verbose`.
//
// 1. Shell script:
//
//	   "postCommit": {
//	     "script": "rake test --verbose",
//	   }
//
//	The above is a convenient form which is equivalent to:
//
//	   "postCommit": {
//	     "command": ["/bin/sh", "-ic"],
//	     "args":    ["rake test --verbose"]
//	   }
//
// 2. A command as the image entrypoint:
//
//	   "postCommit": {
//	     "commit": ["rake", "test", "--verbose"]
//	   }
//
//	Command overrides the image entrypoint in the exec form, as documented in
//	Docker: https://docs.docker.com/engine/reference/builder/#entrypoint.
//
// 3. Pass arguments to the default entrypoint:
//
//	       "postCommit": {
//			      "args": ["rake", "test", "--verbose"]
//		      }
//
//	    This form is only useful if the image entrypoint can handle arguments.
//
// 4. Shell script with arguments:
//
//	   "postCommit": {
//	     "script": "rake test $1",
//	     "args":   ["--verbose"]
//	   }
//
//	This form is useful if you need to pass arguments that would otherwise be
//	hard to quote properly in the shell script. In the script, $0 will be
//	"/bin/sh" and $1, $2, etc, are the positional arguments from Args.
//
// 5. Command with arguments:
//
//	   "postCommit": {
//	     "command": ["rake", "test"],
//	     "args":    ["--verbose"]
//	   }
//
//	This form is equivalent to appending the arguments to the Command slice.
//
// It is invalid to provide both Script and Command simultaneously. If none of
// the fields are specified, the hook is not executed.
type BuildPostCommitSpec struct {
	// command is the command to run. It may not be specified with Script.
	// This might be needed if the image doesn't have `/bin/sh`, or if you
	// do not want to use a shell. In all other cases, using Script might be
	// more convenient.
	Command []string `json:"command,omitempty" protobuf:"bytes,1,rep,name=command"`
	// args is a list of arguments that are provided to either Command,
	// Script or the container image's default entrypoint. The arguments are
	// placed immediately after the command to be run.
	Args []string `json:"args,omitempty" protobuf:"bytes,2,rep,name=args"`
	// script is a shell script to be run with `/bin/sh -ic`. It may not be
	// specified with Command. Use Script when a shell script is appropriate
	// to execute the post build hook, for example for running unit tests
	// with `rake test`. If you need control over the image entrypoint, or
	// if the image does not have `/bin/sh`, use Command and/or Args.
	// The `-i` flag is needed to support CentOS and RHEL images that use
	// Software Collections (SCL), in order to have the appropriate
	// collections enabled in the shell. E.g., in the Ruby image, this is
	// necessary to make `ruby`, `bundle` and other binaries available in
	// the PATH.
	Script string `json:"script,omitempty" protobuf:"bytes,3,opt,name=script"`
}

// BuildOutput is input to a build strategy and describes the container image that the strategy
// should produce.
type BuildOutput struct {
	// to defines an optional location to push the output of this build to.
	// Kind must be one of 'ImageStreamTag' or 'DockerImage'.
	// This value will be used to look up a container image repository to push to.
	// In the case of an ImageStreamTag, the ImageStreamTag will be looked for in the namespace of
	// the build unless Namespace is specified.
	To *corev1.ObjectReference `json:"to,omitempty" protobuf:"bytes,1,opt,name=to"`

	// PushSecret is the name of a Secret that would be used for setting
	// up the authentication for executing the Docker push to authentication
	// enabled Docker Registry (or Docker Hub).
	PushSecret *corev1.LocalObjectReference `json:"pushSecret,omitempty" protobuf:"bytes,2,opt,name=pushSecret"`

	// imageLabels define a list of labels that are applied to the resulting image. If there
	// are multiple labels with the same name then the last one in the list is used.
	ImageLabels []ImageLabel `json:"imageLabels,omitempty" protobuf:"bytes,3,rep,name=imageLabels"`
}

// ImageLabel represents a label applied to the resulting image.
type ImageLabel struct {
	// name defines the name of the label. It must have non-zero length.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// value defines the literal value of the label.
	Value string `json:"value,omitempty" protobuf:"bytes,2,opt,name=value"`
}

// +genclient
// +genclient:method=Instantiate,verb=create,subresource=instantiate,input=BuildRequest,result=Build
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Build configurations define a build process for new container images. There are three types of builds possible - a container image build using a Dockerfile, a Source-to-Image build that uses a specially prepared base image that accepts source code that it can make runnable, and a custom build that can run // arbitrary container images as a base and accept the build parameters. Builds run on the cluster and on completion are pushed to the container image registry specified in the "output" section. A build can be triggered via a webhook, when the base image changes, or when a user manually requests a new build be // created.
//
// Each build created by a build configuration is numbered and refers back to its parent configuration. Multiple builds can be triggered at once. Builds that do not have "output" set can be used to test code or run a verification build.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildConfig struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec holds all the input necessary to produce a new build, and the conditions when
	// to trigger them.
	Spec BuildConfigSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
	// status holds any relevant information about a build config
	// +optional
	Status BuildConfigStatus `json:"status" protobuf:"bytes,3,opt,name=status"`
}

// BuildConfigSpec describes when and how builds are created
type BuildConfigSpec struct {

	//triggers determine how new Builds can be launched from a BuildConfig. If
	//no triggers are defined, a new build can only occur as a result of an
	//explicit client build creation.
	// +optional
	Triggers []BuildTriggerPolicy `json:"triggers,omitempty" protobuf:"bytes,1,rep,name=triggers"`

	// RunPolicy describes how the new build created from this build
	// configuration will be scheduled for execution.
	// This is optional, if not specified we default to "Serial".
	RunPolicy BuildRunPolicy `json:"runPolicy,omitempty" protobuf:"bytes,2,opt,name=runPolicy,casttype=BuildRunPolicy"`

	// CommonSpec is the desired build specification
	CommonSpec `json:",inline" protobuf:"bytes,3,opt,name=commonSpec"`

	// successfulBuildsHistoryLimit is the number of old successful builds to retain.
	// When a BuildConfig is created, the 5 most recent successful builds are retained unless this value is set.
	// If removed after the BuildConfig has been created, all successful builds are retained.
	SuccessfulBuildsHistoryLimit *int32 `json:"successfulBuildsHistoryLimit,omitempty" protobuf:"varint,4,opt,name=successfulBuildsHistoryLimit"`

	// failedBuildsHistoryLimit is the number of old failed builds to retain.
	// When a BuildConfig is created, the 5 most recent failed builds are retained unless this value is set.
	// If removed after the BuildConfig has been created, all failed builds are retained.
	FailedBuildsHistoryLimit *int32 `json:"failedBuildsHistoryLimit,omitempty" protobuf:"varint,5,opt,name=failedBuildsHistoryLimit"`
}

// BuildRunPolicy defines the behaviour of how the new builds are executed
// from the existing build configuration.
type BuildRunPolicy string

const (
	// BuildRunPolicyParallel schedules new builds immediately after they are
	// created. Builds will be executed in parallel.
	BuildRunPolicyParallel BuildRunPolicy = "Parallel"

	// BuildRunPolicySerial schedules new builds to execute in a sequence as
	// they are created. Every build gets queued up and will execute when the
	// previous build completes. This is the default policy.
	BuildRunPolicySerial BuildRunPolicy = "Serial"

	// BuildRunPolicySerialLatestOnly schedules only the latest build to execute,
	// cancelling all the previously queued build.
	BuildRunPolicySerialLatestOnly BuildRunPolicy = "SerialLatestOnly"
)

// BuildConfigStatus contains current state of the build config object.
type BuildConfigStatus struct {
	// lastVersion is used to inform about number of last triggered build.
	LastVersion int64 `json:"lastVersion" protobuf:"varint,1,opt,name=lastVersion"`

	// ImageChangeTriggers captures the runtime state of any ImageChangeTrigger specified in the BuildConfigSpec,
	// including the value reconciled by the OpenShift APIServer for the lastTriggeredImageID. There is a single entry
	// in this array for each image change trigger in spec. Each trigger status references the ImageStreamTag that acts as the source of the trigger.
	ImageChangeTriggers []ImageChangeTriggerStatus `json:"imageChangeTriggers,omitempty" protobuf:"bytes,2,rep,name=imageChangeTriggers"`
}

// SecretLocalReference contains information that points to the local secret being used
type SecretLocalReference struct {
	// Name is the name of the resource in the same namespace being referenced
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
}

// WebHookTrigger is a trigger that gets invoked using a webhook type of post
type WebHookTrigger struct {
	// secret used to validate requests.
	// Deprecated: use SecretReference instead.
	Secret string `json:"secret,omitempty" protobuf:"bytes,1,opt,name=secret"`

	// allowEnv determines whether the webhook can set environment variables; can only
	// be set to true for GenericWebHook.
	AllowEnv bool `json:"allowEnv,omitempty" protobuf:"varint,2,opt,name=allowEnv"`

	// secretReference is a reference to a secret in the same namespace,
	// containing the value to be validated when the webhook is invoked.
	// The secret being referenced must contain a key named "WebHookSecretKey", the value
	// of which will be checked against the value supplied in the webhook invocation.
	SecretReference *SecretLocalReference `json:"secretReference,omitempty" protobuf:"bytes,3,opt,name=secretReference"`
}

// ImageChangeTrigger allows builds to be triggered when an ImageStream changes
type ImageChangeTrigger struct {
	// lastTriggeredImageID is used internally by the ImageChangeController to save last
	// used image ID for build
	// This field is deprecated and will be removed in a future release.
	// Deprecated
	LastTriggeredImageID string `json:"lastTriggeredImageID,omitempty" protobuf:"bytes,1,opt,name=lastTriggeredImageID"`

	// from is a reference to an ImageStreamTag that will trigger a build when updated
	// It is optional. If no From is specified, the From image from the build strategy
	// will be used. Only one ImageChangeTrigger with an empty From reference is allowed in
	// a build configuration.
	From *corev1.ObjectReference `json:"from,omitempty" protobuf:"bytes,2,opt,name=from"`

	// paused is true if this trigger is temporarily disabled. Optional.
	Paused bool `json:"paused,omitempty" protobuf:"varint,3,opt,name=paused"`
}

// ImageStreamTagReference references the ImageStreamTag in an image change trigger by namespace and name.
type ImageStreamTagReference struct {
	// namespace is the namespace where the ImageStreamTag for an ImageChangeTrigger is located
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,1,opt,name=namespace"`

	// name is the name of the ImageStreamTag for an ImageChangeTrigger
	Name string `json:"name,omitempty" protobuf:"bytes,2,opt,name=name"`
}

// ImageChangeTriggerStatus tracks the latest resolved status of the associated ImageChangeTrigger policy
// specified in the BuildConfigSpec.Triggers struct.
type ImageChangeTriggerStatus struct {
	// lastTriggeredImageID represents the sha/id of the ImageStreamTag when a Build for this BuildConfig was started.
	// The lastTriggeredImageID is updated each time a Build for this BuildConfig is started, even if this ImageStreamTag is not the reason the Build is started.
	LastTriggeredImageID string `json:"lastTriggeredImageID,omitempty" protobuf:"bytes,1,opt,name=lastTriggeredImageID"`

	// from is the ImageStreamTag that is the source of the trigger.
	From ImageStreamTagReference `json:"from,omitempty" protobuf:"bytes,2,opt,name=from"`

	// lastTriggerTime is the last time this particular ImageStreamTag triggered a Build to start.
	// This field is only updated when this trigger specifically started a Build.
	LastTriggerTime metav1.Time `json:"lastTriggerTime,omitempty" protobuf:"bytes,3,opt,name=lastTriggerTime"`
}

// BuildTriggerPolicy describes a policy for a single trigger that results in a new Build.
type BuildTriggerPolicy struct {
	// type is the type of build trigger. Valid values:
	//
	// - GitHub
	// GitHubWebHookBuildTriggerType represents a trigger that launches builds on
	// GitHub webhook invocations
	//
	// - Generic
	// GenericWebHookBuildTriggerType represents a trigger that launches builds on
	// generic webhook invocations
	//
	// - GitLab
	// GitLabWebHookBuildTriggerType represents a trigger that launches builds on
	// GitLab webhook invocations
	//
	// - Bitbucket
	// BitbucketWebHookBuildTriggerType represents a trigger that launches builds on
	// Bitbucket webhook invocations
	//
	// - ImageChange
	// ImageChangeBuildTriggerType represents a trigger that launches builds on
	// availability of a new version of an image
	//
	// - ConfigChange
	// ConfigChangeBuildTriggerType will trigger a build on an initial build config creation
	// WARNING: In the future the behavior will change to trigger a build on any config change
	Type BuildTriggerType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=BuildTriggerType"`

	// github contains the parameters for a GitHub webhook type of trigger
	GitHubWebHook *WebHookTrigger `json:"github,omitempty" protobuf:"bytes,2,opt,name=github"`

	// generic contains the parameters for a Generic webhook type of trigger
	GenericWebHook *WebHookTrigger `json:"generic,omitempty" protobuf:"bytes,3,opt,name=generic"`

	// imageChange contains parameters for an ImageChange type of trigger
	ImageChange *ImageChangeTrigger `json:"imageChange,omitempty" protobuf:"bytes,4,opt,name=imageChange"`

	// GitLabWebHook contains the parameters for a GitLab webhook type of trigger
	GitLabWebHook *WebHookTrigger `json:"gitlab,omitempty" protobuf:"bytes,5,opt,name=gitlab"`

	// BitbucketWebHook contains the parameters for a Bitbucket webhook type of
	// trigger
	BitbucketWebHook *WebHookTrigger `json:"bitbucket,omitempty" protobuf:"bytes,6,opt,name=bitbucket"`
}

// BuildTriggerType refers to a specific BuildTriggerPolicy implementation.
type BuildTriggerType string

const (
	// GitHubWebHookBuildTriggerType represents a trigger that launches builds on
	// GitHub webhook invocations
	GitHubWebHookBuildTriggerType           BuildTriggerType = "GitHub"
	GitHubWebHookBuildTriggerTypeDeprecated BuildTriggerType = "github"

	// GenericWebHookBuildTriggerType represents a trigger that launches builds on
	// generic webhook invocations
	GenericWebHookBuildTriggerType           BuildTriggerType = "Generic"
	GenericWebHookBuildTriggerTypeDeprecated BuildTriggerType = "generic"

	// GitLabWebHookBuildTriggerType represents a trigger that launches builds on
	// GitLab webhook invocations
	GitLabWebHookBuildTriggerType BuildTriggerType = "GitLab"

	// BitbucketWebHookBuildTriggerType represents a trigger that launches builds on
	// Bitbucket webhook invocations
	BitbucketWebHookBuildTriggerType BuildTriggerType = "Bitbucket"

	// ImageChangeBuildTriggerType represents a trigger that launches builds on
	// availability of a new version of an image
	ImageChangeBuildTriggerType           BuildTriggerType = "ImageChange"
	ImageChangeBuildTriggerTypeDeprecated BuildTriggerType = "imageChange"

	// ConfigChangeBuildTriggerType will trigger a build on an initial build config creation
	// WARNING: In the future the behavior will change to trigger a build on any config change
	ConfigChangeBuildTriggerType BuildTriggerType = "ConfigChange"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BuildList is a collection of Builds.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of builds
	Items []Build `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BuildConfigList is a collection of BuildConfigs.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildConfigList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of build configs
	Items []BuildConfig `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// GenericWebHookEvent is the payload expected for a generic webhook post
type GenericWebHookEvent struct {
	// type is the type of source repository
	// +k8s:conversion-gen=false
	Type BuildSourceType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type,casttype=BuildSourceType"`

	// git is the git information if the Type is BuildSourceGit
	Git *GitInfo `json:"git,omitempty" protobuf:"bytes,2,opt,name=git"`

	// env contains additional environment variables you want to pass into a builder container.
	// ValueFrom is not supported.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,3,rep,name=env"`

	// DockerStrategyOptions contains additional docker-strategy specific options for the build
	DockerStrategyOptions *DockerStrategyOptions `json:"dockerStrategyOptions,omitempty" protobuf:"bytes,4,opt,name=dockerStrategyOptions"`
}

// GitInfo is the aggregated git information for a generic webhook post
type GitInfo struct {
	GitBuildSource    `json:",inline" protobuf:"bytes,1,opt,name=gitBuildSource"`
	GitSourceRevision `json:",inline" protobuf:"bytes,2,opt,name=gitSourceRevision"`

	// Refs is a list of GitRefs for the provided repo - generally sent
	// when used from a post-receive hook. This field is optional and is
	// used when sending multiple refs
	Refs []GitRefInfo `json:"refs" protobuf:"bytes,3,rep,name=refs"`
}

// GitRefInfo is a single ref
type GitRefInfo struct {
	GitBuildSource    `json:",inline" protobuf:"bytes,1,opt,name=gitBuildSource"`
	GitSourceRevision `json:",inline" protobuf:"bytes,2,opt,name=gitSourceRevision"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BuildLog is the (unused) resource associated with the build log redirector
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildLog struct {
	metav1.TypeMeta `json:",inline"`
}

// DockerStrategyOptions contains extra strategy options for container image builds
type DockerStrategyOptions struct {
	// Args contains any build arguments that are to be passed to Docker.  See
	// https://docs.docker.com/engine/reference/builder/#/arg for more details
	BuildArgs []corev1.EnvVar `json:"buildArgs,omitempty" protobuf:"bytes,1,rep,name=buildArgs"`

	// noCache overrides the docker-strategy noCache option in the build config
	NoCache *bool `json:"noCache,omitempty" protobuf:"varint,2,opt,name=noCache"`
}

// SourceStrategyOptions contains extra strategy options for Source builds
type SourceStrategyOptions struct {
	// incremental overrides the source-strategy incremental option in the build config
	Incremental *bool `json:"incremental,omitempty" protobuf:"varint,1,opt,name=incremental"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BuildRequest is the resource used to pass parameters to build generator
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildRequest struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// revision is the information from the source for a specific repo snapshot.
	Revision *SourceRevision `json:"revision,omitempty" protobuf:"bytes,2,opt,name=revision"`

	// triggeredByImage is the Image that triggered this build.
	TriggeredByImage *corev1.ObjectReference `json:"triggeredByImage,omitempty" protobuf:"bytes,3,opt,name=triggeredByImage"`

	// from is the reference to the ImageStreamTag that triggered the build.
	From *corev1.ObjectReference `json:"from,omitempty" protobuf:"bytes,4,opt,name=from"`

	// binary indicates a request to build from a binary provided to the builder
	Binary *BinaryBuildSource `json:"binary,omitempty" protobuf:"bytes,5,opt,name=binary"`

	// lastVersion (optional) is the LastVersion of the BuildConfig that was used
	// to generate the build. If the BuildConfig in the generator doesn't match, a build will
	// not be generated.
	LastVersion *int64 `json:"lastVersion,omitempty" protobuf:"varint,6,opt,name=lastVersion"`

	// env contains additional environment variables you want to pass into a builder container.
	Env []corev1.EnvVar `json:"env,omitempty" protobuf:"bytes,7,rep,name=env"`

	// triggeredBy describes which triggers started the most recent update to the
	// build configuration and contains information about those triggers.
	TriggeredBy []BuildTriggerCause `json:"triggeredBy,omitempty" protobuf:"bytes,8,rep,name=triggeredBy"`

	// DockerStrategyOptions contains additional docker-strategy specific options for the build
	DockerStrategyOptions *DockerStrategyOptions `json:"dockerStrategyOptions,omitempty" protobuf:"bytes,9,opt,name=dockerStrategyOptions"`

	// SourceStrategyOptions contains additional source-strategy specific options for the build
	SourceStrategyOptions *SourceStrategyOptions `json:"sourceStrategyOptions,omitempty" protobuf:"bytes,10,opt,name=sourceStrategyOptions"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BinaryBuildRequestOptions are the options required to fully speficy a binary build request
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BinaryBuildRequestOptions struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// asFile determines if the binary should be created as a file within the source rather than extracted as an archive
	AsFile string `json:"asFile,omitempty" protobuf:"bytes,2,opt,name=asFile"`

	// TODO: Improve map[string][]string conversion so we can handled nested objects

	// revision.commit is the value identifying a specific commit
	Commit string `json:"revision.commit,omitempty" protobuf:"bytes,3,opt,name=revisionCommit"`

	// revision.message is the description of a specific commit
	Message string `json:"revision.message,omitempty" protobuf:"bytes,4,opt,name=revisionMessage"`

	// revision.authorName of the source control user
	AuthorName string `json:"revision.authorName,omitempty" protobuf:"bytes,5,opt,name=revisionAuthorName"`

	// revision.authorEmail of the source control user
	AuthorEmail string `json:"revision.authorEmail,omitempty" protobuf:"bytes,6,opt,name=revisionAuthorEmail"`

	// revision.committerName of the source control user
	CommitterName string `json:"revision.committerName,omitempty" protobuf:"bytes,7,opt,name=revisionCommitterName"`

	// revision.committerEmail of the source control user
	CommitterEmail string `json:"revision.committerEmail,omitempty" protobuf:"bytes,8,opt,name=revisionCommitterEmail"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BuildLogOptions is the REST options for a build log
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildLogOptions struct {
	metav1.TypeMeta `json:",inline"`

	// cointainer for which to stream logs. Defaults to only container if there is one container in the pod.
	Container string `json:"container,omitempty" protobuf:"bytes,1,opt,name=container"`
	// follow if true indicates that the build log should be streamed until
	// the build terminates.
	Follow bool `json:"follow,omitempty" protobuf:"varint,2,opt,name=follow"`
	// previous returns previous build logs. Defaults to false.
	Previous bool `json:"previous,omitempty" protobuf:"varint,3,opt,name=previous"`
	// sinceSeconds is a relative time in seconds before the current time from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceSeconds *int64 `json:"sinceSeconds,omitempty" protobuf:"varint,4,opt,name=sinceSeconds"`
	// sinceTime is an RFC3339 timestamp from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceTime *metav1.Time `json:"sinceTime,omitempty" protobuf:"bytes,5,opt,name=sinceTime"`
	// timestamps, If true, add an RFC3339 or RFC3339Nano timestamp at the beginning of every line
	// of log output. Defaults to false.
	Timestamps bool `json:"timestamps,omitempty" protobuf:"varint,6,opt,name=timestamps"`
	// tailLines, If set, is the number of lines from the end of the logs to show. If not specified,
	// logs are shown from the creation of the container or sinceSeconds or sinceTime
	TailLines *int64 `json:"tailLines,omitempty" protobuf:"varint,7,opt,name=tailLines"`
	// limitBytes, If set, is the number of bytes to read from the server before terminating the
	// log output. This may not display a complete final line of logging, and may return
	// slightly more or slightly less than the specified limit.
	LimitBytes *int64 `json:"limitBytes,omitempty" protobuf:"varint,8,opt,name=limitBytes"`

	// noWait if true causes the call to return immediately even if the build
	// is not available yet. Otherwise the server will wait until the build has started.
	// TODO: Fix the tag to 'noWait' in v2
	NoWait bool `json:"nowait,omitempty" protobuf:"varint,9,opt,name=nowait"`

	// version of the build for which to view logs.
	Version *int64 `json:"version,omitempty" protobuf:"varint,10,opt,name=version"`

	// insecureSkipTLSVerifyBackend indicates that the apiserver should not confirm the validity of the
	// serving certificate of the backend it is connecting to.  This will make the HTTPS connection between the apiserver
	// and the backend insecure. This means the apiserver cannot verify the log data it is receiving came from the real
	// kubelet.  If the kubelet is configured to verify the apiserver's TLS credentials, it does not mean the
	// connection to the real kubelet is vulnerable to a man in the middle attack (e.g. an attacker could not intercept
	// the actual log data coming from the real kubelet).
	// +optional
	InsecureSkipTLSVerifyBackend bool `json:"insecureSkipTLSVerifyBackend,omitempty" protobuf:"varint,11,opt,name=insecureSkipTLSVerifyBackend"`
}

// SecretSpec specifies a secret to be included in a build pod and its corresponding mount point
type SecretSpec struct {
	// secretSource is a reference to the secret
	SecretSource corev1.LocalObjectReference `json:"secretSource" protobuf:"bytes,1,opt,name=secretSource"`

	// mountPath is the path at which to mount the secret
	MountPath string `json:"mountPath" protobuf:"bytes,2,opt,name=mountPath"`
}

// BuildVolume describes a volume that is made available to build pods,
// such that it can be mounted into buildah's runtime environment.
// Only a subset of Kubernetes Volume sources are supported.
type BuildVolume struct {
	// name is a unique identifier for this BuildVolume.
	// It must conform to the Kubernetes DNS label standard and be unique within the pod.
	// Names that collide with those added by the build controller will result in a
	// failed build with an error message detailing which name caused the error.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names
	// +required
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// source represents the location and type of the mounted volume.
	// +required
	Source BuildVolumeSource `json:"source" protobuf:"bytes,2,opt,name=source"`

	// mounts represents the location of the volume in the image build container
	// +required
	// +listType=map
	// +listMapKey=destinationPath
	// +patchMergeKey=destinationPath
	// +patchStrategy=merge
	Mounts []BuildVolumeMount `json:"mounts" patchStrategy:"merge" patchMergeKey:"destinationPath" protobuf:"bytes,3,opt,name=mounts"`
}

// BuildVolumeSourceType represents a build volume source type
type BuildVolumeSourceType string

const (
	// BuildVolumeSourceTypeSecret is the Secret build source volume type
	BuildVolumeSourceTypeSecret BuildVolumeSourceType = "Secret"

	// BuildVolumeSourceTypeConfigmap is the ConfigMap build source volume type
	BuildVolumeSourceTypeConfigMap BuildVolumeSourceType = "ConfigMap"

	// BuildVolumeSourceTypeCSI is the CSI build source volume type
	BuildVolumeSourceTypeCSI BuildVolumeSourceType = "CSI"
)

// BuildVolumeSource represents the source of a volume to mount
// Only one of its supported types may be specified at any given time.
type BuildVolumeSource struct {

	// type is the BuildVolumeSourceType for the volume source.
	// Type must match the populated volume source.
	// Valid types are: Secret, ConfigMap
	Type BuildVolumeSourceType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=BuildVolumeSourceType"`

	// secret represents a Secret that should populate this volume.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes#secret
	// +optional
	Secret *corev1.SecretVolumeSource `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`

	// configMap represents a ConfigMap that should populate this volume
	// +optional
	ConfigMap *corev1.ConfigMapVolumeSource `json:"configMap,omitempty" protobuf:"bytes,3,opt,name=configMap"`

	// csi represents ephemeral storage provided by external CSI drivers which support this capability
	// +optional
	CSI *corev1.CSIVolumeSource `json:"csi,omitempty" protobuf:"bytes,4,opt,name=csi"`
}

// BuildVolumeMount describes the mounting of a Volume within buildah's runtime environment.
type BuildVolumeMount struct {
	// destinationPath is the path within the buildah runtime environment at which the volume should be mounted.
	// The transient mount within the build image and the backing volume will both be mounted read only.
	// Must be an absolute path, must not contain '..' or ':', and must not collide with a destination path generated
	// by the builder process
	// Paths that collide with those added by the build controller will result in a
	// failed build with an error message detailing which path caused the error.
	DestinationPath string `json:"destinationPath" protobuf:"bytes,1,opt,name=destinationPath"`
}
