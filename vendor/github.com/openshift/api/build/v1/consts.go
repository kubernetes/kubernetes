package v1

// annotations
const (
	// BuildAnnotation is an annotation that identifies a Pod as being for a Build
	BuildAnnotation = "openshift.io/build.name"

	// BuildConfigAnnotation is an annotation that identifies the BuildConfig that a Build was created from
	BuildConfigAnnotation = "openshift.io/build-config.name"

	// BuildCloneAnnotation is an annotation whose value is the name of the build this build was cloned from
	BuildCloneAnnotation = "openshift.io/build.clone-of"

	// BuildNumberAnnotation is an annotation whose value is the sequential number for this Build
	BuildNumberAnnotation = "openshift.io/build.number"

	// BuildPodNameAnnotation is an annotation whose value is the name of the pod running this build
	BuildPodNameAnnotation = "openshift.io/build.pod-name"

	// BuildJenkinsStatusJSONAnnotation is an annotation holding the Jenkins status information
	BuildJenkinsStatusJSONAnnotation = "openshift.io/jenkins-status-json"

	// BuildJenkinsLogURLAnnotation is an annotation holding a link to the raw Jenkins build console log
	BuildJenkinsLogURLAnnotation = "openshift.io/jenkins-log-url"

	// BuildJenkinsConsoleLogURLAnnotation is an annotation holding a link to the Jenkins build console log (including Jenkins chrome wrappering)
	BuildJenkinsConsoleLogURLAnnotation = "openshift.io/jenkins-console-log-url"

	// BuildJenkinsBlueOceanLogURLAnnotation is an annotation holding a link to the Jenkins build console log via the Jenkins BlueOcean UI Plugin
	BuildJenkinsBlueOceanLogURLAnnotation = "openshift.io/jenkins-blueocean-log-url"

	// BuildJenkinsBuildURIAnnotation is an annotation holding a link to the Jenkins build
	BuildJenkinsBuildURIAnnotation = "openshift.io/jenkins-build-uri"

	// BuildSourceSecretMatchURIAnnotationPrefix is a prefix for annotations on a Secret which indicate a source URI against which the Secret can be used
	BuildSourceSecretMatchURIAnnotationPrefix = "build.openshift.io/source-secret-match-uri-"

	// BuildConfigPausedAnnotation is an annotation that marks a BuildConfig as paused.
	// New Builds cannot be instantiated from a paused BuildConfig.
	BuildConfigPausedAnnotation = "openshift.io/build-config.paused"
)

// labels
const (
	// BuildConfigLabel is the key of a Build label whose value is the ID of a BuildConfig
	// on which the Build is based. NOTE: The value for this label may not contain the entire
	// BuildConfig name because it will be truncated to maximum label length.
	BuildConfigLabel = "openshift.io/build-config.name"

	// BuildLabel is the key of a Pod label whose value is the Name of a Build which is run.
	// NOTE: The value for this label may not contain the entire Build name because it will be
	// truncated to maximum label length.
	BuildLabel = "openshift.io/build.name"

	// BuildRunPolicyLabel represents the start policy used to start the build.
	BuildRunPolicyLabel = "openshift.io/build.start-policy"

	// BuildConfigLabelDeprecated was used as BuildConfigLabel before adding namespaces.
	// We keep it for backward compatibility.
	BuildConfigLabelDeprecated = "buildconfig"
)

const (
	// StatusReasonError is a generic reason for a build error condition.
	StatusReasonError StatusReason = "Error"

	// StatusReasonCannotCreateBuildPodSpec is an error condition when the build
	// strategy cannot create a build pod spec.
	StatusReasonCannotCreateBuildPodSpec StatusReason = "CannotCreateBuildPodSpec"

	// StatusReasonCannotCreateBuildPod is an error condition when a build pod
	// cannot be created.
	StatusReasonCannotCreateBuildPod StatusReason = "CannotCreateBuildPod"

	// StatusReasonInvalidOutputReference is an error condition when the build
	// output is an invalid reference.
	StatusReasonInvalidOutputReference StatusReason = "InvalidOutputReference"

	// StatusReasonInvalidImageReference is an error condition when the build
	// references an invalid image.
	StatusReasonInvalidImageReference StatusReason = "InvalidImageReference"

	// StatusReasonCancelBuildFailed is an error condition when cancelling a build
	// fails.
	StatusReasonCancelBuildFailed StatusReason = "CancelBuildFailed"

	// StatusReasonBuildPodDeleted is an error condition when the build pod is
	// deleted before build completion.
	StatusReasonBuildPodDeleted StatusReason = "BuildPodDeleted"

	// StatusReasonExceededRetryTimeout is an error condition when the build has
	// not completed and retrying the build times out.
	StatusReasonExceededRetryTimeout StatusReason = "ExceededRetryTimeout"

	// StatusReasonMissingPushSecret indicates that the build is missing required
	// secret for pushing the output image.
	// The build will stay in the pending state until the secret is created, or the build times out.
	StatusReasonMissingPushSecret StatusReason = "MissingPushSecret"

	// StatusReasonPostCommitHookFailed indicates the post-commit hook failed.
	StatusReasonPostCommitHookFailed StatusReason = "PostCommitHookFailed"

	// StatusReasonPushImageToRegistryFailed indicates that an image failed to be
	// pushed to the registry.
	StatusReasonPushImageToRegistryFailed StatusReason = "PushImageToRegistryFailed"

	// StatusReasonPullBuilderImageFailed indicates that we failed to pull the
	// builder image.
	StatusReasonPullBuilderImageFailed StatusReason = "PullBuilderImageFailed"

	// StatusReasonFetchSourceFailed indicates that fetching the source of the
	// build has failed.
	StatusReasonFetchSourceFailed StatusReason = "FetchSourceFailed"

	// StatusReasonFetchImageContentFailed indicates that the fetching of an image and extracting
	// its contents for inclusion in the build has failed.
	StatusReasonFetchImageContentFailed StatusReason = "FetchImageContentFailed"

	// StatusReasonManageDockerfileFailed indicates that the set up of the Dockerfile for the build
	// has failed.
	StatusReasonManageDockerfileFailed StatusReason = "ManageDockerfileFailed"

	// StatusReasonInvalidContextDirectory indicates that the supplied
	// contextDir does not exist
	StatusReasonInvalidContextDirectory StatusReason = "InvalidContextDirectory"

	// StatusReasonCancelledBuild indicates that the build was cancelled by the
	// user.
	StatusReasonCancelledBuild StatusReason = "CancelledBuild"

	// StatusReasonDockerBuildFailed indicates that the container image build strategy has
	// failed.
	StatusReasonDockerBuildFailed StatusReason = "DockerBuildFailed"

	// StatusReasonBuildPodExists indicates that the build tried to create a
	// build pod but one was already present.
	StatusReasonBuildPodExists StatusReason = "BuildPodExists"

	// StatusReasonNoBuildContainerStatus indicates that the build failed because the
	// the build pod has no container statuses.
	StatusReasonNoBuildContainerStatus StatusReason = "NoBuildContainerStatus"

	// StatusReasonFailedContainer indicates that the pod for the build has at least
	// one container with a non-zero exit status.
	StatusReasonFailedContainer StatusReason = "FailedContainer"

	// StatusReasonUnresolvableEnvironmentVariable indicates that an error occurred processing
	// the supplied options for environment variables in the build strategy environment
	StatusReasonUnresolvableEnvironmentVariable StatusReason = "UnresolvableEnvironmentVariable"

	// StatusReasonGenericBuildFailed is the reason associated with a broad
	// range of build failures.
	StatusReasonGenericBuildFailed StatusReason = "GenericBuildFailed"

	// StatusReasonOutOfMemoryKilled indicates that the build pod was killed for its memory consumption
	StatusReasonOutOfMemoryKilled StatusReason = "OutOfMemoryKilled"

	// StatusReasonCannotRetrieveServiceAccount is the reason associated with a failure
	// to look up the service account associated with the BuildConfig.
	StatusReasonCannotRetrieveServiceAccount StatusReason = "CannotRetrieveServiceAccount"

	// StatusReasonBuildPodEvicted is the reason a build fails due to the build pod being evicted
	// from its node
	StatusReasonBuildPodEvicted StatusReason = "BuildPodEvicted"
)

// env vars
// WhitelistEnvVarNames is a list of special env vars allows s2i containers
var WhitelistEnvVarNames = []string{"BUILD_LOGLEVEL", "GIT_SSL_NO_VERIFY", "HTTP_PROXY", "HTTPS_PROXY", "LANG", "NO_PROXY"}

// env vars
const (

	// CustomBuildStrategyBaseImageKey is the environment variable that indicates the base image to be used when
	// performing a custom build, if needed.
	CustomBuildStrategyBaseImageKey = "OPENSHIFT_CUSTOM_BUILD_BASE_IMAGE"

	// AllowedUIDs is an environment variable that contains ranges of UIDs that are allowed in
	// Source builder images
	AllowedUIDs = "ALLOWED_UIDS"
	// DropCapabilities is an environment variable that contains a list of capabilities to drop when
	// executing a Source build
	DropCapabilities = "DROP_CAPS"
)

// keys inside of secrets and configmaps
const (
	// WebHookSecretKey is the key used to identify the value containing the webhook invocation
	// secret within a secret referenced by a webhook trigger.
	WebHookSecretKey = "WebHookSecretKey"

	// RegistryConfKey is the ConfigMap key for the build pod's registry configuration file.
	RegistryConfKey = "registries.conf"

	// SignaturePolicyKey is the ConfigMap key for the build pod's image signature policy file.
	SignaturePolicyKey = "policy.json"

	// ServiceCAKey is the ConfigMap key for the service signing certificate authority mounted into build pods.
	ServiceCAKey = "service-ca.crt"
)
