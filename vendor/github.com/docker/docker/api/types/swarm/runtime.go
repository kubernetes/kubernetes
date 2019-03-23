package swarm

// RuntimeType is the type of runtime used for the TaskSpec
type RuntimeType string

// RuntimeURL is the proto type url
type RuntimeURL string

const (
	// RuntimeContainer is the container based runtime
	RuntimeContainer RuntimeType = "container"
	// RuntimePlugin is the plugin based runtime
	RuntimePlugin RuntimeType = "plugin"
	// RuntimeNetworkAttachment is the network attachment runtime
	RuntimeNetworkAttachment RuntimeType = "attachment"

	// RuntimeURLContainer is the proto url for the container type
	RuntimeURLContainer RuntimeURL = "types.docker.com/RuntimeContainer"
	// RuntimeURLPlugin is the proto url for the plugin type
	RuntimeURLPlugin RuntimeURL = "types.docker.com/RuntimePlugin"
)

// NetworkAttachmentSpec represents the runtime spec type for network
// attachment tasks
type NetworkAttachmentSpec struct {
	ContainerID string
}
