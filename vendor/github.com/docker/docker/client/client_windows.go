package client // import "github.com/docker/docker/client"

// DefaultDockerHost defines os specific default if DOCKER_HOST is unset
const DefaultDockerHost = "npipe:////./pipe/docker_engine"

const defaultProto = "npipe"
const defaultAddr = "//./pipe/docker_engine"
