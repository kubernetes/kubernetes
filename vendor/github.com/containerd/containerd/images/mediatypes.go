package images

// mediatype definitions for image components handled in containerd.
//
// oci components are generally referenced directly, although we may centralize
// here for clarity.
const (
	MediaTypeDockerSchema2Layer            = "application/vnd.docker.image.rootfs.diff.tar"
	MediaTypeDockerSchema2LayerForeign     = "application/vnd.docker.image.rootfs.foreign.diff.tar"
	MediaTypeDockerSchema2LayerGzip        = "application/vnd.docker.image.rootfs.diff.tar.gzip"
	MediaTypeDockerSchema2LayerForeignGzip = "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip"
	MediaTypeDockerSchema2Config           = "application/vnd.docker.container.image.v1+json"
	MediaTypeDockerSchema2Manifest         = "application/vnd.docker.distribution.manifest.v2+json"
	MediaTypeDockerSchema2ManifestList     = "application/vnd.docker.distribution.manifest.list.v2+json"
	// Checkpoint/Restore Media Types
	MediaTypeContainerd1Checkpoint        = "application/vnd.containerd.container.criu.checkpoint.criu.tar"
	MediaTypeContainerd1CheckpointPreDump = "application/vnd.containerd.container.criu.checkpoint.predump.tar"
	MediaTypeContainerd1Resource          = "application/vnd.containerd.container.resource.tar"
	MediaTypeContainerd1RW                = "application/vnd.containerd.container.rw.tar"
	MediaTypeContainerd1CheckpointConfig  = "application/vnd.containerd.container.checkpoint.config.v1+proto"
	// Legacy Docker schema1 manifest
	MediaTypeDockerSchema1Manifest = "application/vnd.docker.distribution.manifest.v1+prettyjws"
)
