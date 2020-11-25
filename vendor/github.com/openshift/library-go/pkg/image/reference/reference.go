package reference

import (
	"net"
	"net/url"
	"strings"

	"github.com/openshift/library-go/pkg/image/internal/digest"
	"github.com/openshift/library-go/pkg/image/internal/reference"
)

// DockerImageReference points to a Docker image.
type DockerImageReference struct {
	Registry  string
	Namespace string
	Name      string
	Tag       string
	ID        string
}

const (
	// DockerDefaultRegistry is the value for the registry when none was provided.
	DockerDefaultRegistry = "docker.io"
	// DockerDefaultV1Registry is the host name of the default v1 registry
	DockerDefaultV1Registry = "index." + DockerDefaultRegistry
	// DockerDefaultV2Registry is the host name of the default v2 registry
	DockerDefaultV2Registry = "registry-1." + DockerDefaultRegistry
)

// Parse parses a Docker pull spec string into a
// DockerImageReference.
func Parse(spec string) (DockerImageReference, error) {
	var ref DockerImageReference

	namedRef, err := reference.ParseNamed(spec)
	if err != nil {
		return ref, err
	}

	name := namedRef.Name()
	i := strings.IndexRune(name, '/')

	// if there are no path components, and it looks like a url (contains a .) or localhost, it's a registry
	isRegistryOnly := i == -1 && (strings.ContainsAny(name, ".") || strings.HasPrefix(name, "localhost"))

	// if there are no path components, and it's not a registry, it's a name
	isNameOnly := i == -1 && !isRegistryOnly

	// if there are path components, and the first component doesn't look like a url, it's a name
	isNameOnly = isNameOnly || (i > -1 && (!strings.ContainsAny(name[:i], ":.") && name[:i] != "localhost"))

	if isRegistryOnly {
		ref.Registry = namedRef.String()
	} else if isNameOnly {
		ref.Name = name
	} else {
		ref.Registry, ref.Name = name[:i], name[i+1:]
	}

	if named, ok := namedRef.(reference.NamedTagged); !isRegistryOnly && ok {
		ref.Tag = named.Tag()
	}

	if named, ok := namedRef.(reference.Canonical); ok {
		ref.ID = named.Digest().String()
	}

	// It's not enough just to use the reference.ParseNamed(). We have to fill
	// ref.Namespace from ref.Name
	if i := strings.IndexRune(ref.Name, '/'); i != -1 {
		ref.Namespace, ref.Name = ref.Name[:i], ref.Name[i+1:]
	}

	return ref, nil
}

// Equal returns true if the other DockerImageReference is equivalent to the
// reference r. The comparison applies defaults to the Docker image reference,
// so that e.g., "foobar" equals "docker.io/library/foobar:latest".
func (r DockerImageReference) Equal(other DockerImageReference) bool {
	defaultedRef := r.DockerClientDefaults()
	otherDefaultedRef := other.DockerClientDefaults()
	return defaultedRef == otherDefaultedRef
}

// DockerClientDefaults sets the default values used by the Docker client.
func (r DockerImageReference) DockerClientDefaults() DockerImageReference {
	if len(r.Registry) == 0 {
		r.Registry = DockerDefaultRegistry
	}
	if len(r.Namespace) == 0 && IsRegistryDockerHub(r.Registry) {
		r.Namespace = "library"
	}
	if len(r.Tag) == 0 {
		r.Tag = "latest"
	}
	return r
}

// Minimal reduces a DockerImageReference to its minimalist form.
func (r DockerImageReference) Minimal() DockerImageReference {
	if r.Tag == "latest" {
		r.Tag = ""
	}
	return r
}

// AsRepository returns the reference without tags or IDs.
func (r DockerImageReference) AsRepository() DockerImageReference {
	r.Tag = ""
	r.ID = ""
	return r
}

// RepositoryName returns the registry relative name
func (r DockerImageReference) RepositoryName() string {
	r.Tag = ""
	r.ID = ""
	r.Registry = ""
	return r.Exact()
}

// RegistryHostPort returns the registry hostname and the port.
// If the port is not specified in the registry hostname we default to 443.
// This will also default to Docker client defaults if the registry hostname is empty.
func (r DockerImageReference) RegistryHostPort(insecure bool) (string, string) {
	registryHost := r.AsV2().DockerClientDefaults().Registry
	if strings.Contains(registryHost, ":") {
		hostname, port, _ := net.SplitHostPort(registryHost)
		return hostname, port
	}
	if insecure {
		return registryHost, "80"
	}
	return registryHost, "443"
}

// RepositoryName returns the registry relative name
func (r DockerImageReference) RegistryURL() *url.URL {
	return &url.URL{
		Scheme: "https",
		Host:   r.AsV2().Registry,
	}
}

// DaemonMinimal clears defaults that Docker assumes.
func (r DockerImageReference) DaemonMinimal() DockerImageReference {
	switch r.Registry {
	case DockerDefaultV1Registry, DockerDefaultV2Registry:
		r.Registry = DockerDefaultRegistry
	}
	if IsRegistryDockerHub(r.Registry) && r.Namespace == "library" {
		r.Namespace = ""
	}
	return r.Minimal()
}

func (r DockerImageReference) AsV2() DockerImageReference {
	switch r.Registry {
	case DockerDefaultV1Registry, DockerDefaultRegistry:
		r.Registry = DockerDefaultV2Registry
	}
	return r
}

// MostSpecific returns the most specific image reference that can be constructed from the
// current ref, preferring an ID over a Tag. Allows client code dealing with both tags and IDs
// to get the most specific reference easily.
func (r DockerImageReference) MostSpecific() DockerImageReference {
	if len(r.ID) == 0 {
		return r
	}
	if _, err := digest.ParseDigest(r.ID); err == nil {
		r.Tag = ""
		return r
	}
	if len(r.Tag) == 0 {
		r.Tag, r.ID = r.ID, ""
		return r
	}
	return r
}

// NameString returns the name of the reference with its tag or ID.
func (r DockerImageReference) NameString() string {
	switch {
	case len(r.Name) == 0:
		return ""
	case len(r.ID) > 0:
		var ref string
		if _, err := digest.ParseDigest(r.ID); err == nil {
			// if it parses as a digest, its v2 pull by id
			ref = "@" + r.ID
		} else {
			// if it doesn't parse as a digest, it's presumably a v1 registry by-id tag
			ref = ":" + r.ID
		}
		return r.Name + ref
	case len(r.Tag) > 0:
		return r.Name + ":" + r.Tag
	default:
		return r.Name
	}
}

// Exact returns a string representation of the set fields on the DockerImageReference
func (r DockerImageReference) Exact() string {
	name := r.NameString()
	if len(name) == 0 {
		return name
	}
	s := r.Registry
	if len(s) > 0 {
		s += "/"
	}

	if len(r.Namespace) != 0 {
		s += r.Namespace + "/"
	}
	return s + name
}

// String converts a DockerImageReference to a Docker pull spec (which implies a default namespace
// according to V1 Docker registry rules). Use Exact() if you want no defaulting.
func (r DockerImageReference) String() string {
	if len(r.Namespace) == 0 && IsRegistryDockerHub(r.Registry) {
		r.Namespace = "library"
	}
	return r.Exact()
}

// IsRegistryDockerHub returns true if the given registry name belongs to
// Docker hub.
func IsRegistryDockerHub(registry string) bool {
	switch registry {
	case DockerDefaultRegistry, DockerDefaultV1Registry, DockerDefaultV2Registry:
		return true
	default:
		return false
	}
}

// DeepCopyInto writing into out. in must be non-nil.
func (in *DockerImageReference) DeepCopyInto(out *DockerImageReference) {
	*out = *in
	return
}

// DeepCopy copies the receiver, creating a new DockerImageReference.
func (in *DockerImageReference) DeepCopy() *DockerImageReference {
	if in == nil {
		return nil
	}
	out := new(DockerImageReference)
	in.DeepCopyInto(out)
	return out
}
