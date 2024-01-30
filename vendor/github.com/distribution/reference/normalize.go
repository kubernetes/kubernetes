package reference

import (
	"fmt"
	"strings"

	"github.com/opencontainers/go-digest"
)

const (
	// legacyDefaultDomain is the legacy domain for Docker Hub (which was
	// originally named "the Docker Index"). This domain is still used for
	// authentication and image search, which were part of the "v1" Docker
	// registry specification.
	//
	// This domain will continue to be supported, but there are plans to consolidate
	// legacy domains to new "canonical" domains. Once those domains are decided
	// on, we must update the normalization functions, but preserve compatibility
	// with existing installs, clients, and user configuration.
	legacyDefaultDomain = "index.docker.io"

	// defaultDomain is the default domain used for images on Docker Hub.
	// It is used to normalize "familiar" names to canonical names, for example,
	// to convert "ubuntu" to "docker.io/library/ubuntu:latest".
	//
	// Note that actual domain of Docker Hub's registry is registry-1.docker.io.
	// This domain will continue to be supported, but there are plans to consolidate
	// legacy domains to new "canonical" domains. Once those domains are decided
	// on, we must update the normalization functions, but preserve compatibility
	// with existing installs, clients, and user configuration.
	defaultDomain = "docker.io"

	// officialRepoPrefix is the namespace used for official images on Docker Hub.
	// It is used to normalize "familiar" names to canonical names, for example,
	// to convert "ubuntu" to "docker.io/library/ubuntu:latest".
	officialRepoPrefix = "library/"

	// defaultTag is the default tag if no tag is provided.
	defaultTag = "latest"
)

// normalizedNamed represents a name which has been
// normalized and has a familiar form. A familiar name
// is what is used in Docker UI. An example normalized
// name is "docker.io/library/ubuntu" and corresponding
// familiar name of "ubuntu".
type normalizedNamed interface {
	Named
	Familiar() Named
}

// ParseNormalizedNamed parses a string into a named reference
// transforming a familiar name from Docker UI to a fully
// qualified reference. If the value may be an identifier
// use ParseAnyReference.
func ParseNormalizedNamed(s string) (Named, error) {
	if ok := anchoredIdentifierRegexp.MatchString(s); ok {
		return nil, fmt.Errorf("invalid repository name (%s), cannot specify 64-byte hexadecimal strings", s)
	}
	domain, remainder := splitDockerDomain(s)
	var remote string
	if tagSep := strings.IndexRune(remainder, ':'); tagSep > -1 {
		remote = remainder[:tagSep]
	} else {
		remote = remainder
	}
	if strings.ToLower(remote) != remote {
		return nil, fmt.Errorf("invalid reference format: repository name (%s) must be lowercase", remote)
	}

	ref, err := Parse(domain + "/" + remainder)
	if err != nil {
		return nil, err
	}
	named, isNamed := ref.(Named)
	if !isNamed {
		return nil, fmt.Errorf("reference %s has no name", ref.String())
	}
	return named, nil
}

// namedTaggedDigested is a reference that has both a tag and a digest.
type namedTaggedDigested interface {
	NamedTagged
	Digested
}

// ParseDockerRef normalizes the image reference following the docker convention,
// which allows for references to contain both a tag and a digest. It returns a
// reference that is either tagged or digested. For references containing both
// a tag and a digest, it returns a digested reference. For example, the following
// reference:
//
//	docker.io/library/busybox:latest@sha256:7cc4b5aefd1d0cadf8d97d4350462ba51c694ebca145b08d7d41b41acc8db5aa
//
// Is returned as a digested reference (with the ":latest" tag removed):
//
//	docker.io/library/busybox@sha256:7cc4b5aefd1d0cadf8d97d4350462ba51c694ebca145b08d7d41b41acc8db5aa
//
// References that are already "tagged" or "digested" are returned unmodified:
//
//	// Already a digested reference
//	docker.io/library/busybox@sha256:7cc4b5aefd1d0cadf8d97d4350462ba51c694ebca145b08d7d41b41acc8db5aa
//
//	// Already a named reference
//	docker.io/library/busybox:latest
func ParseDockerRef(ref string) (Named, error) {
	named, err := ParseNormalizedNamed(ref)
	if err != nil {
		return nil, err
	}
	if canonical, ok := named.(namedTaggedDigested); ok {
		// The reference is both tagged and digested; only return digested.
		newNamed, err := WithName(canonical.Name())
		if err != nil {
			return nil, err
		}
		return WithDigest(newNamed, canonical.Digest())
	}
	return TagNameOnly(named), nil
}

// splitDockerDomain splits a repository name to domain and remote-name.
// If no valid domain is found, the default domain is used. Repository name
// needs to be already validated before.
func splitDockerDomain(name string) (domain, remainder string) {
	i := strings.IndexRune(name, '/')
	if i == -1 || (!strings.ContainsAny(name[:i], ".:") && name[:i] != localhost && strings.ToLower(name[:i]) == name[:i]) {
		domain, remainder = defaultDomain, name
	} else {
		domain, remainder = name[:i], name[i+1:]
	}
	if domain == legacyDefaultDomain {
		domain = defaultDomain
	}
	if domain == defaultDomain && !strings.ContainsRune(remainder, '/') {
		remainder = officialRepoPrefix + remainder
	}
	return
}

// familiarizeName returns a shortened version of the name familiar
// to the Docker UI. Familiar names have the default domain
// "docker.io" and "library/" repository prefix removed.
// For example, "docker.io/library/redis" will have the familiar
// name "redis" and "docker.io/dmcgowan/myapp" will be "dmcgowan/myapp".
// Returns a familiarized named only reference.
func familiarizeName(named namedRepository) repository {
	repo := repository{
		domain: named.Domain(),
		path:   named.Path(),
	}

	if repo.domain == defaultDomain {
		repo.domain = ""
		// Handle official repositories which have the pattern "library/<official repo name>"
		if strings.HasPrefix(repo.path, officialRepoPrefix) {
			// TODO(thaJeztah): this check may be too strict, as it assumes the
			//  "library/" namespace does not have nested namespaces. While this
			//  is true (currently), technically it would be possible for Docker
			//  Hub to use those (e.g. "library/distros/ubuntu:latest").
			//  See https://github.com/distribution/distribution/pull/3769#issuecomment-1302031785.
			if remainder := strings.TrimPrefix(repo.path, officialRepoPrefix); !strings.ContainsRune(remainder, '/') {
				repo.path = remainder
			}
		}
	}
	return repo
}

func (r reference) Familiar() Named {
	return reference{
		namedRepository: familiarizeName(r.namedRepository),
		tag:             r.tag,
		digest:          r.digest,
	}
}

func (r repository) Familiar() Named {
	return familiarizeName(r)
}

func (t taggedReference) Familiar() Named {
	return taggedReference{
		namedRepository: familiarizeName(t.namedRepository),
		tag:             t.tag,
	}
}

func (c canonicalReference) Familiar() Named {
	return canonicalReference{
		namedRepository: familiarizeName(c.namedRepository),
		digest:          c.digest,
	}
}

// TagNameOnly adds the default tag "latest" to a reference if it only has
// a repo name.
func TagNameOnly(ref Named) Named {
	if IsNameOnly(ref) {
		namedTagged, err := WithTag(ref, defaultTag)
		if err != nil {
			// Default tag must be valid, to create a NamedTagged
			// type with non-validated input the WithTag function
			// should be used instead
			panic(err)
		}
		return namedTagged
	}
	return ref
}

// ParseAnyReference parses a reference string as a possible identifier,
// full digest, or familiar name.
func ParseAnyReference(ref string) (Reference, error) {
	if ok := anchoredIdentifierRegexp.MatchString(ref); ok {
		return digestReference("sha256:" + ref), nil
	}
	if dgst, err := digest.Parse(ref); err == nil {
		return digestReference(dgst), nil
	}

	return ParseNormalizedNamed(ref)
}
