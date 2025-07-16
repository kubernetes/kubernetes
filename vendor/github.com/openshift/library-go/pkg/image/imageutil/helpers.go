package imageutil

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/blang/semver/v4"

	"github.com/openshift/api/image/docker10"
	imagev1 "github.com/openshift/api/image/v1"
	digestinternal "github.com/openshift/library-go/pkg/image/internal/digest"
	imagereference "github.com/openshift/library-go/pkg/image/reference"
)

const (
	// DefaultImageTag is used when an image tag is needed and the configuration does not specify a tag to use.
	DefaultImageTag = "latest"
)

var ParseDigest = digestinternal.ParseDigest

// SplitImageStreamTag turns the name of an ImageStreamTag into Name and Tag.
// It returns false if the tag was not properly specified in the name.
func SplitImageStreamTag(nameAndTag string) (name string, tag string, ok bool) {
	parts := strings.SplitN(nameAndTag, ":", 2)
	name = parts[0]
	if len(parts) > 1 {
		tag = parts[1]
	}
	if len(tag) == 0 {
		tag = DefaultImageTag
	}
	return name, tag, len(parts) == 2
}

// SplitImageStreamImage turns the name of an ImageStreamImage into Name and ID.
// It returns false if the ID was not properly specified in the name.
func SplitImageStreamImage(nameAndID string) (name string, id string, ok bool) {
	parts := strings.SplitN(nameAndID, "@", 2)
	name = parts[0]
	if len(parts) > 1 {
		id = parts[1]
	}
	return name, id, len(parts) == 2
}

// JoinImageStreamTag turns a name and tag into the name of an ImageStreamTag
func JoinImageStreamTag(name, tag string) string {
	if len(tag) == 0 {
		tag = DefaultImageTag
	}
	return fmt.Sprintf("%s:%s", name, tag)
}

// JoinImageStreamImage creates a name for image stream image object from an image stream name and an id.
func JoinImageStreamImage(name, id string) string {
	return fmt.Sprintf("%s@%s", name, id)
}

// ParseImageStreamTagName splits a string into its name component and tag component, and returns an error
// if the string is not in the right form.
func ParseImageStreamTagName(istag string) (name string, tag string, err error) {
	if strings.Contains(istag, "@") {
		err = fmt.Errorf("%q is an image stream image, not an image stream tag", istag)
		return
	}
	segments := strings.SplitN(istag, ":", 3)
	switch len(segments) {
	case 2:
		name = segments[0]
		tag = segments[1]
		if len(name) == 0 || len(tag) == 0 {
			err = fmt.Errorf("image stream tag name %q must have a name and a tag", istag)
		}
	default:
		err = fmt.Errorf("expected exactly one : delimiter in the istag %q", istag)
	}
	return
}

// ParseImageStreamImageName splits a string into its name component and ID component, and returns an error
// if the string is not in the right form.
func ParseImageStreamImageName(input string) (name string, id string, err error) {
	segments := strings.SplitN(input, "@", 3)
	switch len(segments) {
	case 2:
		name = segments[0]
		id = segments[1]
		if len(name) == 0 || len(id) == 0 {
			err = fmt.Errorf("image stream image name %q must have a name and ID", input)
		}
	default:
		err = fmt.Errorf("expected exactly one @ in the isimage name %q", input)
	}
	return
}

var (
	reMinorSemantic  = regexp.MustCompile(`^[\d]+\.[\d]+$`)
	reMinorWithPatch = regexp.MustCompile(`^([\d]+\.[\d]+)-\w+$`)
)

type tagPriority int

const (
	// the "latest" tag
	tagPriorityLatest tagPriority = iota

	// a semantic minor version ("5.1", "v5.1", "v5.1-rc1")
	tagPriorityMinor

	// a full semantic version ("5.1.3-other", "v5.1.3-other")
	tagPriorityFull

	// other tags
	tagPriorityOther
)

type prioritizedTag struct {
	tag      string
	priority tagPriority
	semver   semver.Version
	prefix   string
}

func prioritizeTag(tag string) prioritizedTag {
	if tag == "latest" {
		return prioritizedTag{
			tag:      tag,
			priority: tagPriorityLatest,
		}
	}

	short := tag
	prefix := ""
	if strings.HasPrefix(tag, "v") {
		prefix = "v"
		short = tag[1:]
	}

	// 5.1.3
	if v, err := semver.Parse(short); err == nil {
		return prioritizedTag{
			tag:      tag,
			priority: tagPriorityFull,
			semver:   v,
			prefix:   prefix,
		}
	}

	// 5.1
	if reMinorSemantic.MatchString(short) {
		if v, err := semver.Parse(short + ".0"); err == nil {
			return prioritizedTag{
				tag:      tag,
				priority: tagPriorityMinor,
				semver:   v,
				prefix:   prefix,
			}
		}
	}

	// 5.1-rc1
	if match := reMinorWithPatch.FindStringSubmatch(short); match != nil {
		if v, err := semver.Parse(strings.Replace(short, match[1], match[1]+".0", 1)); err == nil {
			return prioritizedTag{
				tag:      tag,
				priority: tagPriorityMinor,
				semver:   v,
				prefix:   prefix,
			}
		}
	}

	// other
	return prioritizedTag{
		tag:      tag,
		priority: tagPriorityOther,
		prefix:   prefix,
	}
}

type prioritizedTags []prioritizedTag

func (t prioritizedTags) Len() int      { return len(t) }
func (t prioritizedTags) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t prioritizedTags) Less(i, j int) bool {
	if t[i].priority != t[j].priority {
		return t[i].priority < t[j].priority
	}

	if t[i].priority == tagPriorityOther {
		return t[i].tag < t[j].tag
	}

	cmp := t[i].semver.Compare(t[j].semver)
	if cmp > 0 { // the newer tag has a higher priority
		return true
	}
	return cmp == 0 && t[i].prefix < t[j].prefix
}

// PrioritizeTags orders a set of image tags with a few conventions:
//
// 1. the "latest" tag, if present, should be first
// 2. any tags that represent a semantic minor version ("5.1", "v5.1", "v5.1-rc1") should be next, in descending order
// 3. any tags that represent a full semantic version ("5.1.3-other", "v5.1.3-other") should be next, in descending order
// 4. any remaining tags should be sorted in lexicographic order
//
// The method updates the tags in place.
func PrioritizeTags(tags []string) {
	ptags := make(prioritizedTags, len(tags))
	for i, tag := range tags {
		ptags[i] = prioritizeTag(tag)
	}
	sort.Sort(ptags)
	for i, pt := range ptags {
		tags[i] = pt.tag
	}
}

// SpecHasTag returns named tag from image stream's spec and boolean whether one was found.
func SpecHasTag(stream *imagev1.ImageStream, name string) (imagev1.TagReference, bool) {
	for _, tag := range stream.Spec.Tags {
		if tag.Name == name {
			return tag, true
		}
	}
	return imagev1.TagReference{}, false
}

// StatusHasTag returns named tag from image stream's status and boolean whether one was found.
func StatusHasTag(stream *imagev1.ImageStream, name string) (imagev1.NamedTagEventList, bool) {
	for _, tag := range stream.Status.Tags {
		if tag.Tag == name {
			return tag, true
		}
	}
	return imagev1.NamedTagEventList{}, false
}

// LatestTaggedImage returns the most recent TagEvent for the specified image
// repository and tag. Will resolve lookups for the empty tag. Returns nil
// if tag isn't present in stream.status.tags.
func LatestTaggedImage(stream *imagev1.ImageStream, tag string) *imagev1.TagEvent {
	if len(tag) == 0 {
		tag = imagev1.DefaultImageTag
	}

	// find the most recent tag event with an image reference
	t, ok := StatusHasTag(stream, tag)
	if ok {
		if len(t.Items) == 0 {
			return nil
		}
		return &t.Items[0]
	}

	return nil
}

// ImageWithMetadata mutates the given image. It parses raw DockerImageManifest data stored in the image and
// fills its DockerImageMetadata and other fields.
// Copied from github.com/openshift/image-registry/pkg/origin-common/util/util.go
func ImageWithMetadata(image *imagev1.Image) error {
	// Check if the metadata are already filled in for this image.
	meta, hasMetadata := image.DockerImageMetadata.Object.(*docker10.DockerImage)
	if hasMetadata && meta.Size > 0 {
		return nil
	}

	version := image.DockerImageMetadataVersion
	if len(version) == 0 {
		version = "1.0"
	}

	obj := &docker10.DockerImage{}
	if len(image.DockerImageMetadata.Raw) != 0 {
		if err := json.Unmarshal(image.DockerImageMetadata.Raw, obj); err != nil {
			return err
		}
		image.DockerImageMetadata.Object = obj
	}

	image.DockerImageMetadataVersion = version

	return nil
}

func ImageWithMetadataOrDie(image *imagev1.Image) {
	if err := ImageWithMetadata(image); err != nil {
		panic(err)
	}
}

// TagReferencesLocalTag returns true if the provided tag reference references another image stream tag
// in the current image stream. This is only true when from points to an ImageStreamTag without a colon
// or from.name is <streamName>:<tag>.
func TagReferencesLocalTag(stream *imagev1.ImageStream, tag imagev1.TagReference) (string, bool) {
	if tag.From == nil || tag.From.Kind != "ImageStreamTag" {
		return "", false
	}
	if len(tag.From.Namespace) > 0 && tag.From.Namespace != stream.Namespace {
		return "", false
	}
	ref := strings.TrimPrefix(tag.From.Name, stream.Name+":")
	if strings.Contains(ref, ":") {
		return "", false
	}
	return ref, true
}

var (
	// ErrNoStreamRepository is returned if the status dockerImageRepository field was unset but the
	// method required that value to create a pull spec.
	ErrNoStreamRepository = fmt.Errorf("no image repository has been set on the image stream status")
	// ErrWaitForPullSpec is returned when a pull spec cannot be inferred from the image stream automatically
	// and the user requires a valid image tag.
	ErrWaitForPullSpec = fmt.Errorf("the pull spec cannot be determined yet")
)

// ResolveNewestPullSpecForTag returns the most recent available pull spec for the given tag, even
// if importing that pull spec is still in progress or has failed. Use this method when the current
// state of the tag as the user sees it is important because you don't want to silently ignore a
// newer tag request that hasn't yet been imported. Note that if no image has been tagged or pushed,
// pullSpec will still be returned pointing to the pull spec for the tag within the image repository
// (<status.dockerImageRepository>:<tag> unless defaultExternal is set) and isTagEmpty will be true.
// hasStatus is true if the returned pull spec points to an imported / pushed image, or false if
// a spec tag has not been specified, the spec tag hasn't been imported, or the import has failed.
// An error is returned only if isTagEmpty is true and status.dockerImageRepository is unset because
// the administrator has not installed a registry server.
//
// Use this method when you need the user intent pull spec and you do not want to tolerate a slightly
// older image (tooling that needs to error if the user's intent in tagging isn't realized).
func ResolveNewestPullSpecForTag(stream *imagev1.ImageStream, tag string, defaultExternal bool) (pullSpec string, hasStatus, isTagEmpty bool, err error) {
	pullSpec, _, hasStatus, isTagEmpty, err = resolvePullSpecForTag(stream, tag, defaultExternal, true)
	return pullSpec, hasStatus, isTagEmpty, err
}

// ResolveRecentPullSpecForTag returns the most recent successfully imported pull sec for the
// given tag, i.e. "last-known-good". Use this method when you can tolerate some lag in picking up
// the newest version. This method is roughly equivalent to the behavior of pulling the pod from
// the internal registry. If no image has been tagged or pushed, pullSpec will still be returned
// pointing to the pull spec for the tag within the image repository
// (<status.dockerImageRepository>:<tag> unless defaultExternal is set) and isTagEmpty will be true.
// hasNewer is true if the pull spec does not represent the newest user input, or false if the
// current user spec tag has been imported successfully. hasStatus is true if the returned pull
// spec points to an imported / pushed image, or false if a spec tag has not been specified, the
// spec tag hasn't been imported, or the import has failed. An error is returned only if isTagEmpty
// is true and status.dockerImageRepository is unset because the administrator has not installed a
// registry server.
//
// This method is typically used by consumers that need the value at the tag and prefer to have a
// slightly older image over not getting any image at all (or if the image can't be imported
// due to temporary network or controller issues).
func ResolveRecentPullSpecForTag(stream *imagev1.ImageStream, tag string, defaultExternal bool) (pullSpec string, hasNewer, hasStatus, isTagEmpty bool, err error) {
	pullSpec, hasNewer, hasStatus, isTagEmpty, err = resolvePullSpecForTag(stream, tag, defaultExternal, false)
	return pullSpec, hasNewer, hasStatus, isTagEmpty, err
}

// resolvePullSpecForTag handles finding the most accurate pull spec depending on whether the user
// requires the latest or simply wants the most recent imported version (ignores pending imports).
// If a pull spec cannot be inferred an error is returned. Otherwise the following status values are
// returned:
//
// * hasNewer - a newer version of this tag is being imported but is not ready
// * hasStatus - this pull spec points to the latest image in the status (has been imported / pushed)
// * isTagEmpty - no pull spec or push has occurred to this tag, but it's still possible to get a pull spec
//
// defaultExternal is considered when isTagEmpty is true (no user input provided) and calculates the pull
// spec from the external repository base (status.publicDockerImageRepository) if it is set.
func resolvePullSpecForTag(stream *imagev1.ImageStream, tag string, defaultExternal, requireLatest bool) (pullSpec string, hasNewer, hasStatus, isTagEmpty bool, err error) {
	if len(tag) == 0 {
		tag = imagev1.DefaultImageTag
	}
	status, _ := StatusHasTag(stream, tag)
	spec, hasSpec := SpecHasTag(stream, tag)
	hasSpecTagRef := hasSpec && spec.From != nil && spec.From.Kind == "DockerImage" && spec.ReferencePolicy.Type == imagev1.SourceTagReferencePolicy

	var event *imagev1.TagEvent
	switch {
	case len(status.Items) == 0:
		// nothing in status:
		// - waiting for import of first image (generation of spec > status)
		// - spec is empty
		// - spec is a ref tag to something else that hasn't been imported yet
		// - spec is a ref tag to another spec tag on this same image stream that doesn't exist

	case hasSpec && spec.Generation != nil && *spec.Generation > status.Items[0].Generation:
		// waiting for import because spec generation is newer and had a previous image
		if requireLatest {
			// note: if spec tag doesn't have a DockerImage kind, we'll have to wait for whatever
			// logic is necessary for import to run (this could happen if a new Kind is introduced)
			if !hasSpecTagRef {
				return "", hasNewer, false, false, ErrWaitForPullSpec
			}
		} else {
			event = &status.Items[0]
			hasNewer = true
		}
	default:
		// this is the latest version of the image
		event = &status.Items[0]
	}

	switch {
	case event != nil:
		hasStatus = true
		pullSpec = resolveReferenceForTagEvent(stream, spec, event)
	case hasSpecTagRef:
		// if the user explicitly provided a spec tag we can use
		pullSpec = resolveReferenceForTagEvent(stream, spec, &imagev1.TagEvent{
			DockerImageReference: spec.From.Name,
		})
	default:
		isTagEmpty = true
		repositorySpec := stream.Status.DockerImageRepository
		if defaultExternal && len(stream.Status.PublicDockerImageRepository) > 0 {
			repositorySpec = stream.Status.PublicDockerImageRepository
		}
		if len(repositorySpec) == 0 {
			return "", false, false, false, ErrNoStreamRepository
		}
		pullSpec = JoinImageStreamTag(repositorySpec, tag)
	}
	return pullSpec, hasNewer, hasStatus, isTagEmpty, nil
}

// ResolveLatestTaggedImage returns the appropriate pull spec for a given tag in
// the image stream, handling the tag's reference policy if necessary to return
// a resolved image. Callers that transform an ImageStreamTag into a pull spec
// should use this method instead of LatestTaggedImage. This method ignores pending
// imports (meaning the requested image may be stale) and will return no pull spec
// even if one is available on the spec tag (when importing kind DockerImage) if
// import has not completed.
//
// Use ResolvePullSpecForTag() if you wish more control over what type of pull spec
// is returned and what scenarios should be handled.
func ResolveLatestTaggedImage(stream *imagev1.ImageStream, tag string) (string, bool) {
	if len(tag) == 0 {
		tag = imagev1.DefaultImageTag
	}
	return resolveTagReference(stream, tag, LatestTaggedImage(stream, tag))
}

// ResolveTagReference applies the tag reference rules for a stream, tag, and tag event for
// that tag. It returns true if the tag is
func resolveTagReference(stream *imagev1.ImageStream, tag string, latest *imagev1.TagEvent) (string, bool) {
	// no image has been imported, so we can't resolve to a tagged image (we need an image id)
	if latest == nil {
		return "", false
	}
	// retrieve spec policy - if not found, we use the latest spec
	ref, ok := SpecHasTag(stream, tag)
	if !ok {
		return latest.DockerImageReference, true
	}
	return resolveReferenceForTagEvent(stream, ref, latest), true
}

// resolveReferenceForTagEvent applies the tag reference rules for a stream, tag, and tag event for
// that tag.
func resolveReferenceForTagEvent(stream *imagev1.ImageStream, ref imagev1.TagReference, latest *imagev1.TagEvent) string {
	switch ref.ReferencePolicy.Type {
	// the local reference policy attempts to use image pull through on the integrated
	// registry if possible
	case imagev1.LocalTagReferencePolicy:
		local := stream.Status.DockerImageRepository
		if len(local) == 0 || len(latest.Image) == 0 {
			// fallback to the originating reference if no local docker registry defined or we
			// lack an image ID
			return latest.DockerImageReference
		}

		// we must use imageapi's helper since we're calling Exact later on, which produces string
		ref, err := imagereference.Parse(local)
		if err != nil {
			// fallback to the originating reference if the reported local repository spec is not valid
			return latest.DockerImageReference
		}

		// create a local pullthrough URL
		ref.Tag = ""
		ref.ID = latest.Image
		return ref.Exact()

	// the default policy is to use the originating image
	default:
		return latest.DockerImageReference
	}
}

// DigestOrImageMatch matches the digest in the image name.
func DigestOrImageMatch(image, imageID string) bool {
	if d, err := ParseDigest(image); err == nil {
		return strings.HasPrefix(d.Hex(), imageID) || strings.HasPrefix(image, imageID)
	}
	return strings.HasPrefix(image, imageID)
}

// ParseDockerImageReference parses a Docker pull spec string into a
// DockerImageReference.
func ParseDockerImageReference(spec string) (imagev1.DockerImageReference, error) {
	ref, err := imagereference.Parse(spec)
	if err != nil {
		return imagev1.DockerImageReference{}, err
	}
	return imagev1.DockerImageReference{
		Registry:  ref.Registry,
		Namespace: ref.Namespace,
		Name:      ref.Name,
		Tag:       ref.Tag,
		ID:        ref.ID,
	}, nil
}
