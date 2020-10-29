package imageutil

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/blang/semver"

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

// ResolveLatestTaggedImage returns the appropriate pull spec for a given tag in
// the image stream, handling the tag's reference policy if necessary to return
// a resolved image. Callers that transform an ImageStreamTag into a pull spec
// should use this method instead of LatestTaggedImage.
func ResolveLatestTaggedImage(stream *imagev1.ImageStream, tag string) (string, bool) {
	if len(tag) == 0 {
		tag = imagev1.DefaultImageTag
	}
	return resolveTagReference(stream, tag, LatestTaggedImage(stream, tag))
}

// ResolveTagReference applies the tag reference rules for a stream, tag, and tag event for
// that tag. It returns true if the tag is
func resolveTagReference(stream *imagev1.ImageStream, tag string, latest *imagev1.TagEvent) (string, bool) {
	if latest == nil {
		return "", false
	}
	return resolveReferenceForTagEvent(stream, tag, latest), true
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

// ResolveReferenceForTagEvent applies the tag reference rules for a stream, tag, and tag event for
// that tag.
func resolveReferenceForTagEvent(stream *imagev1.ImageStream, tag string, latest *imagev1.TagEvent) string {
	// retrieve spec policy - if not found, we use the latest spec
	ref, ok := SpecHasTag(stream, tag)
	if !ok {
		return latest.DockerImageReference
	}

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
