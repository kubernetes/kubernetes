package images

import (
	"time"
)

// ImageStatus image statuses
// http://docs.openstack.org/developer/glance/statuses.html
type ImageStatus string

const (
	// ImageStatusQueued is a status for an image which identifier has
	// been reserved for an image in the image registry.
	ImageStatusQueued ImageStatus = "queued"

	// ImageStatusSaving denotes that an image’s raw data is currently being
	// uploaded to Glance
	ImageStatusSaving ImageStatus = "saving"

	// ImageStatusActive denotes an image that is fully available in Glance.
	ImageStatusActive ImageStatus = "active"

	// ImageStatusKilled denotes that an error occurred during the uploading
	// of an image’s data, and that the image is not readable.
	ImageStatusKilled ImageStatus = "killed"

	// ImageStatusDeleted is used for an image that is no longer available to use.
	// The image information is retained in the image registry.
	ImageStatusDeleted ImageStatus = "deleted"

	// ImageStatusPendingDelete is similar to Delete, but the image is not yet
	// deleted.
	ImageStatusPendingDelete ImageStatus = "pending_delete"

	// ImageStatusDeactivated denotes that access to image data is not allowed to
	// any non-admin user.
	ImageStatusDeactivated ImageStatus = "deactivated"
)

// ImageVisibility denotes an image that is fully available in Glance.
// This occurs when the image data is uploaded, or the image size is explicitly
// set to zero on creation.
// According to design
// https://wiki.openstack.org/wiki/Glance-v2-community-image-visibility-design
type ImageVisibility string

const (
	// ImageVisibilityPublic all users
	ImageVisibilityPublic ImageVisibility = "public"

	// ImageVisibilityPrivate users with tenantId == tenantId(owner)
	ImageVisibilityPrivate ImageVisibility = "private"

	// ImageVisibilityShared images are visible to:
	// - users with tenantId == tenantId(owner)
	// - users with tenantId in the member-list of the image
	// - users with tenantId in the member-list with member_status == 'accepted'
	ImageVisibilityShared ImageVisibility = "shared"

	// ImageVisibilityCommunity images:
	// - all users can see and boot it
	// - users with tenantId in the member-list of the image with
	//	 member_status == 'accepted' have this image in their default image-list.
	ImageVisibilityCommunity ImageVisibility = "community"
)

// MemberStatus is a status for adding a new member (tenant) to an image
// member list.
type ImageMemberStatus string

const (
	// ImageMemberStatusAccepted is the status for an accepted image member.
	ImageMemberStatusAccepted ImageMemberStatus = "accepted"

	// ImageMemberStatusPending shows that the member addition is pending
	ImageMemberStatusPending ImageMemberStatus = "pending"

	// ImageMemberStatusAccepted is the status for a rejected image member
	ImageMemberStatusRejected ImageMemberStatus = "rejected"

	// ImageMemberStatusAll
	ImageMemberStatusAll ImageMemberStatus = "all"
)

// ImageDateFilter represents a valid filter to use for filtering
// images by their date during a List.
type ImageDateFilter string

const (
	FilterGT  ImageDateFilter = "gt"
	FilterGTE ImageDateFilter = "gte"
	FilterLT  ImageDateFilter = "lt"
	FilterLTE ImageDateFilter = "lte"
	FilterNEQ ImageDateFilter = "neq"
	FilterEQ  ImageDateFilter = "eq"
)

// ImageDateQuery represents a date field to be used for listing images.
// If no filter is specified, the query will act as though FilterEQ was
// set.
type ImageDateQuery struct {
	Date   time.Time
	Filter ImageDateFilter
}
