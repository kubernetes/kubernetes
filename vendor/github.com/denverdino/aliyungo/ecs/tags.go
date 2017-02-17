package ecs

import "github.com/denverdino/aliyungo/common"

type TagResourceType string

const (
	TagResourceImage    = TagResourceType("image")
	TagResourceInstance = TagResourceType("instance")
	TagResourceSnapshot = TagResourceType("snapshot")
	TagResourceDisk     = TagResourceType("disk")
)

type AddTagsArgs struct {
	ResourceId   string
	ResourceType TagResourceType //image, instance, snapshot or disk
	RegionId     common.Region
	Tag          map[string]string
}

type AddTagsResponse struct {
	common.Response
}

// AddTags Add tags to resource
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/tags&addtags
func (client *Client) AddTags(args *AddTagsArgs) error {
	response := AddTagsResponse{}
	err := client.Invoke("AddTags", args, &response)
	return err
}

type RemoveTagsArgs struct {
	ResourceId   string
	ResourceType TagResourceType //image, instance, snapshot or disk
	RegionId     common.Region
	Tag          map[string]string
}

type RemoveTagsResponse struct {
	common.Response
}

// RemoveTags remove tags to resource
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/tags&removetags
func (client *Client) RemoveTags(args *RemoveTagsArgs) error {
	response := RemoveTagsResponse{}
	err := client.Invoke("RemoveTags", args, &response)
	return err
}

type ResourceItemType struct {
	ResourceId   string
	ResourceType TagResourceType
	RegionId     common.Region
}

type DescribeResourceByTagsArgs struct {
	ResourceType TagResourceType //image, instance, snapshot or disk
	RegionId     common.Region
	Tag          map[string]string
	common.Pagination
}

type DescribeResourceByTagsResponse struct {
	common.Response
	common.PaginationResult
	Resources struct {
		Resource []ResourceItemType
	}
}

// DescribeResourceByTags describe resource by tags
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/tags&describeresourcebytags
func (client *Client) DescribeResourceByTags(args *DescribeResourceByTagsArgs) (resources []ResourceItemType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeResourceByTagsResponse{}
	err = client.Invoke("DescribeResourceByTags", args, &response)
	if err != nil {
		return nil, nil, err
	}
	return response.Resources.Resource, &response.PaginationResult, nil
}

type TagItemType struct {
	TagKey   string
	TagValue string
}

type DescribeTagsArgs struct {
	RegionId     common.Region
	ResourceType TagResourceType //image, instance, snapshot or disk
	ResourceId   string
	Tag          map[string]string
	common.Pagination
}

type DescribeTagsResponse struct {
	common.Response
	common.PaginationResult
	Tags struct {
		Tag []TagItemType
	}
}

// DescribeResourceByTags describe resource by tags
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/tags&describeresourcebytags
func (client *Client) DescribeTags(args *DescribeTagsArgs) (tags []TagItemType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeTagsResponse{}
	err = client.Invoke("DescribeTags", args, &response)
	if err != nil {
		return nil, nil, err
	}
	return response.Tags.Tag, &response.PaginationResult, nil
}
