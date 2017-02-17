package slb

import "github.com/denverdino/aliyungo/common"

type TagItem struct {
	TagKey   string
	TagValue string
}

type AddTagsArgs struct {
	RegionId       common.Region
	LoadBalancerID string
	Tags           string
}

type AddTagsResponse struct {
	common.Response
}

// AddTags Add tags to resource
//
// You can read doc at https://help.aliyun.com/document_detail/42871.html
func (client *Client) AddTags(args *AddTagsArgs) error {
	response := AddTagsResponse{}
	err := client.Invoke("AddTags", args, &response)
	if err != nil {
		return err
	}
	return err
}

type RemoveTagsArgs struct {
	RegionId       common.Region
	LoadBalancerID string
	Tags           string
}

type RemoveTagsResponse struct {
	common.Response
}

// RemoveTags remove tags to resource
//
// You can read doc at https://help.aliyun.com/document_detail/42872.html
func (client *Client) RemoveTags(args *RemoveTagsArgs) error {
	response := RemoveTagsResponse{}
	err := client.Invoke("RemoveTags", args, &response)
	if err != nil {
		return err
	}
	return err
}

type TagItemType struct {
	TagItem
	InstanceCount int
}

type DescribeTagsArgs struct {
	RegionId       common.Region
	LoadBalancerID string
	Tags           string
	common.Pagination
}

type DescribeTagsResponse struct {
	common.Response
	common.PaginationResult
	TagSets struct {
		TagSet []TagItemType
	}
}

// DescribeResourceByTags describe resource by tags
//
// You can read doc at https://help.aliyun.com/document_detail/42873.html?spm=5176.doc42872.6.267.CP1iWu
func (client *Client) DescribeTags(args *DescribeTagsArgs) (tags []TagItemType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeTagsResponse{}
	err = client.Invoke("DescribeTags", args, &response)
	if err != nil {
		return nil, nil, err
	}
	return response.TagSets.TagSet, &response.PaginationResult, nil
}
