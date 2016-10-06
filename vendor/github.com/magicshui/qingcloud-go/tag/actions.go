package tag

import (
	"github.com/magicshui/qingcloud-go"
)

type DescribeTagsRequest struct {
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeTagsResponse struct {
	qingcloud.CommonResponse
	TotalCount int   `json:"total_count"`
	TagSet     []Tag `json:"tag_set"`
}

// DescribeTags
// 获取一个或多个标签
// 可根据标签ID，名称作为过滤条件，获取标签列表。 如果不指定任何过滤条件，默认返回你所拥有的所有标签。
func DescribeTags(c *qingcloud.Client, params DescribeTagsRequest) (DescribeTagsResponse, error) {
	var result DescribeTagsResponse
	err := c.Get("DescribeTags", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateTagRequest struct {
	TagName qingcloud.String
}
type CreateTagResponse struct {
	TagName string `json:"tag_name"`
	qingcloud.CommonResponse
}

// CreateTag
// 创建标签，每个标签可以绑定多个资源。
// 注意: 标签名称少于15个字符, 不可重复.
// 标签数据可以随时通过 DescribeTags 得到。
func CreateTag(c *qingcloud.Client, params CreateTagRequest) (CreateTagResponse, error) {
	var result CreateTagResponse
	err := c.Get("CreateTag", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteTagsRequest struct {
	TagsN qingcloud.NumberedString
}
type DeleteTagsResponse struct {
	Tags []string `json:"tags"`
	qingcloud.CommonResponse
}

// DeleteTags
// 删除一个或多个你拥有的标签，该标签绑定的所有资源自动解除绑定关系 关于解绑标签可参考 DetachTags
func DeleteTags(c *qingcloud.Client, params DeleteTagsRequest) (DeleteTagsResponse, error) {
	var result DeleteTagsResponse
	err := c.Get("DeleteTags", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyTagAttributesRequest struct {
	Tag         qingcloud.String
	TagName     qingcloud.String
	Description qingcloud.String
}
type ModifyTagAttributesResponse qingcloud.CommonResponse

// ModifyTagAttributes 修改标签的名称和描述。
// 一次只能修改一个标签。
func ModifyTagAttributes(c *qingcloud.Client, params ModifyTagAttributesRequest) (ModifyTagAttributesResponse, error) {
	var result ModifyTagAttributesResponse
	err := c.Get("ModifyTagAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AttachTagsRequest struct {
	ResourceTagPairsNTagId        qingcloud.NumberedString
	ResourceTagPairsNResourceType qingcloud.NumberedString
	ResourceTagPairsNResourceId   qingcloud.NumberedString
}
type AttachTagsResponse qingcloud.CommonResponse

// AttachTags
// 将标签绑定到资源上, 绑定之后，获取资源列表（例如 DescribeInstances） 的时候，可以传参数tags来过滤该标签的资源, 获取资源列表(例如DescribeInstances), 资源详情也会包含已绑定的标签信息
func AttachTags(c *qingcloud.Client, params AttachTagsRequest) (AttachTagsResponse, error) {
	var result AttachTagsResponse
	err := c.Get("AttachTags", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DetachTagsRequest struct {
	ResourceTagPairsNTagId        qingcloud.NumberedString
	ResourceTagPairsNResourceType qingcloud.NumberedString
	ResourceTagPairsNResourceId   qingcloud.NumberedString
}
type DetachTagsResponse qingcloud.CommonResponse

// DetachTags
// 将标签从资源上解绑
func DetachTags(c *qingcloud.Client, params DetachTagsRequest) (DetachTagsResponse, error) {
	var result DetachTagsResponse
	err := c.Get("DetachTags", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
