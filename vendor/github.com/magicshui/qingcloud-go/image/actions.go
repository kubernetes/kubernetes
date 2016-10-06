package image

import (
	"github.com/magicshui/qingcloud-go"
)

type IMAGE struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *IMAGE {
	return &IMAGE{
		Client: clt,
	}
}

type DescribeImagesRequest struct {
	ImagesN       qingcloud.NumberedString
	ProcessorType qingcloud.String
	OsFamily      qingcloud.String
	Visibility    qingcloud.String
	Provider      qingcloud.String
	StatusN       qingcloud.NumberedString

	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeImagesResponse struct {
	TotalCount int     `json:"total_count"`
	ImageSet   []Image `json:"image_set"`
	qingcloud.CommonResponse
}

// DescribeImages
// 获取一个或多个映像
// 可根据映像ID，状态，映像名称、操作系统平台作过滤条件，来获取映像列表。 如果不指定任何过滤条件，默认返回你所拥有的所有映像。 如果指定不存在的映像ID，或非法状态值，则会返回错误信息。
func DescribeImages(c *qingcloud.Client, params DescribeImagesRequest) (DescribeImagesResponse, error) {
	var result DescribeImagesResponse
	err := c.Get("DescribeImages", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CaptureInstanceRequest struct {
	Instance  qingcloud.String
	ImageName qingcloud.String
}
type CaptureInstanceResponse struct {
	ImageId string `json:"image_id"`
	qingcloud.CommonResponse
}

// CaptureInstance
// 将某个已关闭的主机制作成模板（或称“自有映像”），之后便可将其用于创建新的主机。 被捕获的主机必须是已关闭（ stopped ）状态，否则会返回错误。
// 由主机制成的自有映像，会保留主机中安装的软件、配置及数据， 因此基于这个自有映像创建的主机，就直接获得了相同的系统环境。
func CaptureInstance(c *qingcloud.Client, params CaptureInstanceRequest) (CaptureInstanceResponse, error) {
	var result CaptureInstanceResponse
	err := c.Get("CaptureInstance", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteImagesRequest struct {
	ImagesN qingcloud.NumberedString
}
type DeleteImagesResponse qingcloud.CommonResponse

// DeleteImages
// 删除一个或多个自有映像。映像须在可用（ available ） 状态下才能被删除。
func DeleteImages(c *qingcloud.Client, params DeleteImagesRequest) (DeleteImagesResponse, error) {
	var result DeleteImagesResponse
	err := c.Get("DeleteImages", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyImageAttributesRequest struct {
	Image       qingcloud.String
	ImageName   qingcloud.String
	Description qingcloud.String
}
type ModifyImageAttributesResponse qingcloud.CommonResponse

// ModifyImageAttributes
// 修改映像的名称和描述。
// 修改时不受映像状态限制。
// 一次只能修改一个映像。
func ModifyImageAttributes(c *qingcloud.Client, params ModifyImageAttributesRequest) (ModifyImageAttributesResponse, error) {
	var result ModifyImageAttributesResponse
	err := c.Get("ModifyImageAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type GrantImageToUsersRequest struct {
	Image  qingcloud.String
	UsersN qingcloud.NumberedString
}
type GrantImageToUsersResponse qingcloud.CommonResponse

// GrantImageToUsers
// 共享镜像给指定的用户。
func GrantImageToUsers(c *qingcloud.Client, params GrantImageToUsersRequest) (GrantImageToUsersResponse, error) {
	var result GrantImageToUsersResponse
	err := c.Get("GrantImageToUsers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type RevokeImageFromUsersRequest struct {
	Image  qingcloud.String
	UsersN qingcloud.NumberedString
}
type RevokeImageFromUsersResponse qingcloud.CommonResponse

// RevokeImageFromUsers
// 撤销共享给用户。
func RevokeImageFromUsers(c *qingcloud.Client, params RevokeImageFromUsersRequest) (RevokeImageFromUsersResponse, error) {
	var result RevokeImageFromUsersResponse
	err := c.Get("RevokeImageFromUsers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeImageUsersRequest struct {
	ImageId qingcloud.String
	Offset  qingcloud.Integer
	Limit   qingcloud.Integer
}
type DescribeImageUsersResponse struct {
	ImageUserSet []ImageUser `json:"image_user_set"`
	qingcloud.CommonResponse
}

// DescribeImageUsers
// 可根据映像ID, 查询该映像分享给的用户的列表
// 如果指定不存在的映像ID，或非法状态值，则会返回错误信息。
func DescribeImageUsers(c *qingcloud.Client, params DescribeImageUsersRequest) (DescribeImageUsersResponse, error) {
	var result DescribeImageUsersResponse
	err := c.Get("DescribeImageUsers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
