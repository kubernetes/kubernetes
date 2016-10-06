package keypair

import (
	"log"

	"github.com/magicshui/qingcloud-go"
)

// KEYPAIR 密钥服务
type KEYPAIR struct {
	*qingcloud.Client
}

// NewClient 创建新的密钥服务
func NewClient(clt *qingcloud.Client) *KEYPAIR {
	return &KEYPAIR{
		Client: clt,
	}
}

// DescribeKeyPairsRequest 查找密钥请求
type DescribeKeyPairsRequest struct {
	KeypairsN     qingcloud.NumberedString
	InstanceID    qingcloud.String
	EncryptMethod qingcloud.String

	StatusN    qingcloud.String
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}

// DescribeKeyPairsResponse 密钥请求结果
type DescribeKeyPairsResponse struct {
	qingcloud.CommonResponse
	TotalCount int       `json:"total_count"`
	KeypairSet []Keypair `json:"keypair_set"`
}

// DescribeKeyPairs 获取一个或多个 SSH 密钥
// 可根据密钥ID，密钥名称，主机ID，加密方式作为过滤条件，获取密钥列表。 如果不指定任何过滤条件，默认返回你所拥有的所有密钥。 如果指定不支持的加密方式，则会返回错误信息。
func (c *KEYPAIR) DescribeKeyPairs(params DescribeKeyPairsRequest) (DescribeKeyPairsResponse, error) {
	var result DescribeKeyPairsResponse
	err := c.Get("DescribeKeyPairs", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateKeyPairRequest struct {
	KeypairName   qingcloud.String
	Mode          qingcloud.String
	EncryptMethod qingcloud.String
	PublicKey     qingcloud.String
}
type CreateKeyPairResponse struct {
	PrivateKey string `json:"private_key"`
	KeypairId  string `json:"keypair_id"`
	qingcloud.CommonResponse
}

// CreateKeyPair
// 创建 SSH 密钥对，每对密钥都可加载到任意多台主机中。
// 支持以下两种加密算法：
//  + 1024-位 DSS
//  + 2048-位 RSA （默认）
// 创建密钥对成功后，请及时从 API 返回结果中保存私钥， 因为我们不会保存用户的私钥数据。 公钥数据可以随时通过 DescribeKeyPairs 得到。

// 另外用户也可以通过已有公钥来创建 SSH 密钥。
func (c *KEYPAIR) CreateKeyPair(params CreateKeyPairRequest) (CreateKeyPairResponse, error) {
	var result CreateKeyPairResponse
	params.Mode.Enum("system", "user")
	// optional
	params.EncryptMethod.Enum("ssh-rsa", "ssh-dss")

	err := c.Get("CreateKeyPair", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteKeyPairsRequest struct {
	KeypairsN qingcloud.NumberedString
}
type DeleteKeyPairsResponse struct {
	Keypris []string `json:"keypairs"`
	qingcloud.CommonResponse
}

// DeleteKeyPairs
// 删除一个或多个你拥有的密钥对。密钥对须在未使用的情况下才能被删除， 已加载到主机的密钥对需先卸载后才能删除， 关于卸载密钥对可参考 DetachKeyPairs
func (c *KEYPAIR) DeleteKeyPairs(params DeleteKeyPairsRequest) (DeleteKeyPairsResponse, error) {
	var result DeleteKeyPairsResponse
	log.Printf("Begint to parse", "")
	err := c.Get("DeleteKeyPairs", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AttachKeyPairsRequest struct {
	KeypairsN  qingcloud.NumberedString
	InstancesN qingcloud.NumberedString
}
type AttachKeyPairsResponse qingcloud.CommonResponse

// AttachKeyPairs
// 将任意数量密钥对加载到任意数量的主机， 主机状态须为“运行中”（ running ）或“已关机”（ stopped ）。
func (c *KEYPAIR) AttachKeyPairs(params AttachKeyPairsRequest) (AttachKeyPairsResponse, error) {
	var result AttachKeyPairsResponse
	err := c.Get("AttachKeyPairs", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DetachKeyPairsRequest struct {
	KeypairsN  qingcloud.NumberedString
	InstancesN qingcloud.NumberedString
}
type DetachKeyPairsResponse qingcloud.CommonResponse

// DetachKeyPairs
// 将任意数量的密钥对从主机中卸载， 主机状态须为“运行中”（ running ）或“已关机”（ stopped ）。
func (c *KEYPAIR) DetachKeyPairs(params DetachKeyPairsRequest) (DetachKeyPairsResponse, error) {
	var result DetachKeyPairsResponse
	err := c.Get("DetachKeyPairs", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyKeyPairAttributesRequest struct {
	Keypair     qingcloud.String
	KeypairName qingcloud.String
	Description qingcloud.String
}
type ModifyKeyPairAttributesResponse qingcloud.CommonResponse

// ModifyKeyPairAttributes 修改密钥对的名称和描述。
// 一次只能修改一个密钥对。
func (c *KEYPAIR) ModifyKeyPairAttributes(params ModifyKeyPairAttributesRequest) (ModifyKeyPairAttributesResponse, error) {
	var result ModifyKeyPairAttributesResponse
	err := c.Get("ModifyKeyPairAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
