package dnsalias

import (
	"github.com/magicshui/qingcloud-go"
)

type DescribeDNSAliasesRequest struct {
	DnsAliasesN qingcloud.NumberedString
	ResourcesId qingcloud.String

	SearchWord qingcloud.String
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeDNSAliasesResponse struct {
	qingcloud.CommonResponse
	TotalCount  int        `json:"total_count"`
	DnsAliasSet []DnsAlias `json:"dns_alias_set"`
}

// DescribeDNSAliases
// 可根据资源 ID 作过滤条件，获取内网域名别名列表。 如果不指定任何过滤条件，默认返回你所拥有的所有内网域名别名。
func DescribeDNSAliases(c *qingcloud.Client, params DescribeDNSAliasesRequest) (DescribeDNSAliasesResponse, error) {
	var result DescribeDNSAliasesResponse
	err := c.Get("DescribeDNSAliases", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AssociateDNSAliasRequest struct {
	Prefix       qingcloud.String
	Resouce      qingcloud.String
	DnsAliasName qingcloud.String
}
type AssociateDNSAliasResponse struct {
	DNSAliasID string `json:"dns_alias_id"`
	DomainName string `json:"domain_name"`
	qingcloud.CommonResponse
}

// AssociateDNSAlias
// 绑定内网域名别名到资源，资源可以是处于基础网络的主机，以及路由器。
func AssociateDNSAlias(c *qingcloud.Client, params AssociateDNSAliasRequest) (AssociateDNSAliasResponse, error) {
	var result AssociateDNSAliasResponse
	err := c.Get("AssociateDNSAlias", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DissociateDNSAliasesRequest struct {
	// TODO:文档有错误
	DnsAliasN qingcloud.NumberedString
}
type DissociateDNSAliasesResponse qingcloud.CommonResponse

// DissociateDNSAliases
// 从资源上解绑一个或多个内网域名。
func DissociateDNSAliases(c *qingcloud.Client, params DissociateDNSAliasesRequest) (DissociateDNSAliasesResponse, error) {
	var result DissociateDNSAliasesResponse
	err := c.Get("DissociateDNSAliases", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type GetDNSLabelRequest struct{}
type GetDNSLabelResponse struct {
	DnsLabel   string
	DomainName string
	qingcloud.CommonResponse
}

// GetDNSLabel
// 获取内网域名标记(label) 和域名名称(domain name)。当给资源绑定内网域名时，此标记会与 prefix 一起组成内网域名，即： 内网域名 ＝ prefix + domain name
func GetDNSLabel(c *qingcloud.Client, params GetDNSLabelRequest) (GetDNSLabelResponse, error) {
	var result GetDNSLabelResponse
	err := c.Get("GetDNSLabel", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
