package qingcloud

import (
	"crypto/hmac"
	"crypto/sha256"
	b64 "encoding/base64"
	"fmt"
	"net/url"
	"sort"
	"strings"
)

// Param 请求参数
type Param struct {
	Name  string
	Value interface{}
}

// Params 请求参数数组
type Params []*Param

// Len 长度
func (ps Params) Len() int {
	return len(ps)
}

// Swap swap
func (ps Params) Swap(i, j int) { ps[i], ps[j] = ps[j], ps[i] }

// Less less
func (ps Params) Less(i, j int) bool { return ps[i].Name < ps[j].Name }

func sortParamsByKey(ps Params) Params {
	sort.Sort(ps)
	return ps
}

// 对特定的 + 进行转义
func urlEscapeParams(ps Params) Params {
	for _, v := range ps {
		if str, ok := v.Value.(string); ok {
			v.Value = strings.Replace(url.QueryEscape(str), "+", "%20", -1)
		}
	}
	return ps
}

// 生成参数
func generateURLByParams(ps Params) string {
	var urls []string
	for _, v := range ps {
		urls = append(urls, fmt.Sprintf("%v=%v", v.Name, v.Value))
	}
	return strings.Join(urls, "&")
}

func genSignatureURL(httpMethod string, uri string, url string) string {
	return fmt.Sprintf("%v\n%v\n%v", httpMethod, uri, url)
}

func genSignature(signURL, secret string) string {
	key := []byte(secret)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(signURL))
	sEnc := b64.StdEncoding.EncodeToString(mac.Sum(nil))
	strings.Replace(sEnc, " ", "+", -1)
	fin := url.QueryEscape(sEnc)
	return fin
}
