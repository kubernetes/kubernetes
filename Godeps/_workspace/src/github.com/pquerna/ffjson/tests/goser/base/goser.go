/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package goser

import (
	"io"
	"net"
	"time"
)

type CacheStatus int32

const (
	CacheStatus_CACHESTATUS_UNKNOWN CacheStatus = 0
	CacheStatus_MISS                CacheStatus = 1
	CacheStatus_EXPIRED             CacheStatus = 2
	CacheStatus_HIT                 CacheStatus = 3
)

type HTTP_Protocol int32

const (
	HTTP_HTTP_PROTOCOL_UNKNOWN HTTP_Protocol = 0
	HTTP_HTTP10                HTTP_Protocol = 1
	HTTP_HTTP11                HTTP_Protocol = 2
)

type HTTP_Method int32

const (
	HTTP_METHOD_UNKNOWN HTTP_Method = 0
	HTTP_GET            HTTP_Method = 1
	HTTP_POST           HTTP_Method = 2
	HTTP_DELETE         HTTP_Method = 3
	HTTP_PUT            HTTP_Method = 4
	HTTP_HEAD           HTTP_Method = 5
	HTTP_PURGE          HTTP_Method = 6
	HTTP_OPTIONS        HTTP_Method = 7
	HTTP_PROPFIND       HTTP_Method = 8
	HTTP_MKCOL          HTTP_Method = 9
	HTTP_PATCH          HTTP_Method = 10
)

type Origin_Protocol int32

const (
	Origin_ORIGIN_PROTOCOL_UNKNOWN Origin_Protocol = 0
	Origin_HTTP                    Origin_Protocol = 1
	Origin_HTTPS                   Origin_Protocol = 2
)

type HTTP struct {
	Protocol         HTTP_Protocol `json:"protocol"`
	Status           uint32        `json:"status"`
	HostStatus       uint32        `json:"hostStatus"`
	UpStatus         uint32        `json:"upStatus"`
	Method           HTTP_Method   `json:"method"`
	ContentType      string        `json:"contentType"`
	UserAgent        string        `json:"userAgent"`
	Referer          string        `json:"referer"`
	RequestURI       string        `json:"requestURI"`
	XXX_unrecognized []byte        `json:"-"`
}

type Origin struct {
	Ip       IP              `json:"ip"`
	Port     uint32          `json:"port"`
	Hostname string          `json:"hostname"`
	Protocol Origin_Protocol `json:"protocol"`
}

type ZonePlan int32

const (
	ZonePlan_ZONEPLAN_UNKNOWN ZonePlan = 0
	ZonePlan_FREE             ZonePlan = 1
	ZonePlan_PRO              ZonePlan = 2
	ZonePlan_BIZ              ZonePlan = 3
	ZonePlan_ENT              ZonePlan = 4
)

type Country int32

const (
	Country_UNKNOWN Country = 0
	Country_US      Country = 238
)

type Log struct {
	Timestamp        int64       `json:"timestamp"`
	ZoneId           uint32      `json:"zoneId"`
	ZonePlan         ZonePlan    `json:"zonePlan"`
	Http             HTTP        `json:"http"`
	Origin           Origin      `json:"origin"`
	Country          Country     `json:"country"`
	CacheStatus      CacheStatus `json:"cacheStatus"`
	ServerIp         IP          `json:"serverIp"`
	ServerName       string      `json:"serverName"`
	RemoteIp         IP          `json:"remoteIp"`
	BytesDlv         uint64      `json:"bytesDlv"`
	RayId            string      `json:"rayId"`
	XXX_unrecognized []byte      `json:"-"`
}

type IP net.IP

func (ip IP) MarshalJSON() ([]byte, error) {
	return []byte("\"" + net.IP(ip).String() + "\""), nil
}

func (ip *IP) UnmarshalJSON(data []byte) error {
	if len(data) < 2 {
		return io.ErrShortBuffer
	}
	*ip = IP(net.ParseIP(string(data[1 : len(data)-1])).To4())
	return nil
}

const userAgent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36"

func NewLog(record *Log) {
	record.Timestamp = time.Now().UnixNano()
	record.ZoneId = 123456
	record.ZonePlan = ZonePlan_FREE

	record.Http = HTTP{
		Protocol:    HTTP_HTTP11,
		Status:      200,
		HostStatus:  503,
		UpStatus:    520,
		Method:      HTTP_GET,
		ContentType: "text/html",
		UserAgent:   userAgent,
		Referer:     "https://www.cloudflare.com/",
		RequestURI:  "/cdn-cgi/trace",
	}

	record.Origin = Origin{
		Ip:       IP(net.IPv4(1, 2, 3, 4).To4()),
		Port:     8080,
		Hostname: "www.example.com",
		Protocol: Origin_HTTPS,
	}

	record.Country = Country_US
	record.CacheStatus = CacheStatus_HIT
	record.ServerIp = IP(net.IPv4(192, 168, 1, 1).To4())
	record.ServerName = "metal.cloudflare.com"
	record.RemoteIp = IP(net.IPv4(10, 1, 2, 3).To4())
	record.BytesDlv = 123456
	record.RayId = "10c73629cce30078-LAX"
}
