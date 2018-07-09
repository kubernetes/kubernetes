package csm

import (
	"strconv"
	"time"
)

type metricTime time.Time

func (t metricTime) MarshalJSON() ([]byte, error) {
	ns := time.Duration(time.Time(t).UnixNano())
	return []byte(strconv.FormatInt(int64(ns/time.Millisecond), 10)), nil
}

type metric struct {
	ClientID  *string     `json:"ClientId,omitempty"`
	API       *string     `json:"Api,omitempty"`
	Service   *string     `json:"Service,omitempty"`
	Timestamp *metricTime `json:"Timestamp,omitempty"`
	Type      *string     `json:"Type,omitempty"`
	Version   *int        `json:"Version,omitempty"`

	AttemptCount *int `json:"AttemptCount,omitempty"`
	Latency      *int `json:"Latency,omitempty"`

	Fqdn           *string `json:"Fqdn,omitempty"`
	UserAgent      *string `json:"UserAgent,omitempty"`
	AttemptLatency *int    `json:"AttemptLatency,omitempty"`

	SessionToken   *string `json:"SessionToken,omitempty"`
	Region         *string `json:"Region,omitempty"`
	AccessKey      *string `json:"AccessKey,omitempty"`
	HTTPStatusCode *int    `json:"HttpStatusCode,omitempty"`
	XAmzID2        *string `json:"XAmzId2,omitempty"`
	XAmzRequestID  *string `json:"XAmznRequestId,omitempty"`

	AWSException        *string `json:"AwsException,omitempty"`
	AWSExceptionMessage *string `json:"AwsExceptionMessage,omitempty"`
	SDKException        *string `json:"SdkException,omitempty"`
	SDKExceptionMessage *string `json:"SdkExceptionMessage,omitempty"`

	DestinationIP    *string `json:"DestinationIp,omitempty"`
	ConnectionReused *int    `json:"ConnectionReused,omitempty"`

	AcquireConnectionLatency *int `json:"AcquireConnectionLatency,omitempty"`
	ConnectLatency           *int `json:"ConnectLatency,omitempty"`
	RequestLatency           *int `json:"RequestLatency,omitempty"`
	DNSLatency               *int `json:"DnsLatency,omitempty"`
	TCPLatency               *int `json:"TcpLatency,omitempty"`
	SSLLatency               *int `json:"SslLatency,omitempty"`
}
