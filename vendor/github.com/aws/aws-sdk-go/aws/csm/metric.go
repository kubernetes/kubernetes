package csm

import (
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/aws"
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

	FinalHTTPStatusCode      *int    `json:"FinalHttpStatusCode,omitempty"`
	FinalAWSException        *string `json:"FinalAwsException,omitempty"`
	FinalAWSExceptionMessage *string `json:"FinalAwsExceptionMessage,omitempty"`
	FinalSDKException        *string `json:"FinalSdkException,omitempty"`
	FinalSDKExceptionMessage *string `json:"FinalSdkExceptionMessage,omitempty"`

	DestinationIP    *string `json:"DestinationIp,omitempty"`
	ConnectionReused *int    `json:"ConnectionReused,omitempty"`

	AcquireConnectionLatency *int `json:"AcquireConnectionLatency,omitempty"`
	ConnectLatency           *int `json:"ConnectLatency,omitempty"`
	RequestLatency           *int `json:"RequestLatency,omitempty"`
	DNSLatency               *int `json:"DnsLatency,omitempty"`
	TCPLatency               *int `json:"TcpLatency,omitempty"`
	SSLLatency               *int `json:"SslLatency,omitempty"`

	MaxRetriesExceeded *int `json:"MaxRetriesExceeded,omitempty"`
}

func (m *metric) TruncateFields() {
	m.ClientID = truncateString(m.ClientID, 255)
	m.UserAgent = truncateString(m.UserAgent, 256)

	m.AWSException = truncateString(m.AWSException, 128)
	m.AWSExceptionMessage = truncateString(m.AWSExceptionMessage, 512)

	m.SDKException = truncateString(m.SDKException, 128)
	m.SDKExceptionMessage = truncateString(m.SDKExceptionMessage, 512)

	m.FinalAWSException = truncateString(m.FinalAWSException, 128)
	m.FinalAWSExceptionMessage = truncateString(m.FinalAWSExceptionMessage, 512)

	m.FinalSDKException = truncateString(m.FinalSDKException, 128)
	m.FinalSDKExceptionMessage = truncateString(m.FinalSDKExceptionMessage, 512)
}

func truncateString(v *string, l int) *string {
	if v != nil && len(*v) > l {
		nv := (*v)[:l]
		return &nv
	}

	return v
}

func (m *metric) SetException(e metricException) {
	switch te := e.(type) {
	case awsException:
		m.AWSException = aws.String(te.exception)
		m.AWSExceptionMessage = aws.String(te.message)
	case sdkException:
		m.SDKException = aws.String(te.exception)
		m.SDKExceptionMessage = aws.String(te.message)
	}
}

func (m *metric) SetFinalException(e metricException) {
	switch te := e.(type) {
	case awsException:
		m.FinalAWSException = aws.String(te.exception)
		m.FinalAWSExceptionMessage = aws.String(te.message)
	case sdkException:
		m.FinalSDKException = aws.String(te.exception)
		m.FinalSDKExceptionMessage = aws.String(te.message)
	}
}
