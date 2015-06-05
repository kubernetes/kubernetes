package tff

type Foo struct {
	Blah int
}

type Record struct {
	Timestamp int64 `json:"id,omitempty"`
	OriginId  uint32
	Bar       Foo
	Method    string `json:"meth"`
	ReqId     string
	ServerIp  string
	RemoteIp  string
	BytesSent uint64
}
