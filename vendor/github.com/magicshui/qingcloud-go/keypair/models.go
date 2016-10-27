package keypair

import (
	"time"
)

type Keypair struct {
	Description   interface{} `json:"description"`
	EncryptMethod string      `json:"encrypt_method"`
	KeypairName   string      `json:"keypair_name"`
	InstanceIds   []string    `json:"instance_ids"`
	CreateTime    time.Time   `json:"create_time"`
	KeypairID     string      `json:"keypair_id"`
	PubKey        string      `json:"pub_key"`
}
