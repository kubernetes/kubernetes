package client

type ShardSpace struct {
	// required, must be unique within the database
	Name string `json:"name"`
	// required, a database has many shard spaces and a shard space belongs to a database
	Database string `json:"database"`
	// this is optional, if they don't set it, we'll set to /.*/
	Regex string `json:"regex"`
	// this is optional, if they don't set it, it will default to the storage.dir in the config
	RetentionPolicy   string `json:"retentionPolicy"`
	ShardDuration     string `json:"shardDuration"`
	ReplicationFactor uint32 `json:"replicationFactor"`
	Split             uint32 `json:"split"`
}
