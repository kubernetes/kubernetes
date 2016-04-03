package protosize

// We expect that Size field will have no suffix and ProtoSize will be present
var (
	_ = SizeMessage{}.Size
	_ = (&SizeMessage{}).GetSize

	_ = (&SizeMessage{}).ProtoSize
)
