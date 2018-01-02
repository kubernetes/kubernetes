package types

// Args is the type for funker args
type Args struct {
	// ChunkID is an unique number of the chunk
	ChunkID int `json:"chunk_id"`
	// Tests is the set of the strings that are passed as `-check.f` filters
	Tests []string `json:"tests"`
}

// Result is the type for funker result
type Result struct {
	// ChunkID corresponds to Args.ChunkID
	ChunkID int `json:"chunk_id"`
	// Code is the exit code
	Code   int    `json:"code"`
	RawLog string `json:"raw_log"`
}
