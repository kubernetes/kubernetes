package antlr

// A JStatRec is a record of a particular use of a [JStore], [JMap] or JPCMap] collection. Typically, it will be
// used to look for unused collections that wre allocated anyway, problems with hash bucket clashes, and anomalies
// such as huge numbers of Gets with no entries found GetNoEnt. You can refer to the CollectionAnomalies() function
// for ideas on what can be gleaned from these statistics about collections.
type JStatRec struct {
	Source           CollectionSource
	MaxSize          int
	CurSize          int
	Gets             int
	GetHits          int
	GetMisses        int
	GetHashConflicts int
	GetNoEnt         int
	Puts             int
	PutHits          int
	PutMisses        int
	PutHashConflicts int
	MaxSlotSize      int
	Description      string
	CreateStack      []byte
}
