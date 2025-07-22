package antlr

var BasePredictionContextEMPTY = &PredictionContext{
	cachedHash:  calculateEmptyHash(),
	pcType:      PredictionContextEmpty,
	returnState: BasePredictionContextEmptyReturnState,
}

// PredictionContextCache is Used to cache [PredictionContext] objects. It is used for the shared
// context cash associated with contexts in DFA states. This cache
// can be used for both lexers and parsers.
type PredictionContextCache struct {
	cache *JMap[*PredictionContext, *PredictionContext, Comparator[*PredictionContext]]
}

func NewPredictionContextCache() *PredictionContextCache {
	return &PredictionContextCache{
		cache: NewJMap[*PredictionContext, *PredictionContext, Comparator[*PredictionContext]](pContextEqInst, PredictionContextCacheCollection, "NewPredictionContextCache()"),
	}
}

// Add a context to the cache and return it. If the context already exists,
// return that one instead and do not add a new context to the cache.
// Protect shared cache from unsafe thread access.
func (p *PredictionContextCache) add(ctx *PredictionContext) *PredictionContext {
	if ctx.isEmpty() {
		return BasePredictionContextEMPTY
	}

	// Put will return the existing entry if it is present (note this is done via Equals, not whether it is
	// the same pointer), otherwise it will add the new entry and return that.
	//
	existing, present := p.cache.Get(ctx)
	if present {
		return existing
	}
	p.cache.Put(ctx, ctx)
	return ctx
}

func (p *PredictionContextCache) Get(ctx *PredictionContext) (*PredictionContext, bool) {
	pc, exists := p.cache.Get(ctx)
	return pc, exists
}

func (p *PredictionContextCache) length() int {
	return p.cache.Len()
}
