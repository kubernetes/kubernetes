package quic

import (
	"container/list"
	"sync"

	"github.com/lucas-clemente/quic-go/internal/utils"
)

type singleOriginTokenStore struct {
	tokens []*ClientToken
	len    int
	p      int
}

func newSingleOriginTokenStore(size int) *singleOriginTokenStore {
	return &singleOriginTokenStore{tokens: make([]*ClientToken, size)}
}

func (s *singleOriginTokenStore) Add(token *ClientToken) {
	s.tokens[s.p] = token
	s.p = s.index(s.p + 1)
	s.len = utils.Min(s.len+1, len(s.tokens))
}

func (s *singleOriginTokenStore) Pop() *ClientToken {
	s.p = s.index(s.p - 1)
	token := s.tokens[s.p]
	s.tokens[s.p] = nil
	s.len = utils.Max(s.len-1, 0)
	return token
}

func (s *singleOriginTokenStore) Len() int {
	return s.len
}

func (s *singleOriginTokenStore) index(i int) int {
	mod := len(s.tokens)
	return (i + mod) % mod
}

type lruTokenStoreEntry struct {
	key   string
	cache *singleOriginTokenStore
}

type lruTokenStore struct {
	mutex sync.Mutex

	m                map[string]*list.Element
	q                *list.List
	capacity         int
	singleOriginSize int
}

var _ TokenStore = &lruTokenStore{}

// NewLRUTokenStore creates a new LRU cache for tokens received by the client.
// maxOrigins specifies how many origins this cache is saving tokens for.
// tokensPerOrigin specifies the maximum number of tokens per origin.
func NewLRUTokenStore(maxOrigins, tokensPerOrigin int) TokenStore {
	return &lruTokenStore{
		m:                make(map[string]*list.Element),
		q:                list.New(),
		capacity:         maxOrigins,
		singleOriginSize: tokensPerOrigin,
	}
}

func (s *lruTokenStore) Put(key string, token *ClientToken) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if el, ok := s.m[key]; ok {
		entry := el.Value.(*lruTokenStoreEntry)
		entry.cache.Add(token)
		s.q.MoveToFront(el)
		return
	}

	if s.q.Len() < s.capacity {
		entry := &lruTokenStoreEntry{
			key:   key,
			cache: newSingleOriginTokenStore(s.singleOriginSize),
		}
		entry.cache.Add(token)
		s.m[key] = s.q.PushFront(entry)
		return
	}

	elem := s.q.Back()
	entry := elem.Value.(*lruTokenStoreEntry)
	delete(s.m, entry.key)
	entry.key = key
	entry.cache = newSingleOriginTokenStore(s.singleOriginSize)
	entry.cache.Add(token)
	s.q.MoveToFront(elem)
	s.m[key] = elem
}

func (s *lruTokenStore) Pop(key string) *ClientToken {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	var token *ClientToken
	if el, ok := s.m[key]; ok {
		s.q.MoveToFront(el)
		cache := el.Value.(*lruTokenStoreEntry).cache
		token = cache.Pop()
		if cache.Len() == 0 {
			s.q.Remove(el)
			delete(s.m, key)
		}
	}
	return token
}
