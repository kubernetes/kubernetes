// +build !go1.9

package dynamodbattribute

var fieldCache fieldCacher

type fieldCacher struct{}

func (c fieldCacher) Load(t interface{}) (*cachedFields, bool) {
	return nil, false
}

func (c fieldCacher) LoadOrStore(t interface{}, fs *cachedFields) (*cachedFields, bool) {
	return fs, false
}
