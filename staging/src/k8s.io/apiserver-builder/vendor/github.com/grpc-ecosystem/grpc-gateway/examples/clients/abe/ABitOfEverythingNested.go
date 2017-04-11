package abe

import (
)

type ABitOfEverythingNested struct {
    Amount  int64  `json:"amount,omitempty"`
    Name  string  `json:"name,omitempty"`
    Ok  NestedDeepEnum  `json:"ok,omitempty"`
    
}
