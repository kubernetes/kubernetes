// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import "sort"

// TODO: this is brittle, as it depends on z.go's init() being called last.
// The current build tools all honor that files are passed in lexical order.
// However, we should consider using an init_channel,
// that each person doing init will write to.

func init() {
	if !useLookupRecognizedTypes {
		return
	}
	sort.Sort(uintptrSlice(recognizedRtids))
	sort.Sort(uintptrSlice(recognizedRtidPtrs))
	recognizedRtidOrPtrs = make([]uintptr, len(recognizedRtids)+len(recognizedRtidPtrs))
	copy(recognizedRtidOrPtrs, recognizedRtids)
	copy(recognizedRtidOrPtrs[len(recognizedRtids):], recognizedRtidPtrs)
	sort.Sort(uintptrSlice(recognizedRtidOrPtrs))
}
