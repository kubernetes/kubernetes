package roaring

import "container/heap"

/////////////
// The priorityQueue is used to keep Bitmaps sorted.
////////////

type item struct {
	value *Bitmap
	index int
}

type priorityQueue []*item

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].value.GetSizeInBytes() < pq[j].value.GetSizeInBytes()
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

func (pq *priorityQueue) update(item *item, value *Bitmap) {
	item.value = value
	heap.Fix(pq, item.index)
}

/////////////
// The containerPriorityQueue is used to keep the containers of various Bitmaps sorted.
////////////

type containeritem struct {
	value    *Bitmap
	keyindex int
	index    int
}

type containerPriorityQueue []*containeritem

func (pq containerPriorityQueue) Len() int { return len(pq) }

func (pq containerPriorityQueue) Less(i, j int) bool {
	k1 := pq[i].value.highlowcontainer.getKeyAtIndex(pq[i].keyindex)
	k2 := pq[j].value.highlowcontainer.getKeyAtIndex(pq[j].keyindex)
	if k1 != k2 {
		return k1 < k2
	}
	c1 := pq[i].value.highlowcontainer.getContainerAtIndex(pq[i].keyindex)
	c2 := pq[j].value.highlowcontainer.getContainerAtIndex(pq[j].keyindex)

	return c1.getCardinality() > c2.getCardinality()
}

func (pq containerPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *containerPriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*containeritem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *containerPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

//func (pq *containerPriorityQueue) update(item *containeritem, value *Bitmap, keyindex int) {
//	item.value = value
//	item.keyindex = keyindex
//	heap.Fix(pq, item.index)
//}
