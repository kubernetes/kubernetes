package etcd

import (
	"math/rand"
)

func shuffleStringSlice(cards []string) []string {
	size := len(cards)
	//Do not need to copy if nothing changed
	if size <= 1 {
		return cards
	}
	shuffled := make([]string, size)
	index := rand.Perm(size)
	for i := range cards {
		shuffled[index[i]] = cards[i]
	}
	return shuffled
}
