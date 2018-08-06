// Package ubiquity contains the ubiquity scoring logic for CFSSL bundling.
package ubiquity

// Ubiquity is addressed as selecting the chains that are most likely being accepted for different client systems.
// To select, we decide to do multi-round filtering from different ranking perpectives.
import (
	"crypto/x509"
)

// RankingFunc returns the relative rank between chain1 and chain2.
// Return value:
//	positive integer if rank(chain1) > rank(chain2),
//	negative integer if rank(chain1) < rank(chain2),
//	0 if rank(chain1) == (chain2).
type RankingFunc func(chain1, chain2 []*x509.Certificate) int

// Filter filters out the chains with highest rank according to the ranking function f.
func Filter(chains [][]*x509.Certificate, f RankingFunc) [][]*x509.Certificate {
	// If there are no chain or only 1 chain, we are done.
	if len(chains) <= 1 {
		return chains
	}

	bestChain := chains[0]
	var candidateChains [][]*x509.Certificate
	for _, chain := range chains {
		r := f(bestChain, chain)
		if r < 0 {
			bestChain = chain
			candidateChains = [][]*x509.Certificate{chain}
		} else if r == 0 {
			candidateChains = append(candidateChains, chain)
		}
	}
	return candidateChains
}
