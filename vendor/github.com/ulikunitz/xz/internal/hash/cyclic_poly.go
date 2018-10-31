// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hash

// CyclicPoly provides a cyclic polynomial rolling hash.
type CyclicPoly struct {
	h uint64
	p []uint64
	i int
}

// ror rotates the unsigned 64-bit integer to right. The argument s must be
// less than 64.
func ror(x uint64, s uint) uint64 {
	return (x >> s) | (x << (64 - s))
}

// NewCyclicPoly creates a new instance of the CyclicPoly structure. The
// argument n gives the number of bytes for which a hash will be executed.
// This number must be positive; the method panics if this isn't the case.
func NewCyclicPoly(n int) *CyclicPoly {
	if n < 1 {
		panic("argument n must be positive")
	}
	return &CyclicPoly{p: make([]uint64, 0, n)}
}

// Len returns the length of the byte sequence for which a hash is generated.
func (r *CyclicPoly) Len() int {
	return cap(r.p)
}

// RollByte hashes the next byte and returns a hash value. The complete becomes
// available after at least Len() bytes have been hashed.
func (r *CyclicPoly) RollByte(x byte) uint64 {
	y := hash[x]
	if len(r.p) < cap(r.p) {
		r.h = ror(r.h, 1) ^ y
		r.p = append(r.p, y)
	} else {
		r.h ^= ror(r.p[r.i], uint(cap(r.p)-1))
		r.h = ror(r.h, 1) ^ y
		r.p[r.i] = y
		r.i = (r.i + 1) % cap(r.p)
	}
	return r.h
}

// Stores the hash for the individual bytes.
var hash = [256]uint64{
	0x2e4fc3f904065142, 0xc790984cfbc99527,
	0x879f95eb8c62f187, 0x3b61be86b5021ef2,
	0x65a896a04196f0a5, 0xc5b307b80470b59e,
	0xd3bff376a70df14b, 0xc332f04f0b3f1701,
	0x753b5f0e9abf3e0d, 0xb41538fdfe66ef53,
	0x1906a10c2c1c0208, 0xfb0c712a03421c0d,
	0x38be311a65c9552b, 0xfee7ee4ca6445c7e,
	0x71aadeded184f21e, 0xd73426fccda23b2d,
	0x29773fb5fb9600b5, 0xce410261cd32981a,
	0xfe2848b3c62dbc2d, 0x459eaaff6e43e11c,
	0xc13e35fc9c73a887, 0xf30ed5c201e76dbc,
	0xa5f10b3910482cea, 0x2945d59be02dfaad,
	0x06ee334ff70571b5, 0xbabf9d8070f44380,
	0xee3e2e9912ffd27c, 0x2a7118d1ea6b8ea7,
	0x26183cb9f7b1664c, 0xea71dac7da068f21,
	0xea92eca5bd1d0bb7, 0x415595862defcd75,
	0x248a386023c60648, 0x9cf021ab284b3c8a,
	0xfc9372df02870f6c, 0x2b92d693eeb3b3fc,
	0x73e799d139dc6975, 0x7b15ae312486363c,
	0xb70e5454a2239c80, 0x208e3fb31d3b2263,
	0x01f563cabb930f44, 0x2ac4533d2a3240d8,
	0x84231ed1064f6f7c, 0xa9f020977c2a6d19,
	0x213c227271c20122, 0x09fe8a9a0a03d07a,
	0x4236dc75bcaf910c, 0x460a8b2bead8f17e,
	0xd9b27be1aa07055f, 0xd202d5dc4b11c33e,
	0x70adb010543bea12, 0xcdae938f7ea6f579,
	0x3f3d870208672f4d, 0x8e6ccbce9d349536,
	0xe4c0871a389095ae, 0xf5f2a49152bca080,
	0x9a43f9b97269934e, 0xc17b3753cb6f475c,
	0xd56d941e8e206bd4, 0xac0a4f3e525eda00,
	0xa06d5a011912a550, 0x5537ed19537ad1df,
	0xa32fe713d611449d, 0x2a1d05b47c3b579f,
	0x991d02dbd30a2a52, 0x39e91e7e28f93eb0,
	0x40d06adb3e92c9ac, 0x9b9d3afde1c77c97,
	0x9a3f3f41c02c616f, 0x22ecd4ba00f60c44,
	0x0b63d5d801708420, 0x8f227ca8f37ffaec,
	0x0256278670887c24, 0x107e14877dbf540b,
	0x32c19f2786ac1c05, 0x1df5b12bb4bc9c61,
	0xc0cac129d0d4c4e2, 0x9fdb52ee9800b001,
	0x31f601d5d31c48c4, 0x72ff3c0928bcaec7,
	0xd99264421147eb03, 0x535a2d6d38aefcfe,
	0x6ba8b4454a916237, 0xfa39366eaae4719c,
	0x10f00fd7bbb24b6f, 0x5bd23185c76c84d4,
	0xb22c3d7e1b00d33f, 0x3efc20aa6bc830a8,
	0xd61c2503fe639144, 0x30ce625441eb92d3,
	0xe5d34cf359e93100, 0xa8e5aa13f2b9f7a5,
	0x5c2b8d851ca254a6, 0x68fb6c5e8b0d5fdf,
	0xc7ea4872c96b83ae, 0x6dd5d376f4392382,
	0x1be88681aaa9792f, 0xfef465ee1b6c10d9,
	0x1f98b65ed43fcb2e, 0x4d1ca11eb6e9a9c9,
	0x7808e902b3857d0b, 0x171c9c4ea4607972,
	0x58d66274850146df, 0x42b311c10d3981d1,
	0x647fa8c621c41a4c, 0xf472771c66ddfedc,
	0x338d27e3f847b46b, 0x6402ce3da97545ce,
	0x5162db616fc38638, 0x9c83be97bc22a50e,
	0x2d3d7478a78d5e72, 0xe621a9b938fd5397,
	0x9454614eb0f81c45, 0x395fb6e742ed39b6,
	0x77dd9179d06037bf, 0xc478d0fee4d2656d,
	0x35d9d6cb772007af, 0x83a56e92c883f0f6,
	0x27937453250c00a1, 0x27bd6ebc3a46a97d,
	0x9f543bf784342d51, 0xd158f38c48b0ed52,
	0x8dd8537c045f66b4, 0x846a57230226f6d5,
	0x6b13939e0c4e7cdf, 0xfca25425d8176758,
	0x92e5fc6cd52788e6, 0x9992e13d7a739170,
	0x518246f7a199e8ea, 0xf104c2a71b9979c7,
	0x86b3ffaabea4768f, 0x6388061cf3e351ad,
	0x09d9b5295de5bbb5, 0x38bf1638c2599e92,
	0x1d759846499e148d, 0x4c0ff015e5f96ef4,
	0xa41a94cfa270f565, 0x42d76f9cb2326c0b,
	0x0cf385dd3c9c23ba, 0x0508a6c7508d6e7a,
	0x337523aabbe6cf8d, 0x646bb14001d42b12,
	0xc178729d138adc74, 0xf900ef4491f24086,
	0xee1a90d334bb5ac4, 0x9755c92247301a50,
	0xb999bf7c4ff1b610, 0x6aeeb2f3b21e8fc9,
	0x0fa8084cf91ac6ff, 0x10d226cf136e6189,
	0xd302057a07d4fb21, 0x5f03800e20a0fcc3,
	0x80118d4ae46bd210, 0x58ab61a522843733,
	0x51edd575c5432a4b, 0x94ee6ff67f9197f7,
	0x765669e0e5e8157b, 0xa5347830737132f0,
	0x3ba485a69f01510c, 0x0b247d7b957a01c3,
	0x1b3d63449fd807dc, 0x0fdc4721c30ad743,
	0x8b535ed3829b2b14, 0xee41d0cad65d232c,
	0xe6a99ed97a6a982f, 0x65ac6194c202003d,
	0x692accf3a70573eb, 0xcc3c02c3e200d5af,
	0x0d419e8b325914a3, 0x320f160f42c25e40,
	0x00710d647a51fe7a, 0x3c947692330aed60,
	0x9288aa280d355a7a, 0xa1806a9b791d1696,
	0x5d60e38496763da1, 0x6c69e22e613fd0f4,
	0x977fc2a5aadffb17, 0xfb7bd063fc5a94ba,
	0x460c17992cbaece1, 0xf7822c5444d3297f,
	0x344a9790c69b74aa, 0xb80a42e6cae09dce,
	0x1b1361eaf2b1e757, 0xd84c1e758e236f01,
	0x88e0b7be347627cc, 0x45246009b7a99490,
	0x8011c6dd3fe50472, 0xc341d682bffb99d7,
	0x2511be93808e2d15, 0xd5bc13d7fd739840,
	0x2a3cd030679ae1ec, 0x8ad9898a4b9ee157,
	0x3245fef0a8eaf521, 0x3d6d8dbbb427d2b0,
	0x1ed146d8968b3981, 0x0c6a28bf7d45f3fc,
	0x4a1fd3dbcee3c561, 0x4210ff6a476bf67e,
	0xa559cce0d9199aac, 0xde39d47ef3723380,
	0xe5b69d848ce42e35, 0xefa24296f8e79f52,
	0x70190b59db9a5afc, 0x26f166cdb211e7bf,
	0x4deaf2df3c6b8ef5, 0xf171dbdd670f1017,
	0xb9059b05e9420d90, 0x2f0da855c9388754,
	0x611d5e9ab77949cc, 0x2912038ac01163f4,
	0x0231df50402b2fba, 0x45660fc4f3245f58,
	0xb91cc97c7c8dac50, 0xb72d2aafe4953427,
	0xfa6463f87e813d6b, 0x4515f7ee95d5c6a2,
	0x1310e1c1a48d21c3, 0xad48a7810cdd8544,
	0x4d5bdfefd5c9e631, 0xa43ed43f1fdcb7de,
	0xe70cfc8fe1ee9626, 0xef4711b0d8dda442,
	0xb80dd9bd4dab6c93, 0xa23be08d31ba4d93,
	0x9b37db9d0335a39c, 0x494b6f870f5cfebc,
	0x6d1b3c1149dda943, 0x372c943a518c1093,
	0xad27af45e77c09c4, 0x3b6f92b646044604,
	0xac2917909f5fcf4f, 0x2069a60e977e5557,
	0x353a469e71014de5, 0x24be356281f55c15,
	0x2b6d710ba8e9adea, 0x404ad1751c749c29,
	0xed7311bf23d7f185, 0xba4f6976b4acc43e,
	0x32d7198d2bc39000, 0xee667019014d6e01,
	0x494ef3e128d14c83, 0x1f95a152baecd6be,
	0x201648dff1f483a5, 0x68c28550c8384af6,
	0x5fc834a6824a7f48, 0x7cd06cb7365eaf28,
	0xd82bbd95e9b30909, 0x234f0d1694c53f6d,
	0xd2fb7f4a96d83f4a, 0xff0d5da83acac05e,
	0xf8f6b97f5585080a, 0x74236084be57b95b,
	0xa25e40c03bbc36ad, 0x6b6e5c14ce88465b,
	0x4378ffe93e1528c5, 0x94ca92a17118e2d2,
}
