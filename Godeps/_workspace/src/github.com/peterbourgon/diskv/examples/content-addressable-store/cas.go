package main

import (
	"crypto/md5"
	"fmt"
	"io"

	"github.com/peterbourgon/diskv"
)

const transformBlockSize = 2 // grouping of chars per directory depth

func blockTransform(s string) []string {
	var (
		sliceSize = len(s) / transformBlockSize
		pathSlice = make([]string, sliceSize)
	)
	for i := 0; i < sliceSize; i++ {
		from, to := i*transformBlockSize, (i*transformBlockSize)+transformBlockSize
		pathSlice[i] = s[from:to]
	}
	return pathSlice
}

func main() {
	d := diskv.New(diskv.Options{
		BasePath:     "data",
		Transform:    blockTransform,
		CacheSizeMax: 1024 * 1024, // 1MB
	})

	for _, valueStr := range []string{
		"I am the very model of a modern Major-General",
		"I've information vegetable, animal, and mineral",
		"I know the kings of England, and I quote the fights historical",
		"From Marathon to Waterloo, in order categorical",
		"I'm very well acquainted, too, with matters mathematical",
		"I understand equations, both the simple and quadratical",
		"About binomial theorem I'm teeming with a lot o' news",
		"With many cheerful facts about the square of the hypotenuse",
	} {
		d.Write(md5sum(valueStr), []byte(valueStr))
	}

	var keyCount int
	for key := range d.Keys(nil) {
		val, err := d.Read(key)
		if err != nil {
			panic(fmt.Sprintf("key %s had no value", key))
		}
		fmt.Printf("%s: %s\n", key, val)
		keyCount++
	}
	fmt.Printf("%d total keys\n", keyCount)

	// d.EraseAll() // leave it commented out to see how data is kept on disk
}

func md5sum(s string) string {
	h := md5.New()
	io.WriteString(h, s)
	return fmt.Sprintf("%x", h.Sum(nil))
}
