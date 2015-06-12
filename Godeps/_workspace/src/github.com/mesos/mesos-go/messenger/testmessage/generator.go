package testmessage

import (
	"math/rand"
)

func generateRandomString(length int) string {
	b := make([]byte, length)
	for i := range b {
		b[i] = byte(rand.Int())
	}
	return string(b)
}

// GenerateSmallMessage generates a small size message.
func GenerateSmallMessage() *SmallMessage {
	v := make([]string, 3)
	for i := range v {
		v[i] = generateRandomString(5)
	}
	return &SmallMessage{Values: v}
}

// GenerateMediumMessage generates a medium size message.
func GenerateMediumMessage() *MediumMessage {
	v := make([]string, 10)
	for i := range v {
		v[i] = generateRandomString(10)
	}
	return &MediumMessage{Values: v}
}

// GenerateBigMessage generates a big size message.
func GenerateBigMessage() *BigMessage {
	v := make([]string, 20)
	for i := range v {
		v[i] = generateRandomString(20)
	}
	return &BigMessage{Values: v}
}

// GenerateLargeMessage generates a large size message.
func GenerateLargeMessage() *LargeMessage {
	v := make([]string, 30)
	for i := range v {
		v[i] = generateRandomString(30)
	}
	return &LargeMessage{Values: v}
}
