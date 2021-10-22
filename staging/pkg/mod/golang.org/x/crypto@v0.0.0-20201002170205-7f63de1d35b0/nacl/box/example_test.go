package box_test

import (
	crypto_rand "crypto/rand" // Custom so it's clear which rand we're using.
	"fmt"
	"io"

	"golang.org/x/crypto/nacl/box"
)

func Example() {
	senderPublicKey, senderPrivateKey, err := box.GenerateKey(crypto_rand.Reader)
	if err != nil {
		panic(err)
	}

	recipientPublicKey, recipientPrivateKey, err := box.GenerateKey(crypto_rand.Reader)
	if err != nil {
		panic(err)
	}

	// You must use a different nonce for each message you encrypt with the
	// same key. Since the nonce here is 192 bits long, a random value
	// provides a sufficiently small probability of repeats.
	var nonce [24]byte
	if _, err := io.ReadFull(crypto_rand.Reader, nonce[:]); err != nil {
		panic(err)
	}

	msg := []byte("Alas, poor Yorick! I knew him, Horatio")
	// This encrypts msg and appends the result to the nonce.
	encrypted := box.Seal(nonce[:], msg, &nonce, recipientPublicKey, senderPrivateKey)

	// The recipient can decrypt the message using their private key and the
	// sender's public key. When you decrypt, you must use the same nonce you
	// used to encrypt the message. One way to achieve this is to store the
	// nonce alongside the encrypted message. Above, we stored the nonce in the
	// first 24 bytes of the encrypted text.
	var decryptNonce [24]byte
	copy(decryptNonce[:], encrypted[:24])
	decrypted, ok := box.Open(nil, encrypted[24:], &decryptNonce, senderPublicKey, recipientPrivateKey)
	if !ok {
		panic("decryption error")
	}
	fmt.Println(string(decrypted))
	// Output: Alas, poor Yorick! I knew him, Horatio
}

func Example_precompute() {
	senderPublicKey, senderPrivateKey, err := box.GenerateKey(crypto_rand.Reader)
	if err != nil {
		panic(err)
	}

	recipientPublicKey, recipientPrivateKey, err := box.GenerateKey(crypto_rand.Reader)
	if err != nil {
		panic(err)
	}

	// The shared key can be used to speed up processing when using the same
	// pair of keys repeatedly.
	sharedEncryptKey := new([32]byte)
	box.Precompute(sharedEncryptKey, recipientPublicKey, senderPrivateKey)

	// You must use a different nonce for each message you encrypt with the
	// same key. Since the nonce here is 192 bits long, a random value
	// provides a sufficiently small probability of repeats.
	var nonce [24]byte
	if _, err := io.ReadFull(crypto_rand.Reader, nonce[:]); err != nil {
		panic(err)
	}

	msg := []byte("A fellow of infinite jest, of most excellent fancy")
	// This encrypts msg and appends the result to the nonce.
	encrypted := box.SealAfterPrecomputation(nonce[:], msg, &nonce, sharedEncryptKey)

	// The shared key can be used to speed up processing when using the same
	// pair of keys repeatedly.
	var sharedDecryptKey [32]byte
	box.Precompute(&sharedDecryptKey, senderPublicKey, recipientPrivateKey)

	// The recipient can decrypt the message using the shared key. When you
	// decrypt, you must use the same nonce you used to encrypt the message.
	// One way to achieve this is to store the nonce alongside the encrypted
	// message. Above, we stored the nonce in the first 24 bytes of the
	// encrypted text.
	var decryptNonce [24]byte
	copy(decryptNonce[:], encrypted[:24])
	decrypted, ok := box.OpenAfterPrecomputation(nil, encrypted[24:], &decryptNonce, &sharedDecryptKey)
	if !ok {
		panic("decryption error")
	}
	fmt.Println(string(decrypted))
	// Output: A fellow of infinite jest, of most excellent fancy
}
