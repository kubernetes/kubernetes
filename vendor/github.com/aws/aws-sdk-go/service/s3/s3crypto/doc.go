/*
Package s3crypto provides encryption to S3 using KMS and AES GCM.

Keyproviders are interfaces that handle masterkeys. Masterkeys are used to encrypt and decrypt the randomly
generated cipher keys. The SDK currently uses KMS to do this. A user does not need to provide a master key
since all that information is hidden in KMS.

Modes are interfaces that handle content encryption and decryption. It is an abstraction layer that instantiates
the ciphers. If content is being encrypted we generate the key and iv of the cipher. For decryption, we use the
metadata stored either on the object or an instruction file object to decrypt the contents.

Ciphers are interfaces that handle encryption and decryption of data. This may be key wrap ciphers or content
ciphers.

Creating an S3 cryptography client

	cmkID := "<some key ID>"
	sess := session.New()
	// Create the KeyProvider
	handler := s3crypto.NewKMSKeyGenerator(kms.New(sess), cmkID)

	// Create an encryption and decryption client
	// We need to pass the session here so S3 can use it. In addition, any decryption that
	// occurs will use the KMS client.
	svc := s3crypto.NewEncryptionClient(sess, s3crypto.AESGCMContentCipherBuilder(handler))
	svc := s3crypto.NewDecryptionClient(sess)

Configuration of the S3 cryptography client

	cfg := s3crypto.EncryptionConfig{
		// Save instruction files to separate objects
		SaveStrategy: NewS3SaveStrategy(session.New(), ""),
		// Change instruction file suffix to .example
		InstructionFileSuffix: ".example",
		// Set temp folder path
		TempFolderPath: "/path/to/tmp/folder/",
		// Any content less than the minimum file size will use memory
		// instead of writing the contents to a temp file.
		MinFileSize: int64(1024 * 1024 * 1024),
	}

The default SaveStrategy is to the object's header.

The InstructionFileSuffix defaults to .instruction. Careful here though, if you do this, be sure you know
what that suffix is in grabbing data.  All requests will look for fooKey.example instead of fooKey.instruction.
This suffix only affects gets and not puts. Put uses the keyprovider's suffix.

Registration of new wrap or cek algorithms are also supported by the SDK. Let's say we want to support `AES Wrap`
and `AES CTR`. Let's assume we have already defined the functionality.

	svc := s3crypto.NewDecryptionClient(sess)
	svc.WrapRegistry["AESWrap"] = NewAESWrap
	svc.CEKRegistry["AES/CTR/NoPadding"] = NewAESCTR

We have now registered these new algorithms to the decryption client. When the client calls `GetObject` and sees
the wrap as `AESWrap` then it'll use that wrap algorithm. This is also true for `AES/CTR/NoPadding`.

For encryption adding a custom content cipher builder and key handler will allow for encryption of custom
defined ciphers.

	// Our wrap algorithm, AESWrap
	handler := NewAESWrap(key, iv)
	// Our content cipher builder, AESCTRContentCipherBuilder
	svc := s3crypto.NewEncryptionClient(sess, NewAESCTRContentCipherBuilder(handler))
*/
package s3crypto
