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
	sess := session.Must(session.NewSession())
	kmsClient := kms.New(sess)
	// Create the KeyProvider
	var matdesc s3crypto.MaterialDescription
	handler := s3crypto.NewKMSContextKeyGenerator(kmsClient, cmkID, matdesc)

	// Create an encryption and decryption client
	// We need to pass the session here so S3 can use it. In addition, any decryption that
	// occurs will use the KMS client.
	svc, err := s3crypto.NewEncryptionClientV2(sess, s3crypto.AESGCMContentCipherBuilderV2(handler))
	if err != nil {
		panic(err) // handle error
	}

	// Create a CryptoRegistry and register the algorithms you wish to use for decryption
	cr := s3crypto.NewCryptoRegistry()

	if err := s3crypto.RegisterAESGCMContentCipher(cr); err != nil {
		panic(err) // handle error
	}

	if err := s3crypto.RegisterKMSContextWrapWithAnyCMK(cr, kmsClient); err != nil {
		panic(err) // handle error
	}

	// Create a decryption client to decrypt artifacts
	svc, err := s3crypto.NewDecryptionClientV2(sess, cr)
	if err != nil {
		panic(err) // handle error
	}

Configuration of the S3 cryptography client

	sess := session.Must(session.NewSession())
	handler := s3crypto.NewKMSContextKeyGenerator(kms.New(sess), cmkID, s3crypto.MaterialDescription{})
	svc, err := s3crypto.NewEncryptionClientV2(sess, s3crypto.AESGCMContentCipherBuilderV2(handler), func (o *s3crypto.EncryptionClientOptions) {
		// Save instruction files to separate objects
		o.SaveStrategy = NewS3SaveStrategy(sess, "")

		// Change instruction file suffix to .example
		o.InstructionFileSuffix = ".example"

		// Set temp folder path
		o.TempFolderPath = "/path/to/tmp/folder/"

		// Any content less than the minimum file size will use memory
		// instead of writing the contents to a temp file.
		o.MinFileSize = int64(1024 * 1024 * 1024)
	})
	if err != nil {
		panic(err) // handle error
	}

Object Metadata SaveStrategy

The default SaveStrategy is to save metadata to an object's headers. An alternative SaveStrategy can be provided to the EncryptionClientV2.
For example, the S3SaveStrategy can be used to save the encryption metadata to a instruction file that is stored in S3
using the objects KeyName+InstructionFileSuffix. The InstructionFileSuffix defaults to .instruction. If using this strategy you will need to
configure the DecryptionClientV2 to use the matching S3LoadStrategy LoadStrategy in order to decrypt object using this save strategy.

Custom Key Wrappers and Custom Content Encryption Algorithms

Registration of custom key wrapping or content encryption algorithms not provided by AWS is allowed by the SDK, but
security and compatibility with custom types can not be guaranteed. For example if you want to support `CustomWrap`
key wrapping algorithm and `CustomCEK` content encryption algorithm. You can use the CryptoRegistry to register these types.

	cr := s3crypto.NewCryptoRegistry()

	// Register a custom key wrap algorithm to the CryptoRegistry
	if err := cr.AddWrap("CustomWrap", NewCustomWrapEntry); err != nil {
		panic(err) // handle error
	}

	// Register a custom content encryption algorithm to the CryptoRegistry
	if err := cr.AddCEK("CustomCEK", NewCustomCEKEntry); err != nil {
		panic(err) // handle error
	}

	svc, err := s3crypto.NewDecryptionClientV2(sess, cr)
	if err != nil {
		panic(err) // handle error
	}

We have now registered these new algorithms to the decryption client. When the client calls `GetObject` and sees
the wrap as `CustomWrap` then it'll use that wrap algorithm. This is also true for `CustomCEK`.

For encryption adding a custom content cipher builder and key handler will allow for encryption of custom
defined ciphers.

	// Our wrap algorithm, CustomWrap
	handler := NewCustomWrap(key, iv)
	// Our content cipher builder, NewCustomCEKContentBuilder
	svc := s3crypto.NewEncryptionClientV2(sess, NewCustomCEKContentBuilder(handler))

Maintenance Mode Notification for V1 Clients

The EncryptionClient and DecryptionClient are in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
Users of these clients should migrate to EncryptionClientV2 and DecryptionClientV2.

EncryptionClientV2 removes encryption support of the following features
	* AES/CBC (content cipher)
	* kms (key wrap algorithm)

Attempting to construct an EncryptionClientV2 with deprecated features will result in an error returned back to the
calling application during construction of the client.

Users of `AES/CBC` will need to migrate usage to `AES/GCM`.
Users of `kms` key provider will need to migrate `kms+context`.

DecryptionClientV2 client adds support for the `kms+context` key provider and maintains backwards comparability with
objects encrypted with the V1 EncryptionClient.

Migrating from V1 to V2 Clients

Examples of how to migrate usage of the V1 clients to the V2 equivalents have been documented as usage examples of
the NewEncryptionClientV2 and NewDecryptionClientV2 functions.

Please see the AWS SDK for Go Developer Guide for additional migration steps https://docs.aws.amazon.com/sdk-for-go/v1/developer-guide/s3-encryption-migration.html

*/
package s3crypto
