package s3crypto

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/kms/kmsiface"
)

const (
	// KMSWrap is a constant used during decryption to build a KMS key handler.
	KMSWrap = "kms"
)

// kmsKeyHandler will make calls to KMS to get the masterkey
type kmsKeyHandler struct {
	kms   kmsiface.KMSAPI
	cmkID *string

	// useProvidedCMK is toggled when using `kms` key wrapper with V2 client
	useProvidedCMK bool

	CipherData
}

// NewKMSKeyGenerator builds a new KMS key provider using the customer key ID and material
// description.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cmkID := "arn to key"
//	matdesc := s3crypto.MaterialDescription{}
//	handler := s3crypto.NewKMSKeyGenerator(kms.New(sess), cmkID)
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func NewKMSKeyGenerator(kmsClient kmsiface.KMSAPI, cmkID string) CipherDataGenerator {
	return NewKMSKeyGeneratorWithMatDesc(kmsClient, cmkID, MaterialDescription{})
}

func newKMSKeyHandler(client kmsiface.KMSAPI, cmkID string, matdesc MaterialDescription) *kmsKeyHandler {
	// These values are read only making them thread safe
	kp := &kmsKeyHandler{
		kms:   client,
		cmkID: &cmkID,
	}

	if matdesc == nil {
		matdesc = MaterialDescription{}
	}

	matdesc["kms_cmk_id"] = &cmkID

	kp.CipherData.WrapAlgorithm = KMSWrap
	kp.CipherData.MaterialDescription = matdesc

	return kp
}

// NewKMSKeyGeneratorWithMatDesc builds a new KMS key provider using the customer key ID and material
// description.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cmkID := "arn to key"
//	matdesc := s3crypto.MaterialDescription{}
//	handler := s3crypto.NewKMSKeyGeneratorWithMatDesc(kms.New(sess), cmkID, matdesc)
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func NewKMSKeyGeneratorWithMatDesc(kmsClient kmsiface.KMSAPI, cmkID string, matdesc MaterialDescription) CipherDataGenerator {
	return newKMSKeyHandler(kmsClient, cmkID, matdesc)
}

// NewKMSWrapEntry builds returns a new KMS key provider and its decrypt handler.
//
// Example:
//	sess := session.Must(session.NewSession())
//	customKMSClient := kms.New(sess)
//	decryptHandler := s3crypto.NewKMSWrapEntry(customKMSClient)
//
//	svc := s3crypto.NewDecryptionClient(sess, func(svc *s3crypto.DecryptionClient) {
//		svc.WrapRegistry[s3crypto.KMSWrap] = decryptHandler
//	}))
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func NewKMSWrapEntry(kmsClient kmsiface.KMSAPI) WrapEntry {
	kp := newKMSWrapEntry(kmsClient)
	return kp.decryptHandler
}

// RegisterKMSWrapWithCMK registers the `kms` wrapping algorithm to the given WrapRegistry. The wrapper will be
// configured to call KMS Decrypt with the provided CMK.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cr := s3crypto.NewCryptoRegistry()
//	if err := s3crypto.RegisterKMSWrapWithCMK(cr, kms.New(sess), "cmkId"); err != nil {
//		panic(err) // handle error
//	}
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func RegisterKMSWrapWithCMK(registry *CryptoRegistry, client kmsiface.KMSAPI, cmkID string) error {
	if registry == nil {
		return errNilCryptoRegistry
	}
	return registry.AddWrap(KMSWrap, newKMSWrapEntryWithCMK(client, cmkID))
}

// RegisterKMSWrapWithAnyCMK registers the `kms` wrapping algorithm to the given WrapRegistry. The wrapper will be
// configured to call KMS Decrypt without providing a CMK.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cr := s3crypto.NewCryptoRegistry()
//	if err := s3crypto.RegisterKMSWrapWithAnyCMK(cr, kms.New(sess)); err != nil {
//		panic(err) // handle error
//	}
//
// deprecated: This feature is in maintenance mode, no new updates will be released. Please see https://docs.aws.amazon.com/general/latest/gr/aws_sdk_cryptography.html for more information.
func RegisterKMSWrapWithAnyCMK(registry *CryptoRegistry, client kmsiface.KMSAPI) error {
	if registry == nil {
		return errNilCryptoRegistry
	}
	return registry.AddWrap(KMSWrap, NewKMSWrapEntry(client))
}

// newKMSWrapEntryWithCMK builds returns a new KMS key provider and its decrypt handler. The wrap entry will be configured
// to only attempt to decrypt the data key using the provided CMK.
func newKMSWrapEntryWithCMK(kmsClient kmsiface.KMSAPI, cmkID string) WrapEntry {
	kp := newKMSWrapEntry(kmsClient)
	kp.useProvidedCMK = true
	kp.cmkID = &cmkID
	return kp.decryptHandler
}

func newKMSWrapEntry(kmsClient kmsiface.KMSAPI) *kmsKeyHandler {
	// These values are read only making them thread safe
	kp := &kmsKeyHandler{
		kms: kmsClient,
	}

	return kp
}

// decryptHandler initializes a KMS keyprovider with a material description. This
// is used with Decrypting kms content, due to the cmkID being in the material description.
func (kp kmsKeyHandler) decryptHandler(env Envelope) (CipherDataDecrypter, error) {
	m := MaterialDescription{}
	err := m.decodeDescription([]byte(env.MatDesc))
	if err != nil {
		return nil, err
	}

	_, ok := m["kms_cmk_id"]
	if !ok {
		return nil, awserr.New("MissingCMKIDError", "Material description is missing CMK ID", nil)
	}

	kp.CipherData.MaterialDescription = m
	kp.WrapAlgorithm = KMSWrap

	return &kp, nil
}

// DecryptKey makes a call to KMS to decrypt the key.
func (kp *kmsKeyHandler) DecryptKey(key []byte) ([]byte, error) {
	return kp.DecryptKeyWithContext(aws.BackgroundContext(), key)
}

// DecryptKeyWithContext makes a call to KMS to decrypt the key with request context.
func (kp *kmsKeyHandler) DecryptKeyWithContext(ctx aws.Context, key []byte) ([]byte, error) {
	in := &kms.DecryptInput{
		EncryptionContext: kp.MaterialDescription,
		CiphertextBlob:    key,
		GrantTokens:       []*string{},
	}

	// useProvidedCMK will be true if a constructor was used with the new V2 client
	if kp.useProvidedCMK {
		in.KeyId = kp.cmkID
	}

	out, err := kp.kms.DecryptWithContext(ctx, in)
	if err != nil {
		return nil, err
	}
	return out.Plaintext, nil
}

// GenerateCipherData makes a call to KMS to generate a data key, Upon making
// the call, it also sets the encrypted key.
func (kp *kmsKeyHandler) GenerateCipherData(keySize, ivSize int) (CipherData, error) {
	return kp.GenerateCipherDataWithContext(aws.BackgroundContext(), keySize, ivSize)
}

// GenerateCipherDataWithContext makes a call to KMS to generate a data key,
// Upon making the call, it also sets the encrypted key.
func (kp *kmsKeyHandler) GenerateCipherDataWithContext(ctx aws.Context, keySize, ivSize int) (CipherData, error) {
	cd := kp.CipherData.Clone()

	out, err := kp.kms.GenerateDataKeyWithContext(ctx,
		&kms.GenerateDataKeyInput{
			EncryptionContext: cd.MaterialDescription,
			KeyId:             kp.cmkID,
			KeySpec:           aws.String("AES_256"),
		})
	if err != nil {
		return CipherData{}, err
	}

	iv, err := generateBytes(ivSize)
	if err != nil {
		return CipherData{}, err
	}

	cd.Key = out.Plaintext
	cd.IV = iv
	cd.EncryptedKey = out.CiphertextBlob

	return cd, nil
}

func (kp kmsKeyHandler) isAWSFixture() bool {
	return true
}

var (
	_ CipherDataGenerator            = (*kmsKeyHandler)(nil)
	_ CipherDataGeneratorWithContext = (*kmsKeyHandler)(nil)
	_ CipherDataDecrypter            = (*kmsKeyHandler)(nil)
	_ CipherDataDecrypterWithContext = (*kmsKeyHandler)(nil)
	_ awsFixture                     = (*kmsKeyHandler)(nil)
)
