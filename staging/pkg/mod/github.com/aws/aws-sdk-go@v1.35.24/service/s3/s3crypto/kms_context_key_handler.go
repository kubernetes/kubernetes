package s3crypto

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/kms/kmsiface"
)

const (
	// KMSContextWrap is a constant used during decryption to build a kms+context key handler
	KMSContextWrap      = "kms+context"
	kmsAWSCEKContextKey = "aws:" + cekAlgorithmHeader

	kmsReservedKeyConflictErrMsg = "conflict in reserved KMS Encryption Context key %s. This value is reserved for the S3 Encryption Client and cannot be set by the user"
	kmsMismatchCEKAlg            = "the content encryption algorithm used at encryption time does not match the algorithm stored for decryption time. The object may be altered or corrupted"
)

// NewKMSContextKeyGenerator builds a new kms+context key provider using the customer key ID and material
// description.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cmkID := "KMS Key ARN"
//	var matdesc s3crypto.MaterialDescription
//	handler := s3crypto.NewKMSContextKeyGenerator(kms.New(sess), cmkID, matdesc)
func NewKMSContextKeyGenerator(client kmsiface.KMSAPI, cmkID string, matdesc MaterialDescription) CipherDataGeneratorWithCEKAlg {
	return newKMSContextKeyHandler(client, cmkID, matdesc)
}

// RegisterKMSContextWrapWithCMK registers the kms+context wrapping algorithm to the given WrapRegistry. The wrapper
// will be configured to only call KMS Decrypt using the provided CMK.
//
// Example:
//	cr := s3crypto.NewCryptoRegistry()
//	if err := RegisterKMSContextWrapWithCMK(); err != nil {
//		panic(err) // handle error
//	}
func RegisterKMSContextWrapWithCMK(registry *CryptoRegistry, client kmsiface.KMSAPI, cmkID string) error {
	if registry == nil {
		return errNilCryptoRegistry
	}
	return registry.AddWrap(KMSContextWrap, newKMSContextWrapEntryWithCMK(client, cmkID))
}

// RegisterKMSContextWrapWithAnyCMK registers the kms+context wrapping algorithm to the given WrapRegistry. The wrapper
// will be configured to call KMS decrypt without providing a CMK.
//
// Example:
//	sess := session.Must(session.NewSession())
//	cr := s3crypto.NewCryptoRegistry()
//	if err := s3crypto.RegisterKMSContextWrapWithAnyCMK(cr, kms.New(sess)); err != nil {
//		panic(err) // handle error
//	}
func RegisterKMSContextWrapWithAnyCMK(registry *CryptoRegistry, client kmsiface.KMSAPI) error {
	if registry == nil {
		return errNilCryptoRegistry
	}
	return registry.AddWrap(KMSContextWrap, newKMSContextWrapEntryWithAnyCMK(client))
}

// newKMSContextWrapEntryWithCMK builds returns a new kms+context key provider and its decrypt handler.
// The returned handler will be configured to calls KMS Decrypt API without specifying a specific KMS CMK.
func newKMSContextWrapEntryWithCMK(kmsClient kmsiface.KMSAPI, cmkID string) WrapEntry {
	// These values are read only making them thread safe
	kp := &kmsContextKeyHandler{
		kms:   kmsClient,
		cmkID: &cmkID,
	}

	return kp.decryptHandler
}

// newKMSContextWrapEntryWithAnyCMK builds returns a new kms+context key provider and its decrypt handler.
// The returned handler will be configured to calls KMS Decrypt API without specifying a specific KMS CMK.
func newKMSContextWrapEntryWithAnyCMK(kmsClient kmsiface.KMSAPI) WrapEntry {
	// These values are read only making them thread safe
	kp := &kmsContextKeyHandler{
		kms: kmsClient,
	}

	return kp.decryptHandler
}

// kmsContextKeyHandler wraps the kmsKeyHandler to explicitly make this type incompatible with the v1 client
// by not exposing the old interface implementations.
type kmsContextKeyHandler struct {
	kms   kmsiface.KMSAPI
	cmkID *string

	CipherData
}

func (kp *kmsContextKeyHandler) isAWSFixture() bool {
	return true
}

func newKMSContextKeyHandler(client kmsiface.KMSAPI, cmkID string, matdesc MaterialDescription) *kmsContextKeyHandler {
	kp := &kmsContextKeyHandler{
		kms:   client,
		cmkID: &cmkID,
	}

	if matdesc == nil {
		matdesc = MaterialDescription{}
	}

	kp.CipherData.WrapAlgorithm = KMSContextWrap
	kp.CipherData.MaterialDescription = matdesc

	return kp
}

func (kp *kmsContextKeyHandler) GenerateCipherDataWithCEKAlg(ctx aws.Context, keySize int, ivSize int, cekAlgorithm string) (CipherData, error) {
	cd := kp.CipherData.Clone()

	if len(cekAlgorithm) == 0 {
		return CipherData{}, fmt.Errorf("cek algorithm identifier must not be empty")
	}

	if _, ok := cd.MaterialDescription[kmsAWSCEKContextKey]; ok {
		return CipherData{}, fmt.Errorf(kmsReservedKeyConflictErrMsg, kmsAWSCEKContextKey)
	}
	cd.MaterialDescription[kmsAWSCEKContextKey] = &cekAlgorithm

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

// decryptHandler initializes a KMS keyprovider with a material description. This
// is used with Decrypting kms content, due to the cmkID being in the material description.
func (kp kmsContextKeyHandler) decryptHandler(env Envelope) (CipherDataDecrypter, error) {
	if env.WrapAlg != KMSContextWrap {
		return nil, fmt.Errorf("%s value `%s` did not match the expected algorithm `%s` for this handler", cekAlgorithmHeader, env.WrapAlg, KMSContextWrap)
	}

	m := MaterialDescription{}
	err := m.decodeDescription([]byte(env.MatDesc))
	if err != nil {
		return nil, err
	}

	if v, ok := m[kmsAWSCEKContextKey]; !ok {
		return nil, fmt.Errorf("required key %v is missing from encryption context", kmsAWSCEKContextKey)
	} else if v == nil || *v != env.CEKAlg {
		return nil, fmt.Errorf(kmsMismatchCEKAlg)
	}

	kp.MaterialDescription = m
	kp.WrapAlgorithm = KMSContextWrap

	return &kp, nil
}

// DecryptKey makes a call to KMS to decrypt the key.
func (kp *kmsContextKeyHandler) DecryptKey(key []byte) ([]byte, error) {
	return kp.DecryptKeyWithContext(aws.BackgroundContext(), key)
}

// DecryptKeyWithContext makes a call to KMS to decrypt the key with request context.
func (kp *kmsContextKeyHandler) DecryptKeyWithContext(ctx aws.Context, key []byte) ([]byte, error) {
	out, err := kp.kms.DecryptWithContext(ctx,
		&kms.DecryptInput{
			KeyId:             kp.cmkID, // will be nil and not serialized if created with the AnyCMK constructor
			EncryptionContext: kp.MaterialDescription,
			CiphertextBlob:    key,
			GrantTokens:       []*string{},
		})
	if err != nil {
		return nil, err
	}
	return out.Plaintext, nil
}

var (
	_ CipherDataGeneratorWithCEKAlg  = (*kmsContextKeyHandler)(nil)
	_ CipherDataDecrypter            = (*kmsContextKeyHandler)(nil)
	_ CipherDataDecrypterWithContext = (*kmsContextKeyHandler)(nil)
	_ awsFixture                     = (*kmsContextKeyHandler)(nil)
)
