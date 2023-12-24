// Copyright 2013 Miek Gieben. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs11

/*
#include <stdlib.h>
#include <string.h>
#include "pkcs11go.h"

static inline void putOAEPParams(CK_RSA_PKCS_OAEP_PARAMS_PTR params, CK_VOID_PTR pSourceData, CK_ULONG ulSourceDataLen)
{
	params->pSourceData = pSourceData;
	params->ulSourceDataLen = ulSourceDataLen;
}

static inline void putECDH1SharedParams(CK_ECDH1_DERIVE_PARAMS_PTR params, CK_VOID_PTR pSharedData, CK_ULONG ulSharedDataLen)
{
	params->pSharedData = pSharedData;
	params->ulSharedDataLen = ulSharedDataLen;
}

static inline void putECDH1PublicParams(CK_ECDH1_DERIVE_PARAMS_PTR params, CK_VOID_PTR pPublicData, CK_ULONG ulPublicDataLen)
{
	params->pPublicData = pPublicData;
	params->ulPublicDataLen = ulPublicDataLen;
}
*/
import "C"
import "unsafe"

// GCMParams represents the parameters for the AES-GCM mechanism.
type GCMParams struct {
	arena
	params  *C.CK_GCM_PARAMS
	iv      []byte
	aad     []byte
	tagSize int
}

// NewGCMParams returns a pointer to AES-GCM parameters that can be used with the CKM_AES_GCM mechanism.
// The Free() method must be called after the operation is complete.
//
// Note that some HSMs, like CloudHSM, will ignore the IV you pass in and write their
// own. As a result, to support all libraries, memory is not freed
// automatically, so that after the EncryptInit/Encrypt operation the HSM's IV
// can be read back out. It is up to the caller to ensure that Free() is called
// on the GCMParams object at an appropriate time, which is after
//
// Encrypt/Decrypt. As an example:
//
//    gcmParams := pkcs11.NewGCMParams(make([]byte, 12), nil, 128)
//    p.ctx.EncryptInit(session, []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_AES_GCM, gcmParams)},
//			aesObjHandle)
//    ct, _ := p.ctx.Encrypt(session, pt)
//    iv := gcmParams.IV()
//    gcmParams.Free()
//
func NewGCMParams(iv, aad []byte, tagSize int) *GCMParams {
	return &GCMParams{
		iv:      iv,
		aad:     aad,
		tagSize: tagSize,
	}
}

func cGCMParams(p *GCMParams) []byte {
	params := C.CK_GCM_PARAMS{
		ulTagBits: C.CK_ULONG(p.tagSize),
	}
	var arena arena
	if len(p.iv) > 0 {
		iv, ivLen := arena.Allocate(p.iv)
		params.pIv = C.CK_BYTE_PTR(iv)
		params.ulIvLen = ivLen
		params.ulIvBits = ivLen * 8
	}
	if len(p.aad) > 0 {
		aad, aadLen := arena.Allocate(p.aad)
		params.pAAD = C.CK_BYTE_PTR(aad)
		params.ulAADLen = aadLen
	}
	p.Free()
	p.arena = arena
	p.params = &params
	return C.GoBytes(unsafe.Pointer(&params), C.int(unsafe.Sizeof(params)))
}

// IV returns a copy of the actual IV used for the operation.
//
// Some HSMs may ignore the user-specified IV and write their own at the end of
// the encryption operation; this method allows you to retrieve it.
func (p *GCMParams) IV() []byte {
	if p == nil || p.params == nil {
		return nil
	}
	newIv := C.GoBytes(unsafe.Pointer(p.params.pIv), C.int(p.params.ulIvLen))
	iv := make([]byte, len(newIv))
	copy(iv, newIv)
	return iv
}

// Free deallocates the memory reserved for the HSM to write back the actual IV.
//
// This must be called after the entire operation is complete, i.e. after
// Encrypt or EncryptFinal. It is safe to call Free multiple times.
func (p *GCMParams) Free() {
	if p == nil || p.arena == nil {
		return
	}
	p.arena.Free()
	p.params = nil
	p.arena = nil
}

// NewPSSParams creates a CK_RSA_PKCS_PSS_PARAMS structure and returns it as a byte array for use with the CKM_RSA_PKCS_PSS mechanism.
func NewPSSParams(hashAlg, mgf, saltLength uint) []byte {
	p := C.CK_RSA_PKCS_PSS_PARAMS{
		hashAlg: C.CK_MECHANISM_TYPE(hashAlg),
		mgf:     C.CK_RSA_PKCS_MGF_TYPE(mgf),
		sLen:    C.CK_ULONG(saltLength),
	}
	return C.GoBytes(unsafe.Pointer(&p), C.int(unsafe.Sizeof(p)))
}

// OAEPParams can be passed to NewMechanism to implement CKM_RSA_PKCS_OAEP.
type OAEPParams struct {
	HashAlg    uint
	MGF        uint
	SourceType uint
	SourceData []byte
}

// NewOAEPParams creates a CK_RSA_PKCS_OAEP_PARAMS structure suitable for use with the CKM_RSA_PKCS_OAEP mechanism.
func NewOAEPParams(hashAlg, mgf, sourceType uint, sourceData []byte) *OAEPParams {
	return &OAEPParams{
		HashAlg:    hashAlg,
		MGF:        mgf,
		SourceType: sourceType,
		SourceData: sourceData,
	}
}

func cOAEPParams(p *OAEPParams, arena arena) ([]byte, arena) {
	params := C.CK_RSA_PKCS_OAEP_PARAMS{
		hashAlg: C.CK_MECHANISM_TYPE(p.HashAlg),
		mgf:     C.CK_RSA_PKCS_MGF_TYPE(p.MGF),
		source:  C.CK_RSA_PKCS_OAEP_SOURCE_TYPE(p.SourceType),
	}
	if len(p.SourceData) != 0 {
		buf, len := arena.Allocate(p.SourceData)
		// field is unaligned on windows so this has to call into C
		C.putOAEPParams(&params, buf, len)
	}
	return C.GoBytes(unsafe.Pointer(&params), C.int(unsafe.Sizeof(params))), arena
}

// ECDH1DeriveParams can be passed to NewMechanism to implement CK_ECDH1_DERIVE_PARAMS.
type ECDH1DeriveParams struct {
	KDF           uint
	SharedData    []byte
	PublicKeyData []byte
}

// NewECDH1DeriveParams creates a CK_ECDH1_DERIVE_PARAMS structure suitable for use with the CKM_ECDH1_DERIVE mechanism.
func NewECDH1DeriveParams(kdf uint, sharedData []byte, publicKeyData []byte) *ECDH1DeriveParams {
	return &ECDH1DeriveParams{
		KDF:           kdf,
		SharedData:    sharedData,
		PublicKeyData: publicKeyData,
	}
}

func cECDH1DeriveParams(p *ECDH1DeriveParams, arena arena) ([]byte, arena) {
	params := C.CK_ECDH1_DERIVE_PARAMS{
		kdf: C.CK_EC_KDF_TYPE(p.KDF),
	}

	// SharedData MUST be null if key derivation function (KDF) is CKD_NULL
	if len(p.SharedData) != 0 {
		sharedData, sharedDataLen := arena.Allocate(p.SharedData)
		C.putECDH1SharedParams(&params, sharedData, sharedDataLen)
	}

	publicKeyData, publicKeyDataLen := arena.Allocate(p.PublicKeyData)
	C.putECDH1PublicParams(&params, publicKeyData, publicKeyDataLen)

	return C.GoBytes(unsafe.Pointer(&params), C.int(unsafe.Sizeof(params))), arena
}
