package pkcs11

// Vendor specific range for Ncipher network HSM.
const (
	NFCK_VENDOR_NCIPHER = 0xde436972
	CKA_NCIPHER         = NFCK_VENDOR_NCIPHER
	CKM_NCIPHER         = NFCK_VENDOR_NCIPHER
	CKK_NCIPHER         = NFCK_VENDOR_NCIPHER
)

// Vendor specific mechanisms for HMAC on Ncipher HSMs where Ncipher does not allow use of generic_secret keys.
const (
	CKM_NC_SHA_1_HMAC_KEY_GEN  = CKM_NCIPHER + 0x3  /* no params */
	CKM_NC_MD5_HMAC_KEY_GEN    = CKM_NCIPHER + 0x6  /* no params */
	CKM_NC_SHA224_HMAC_KEY_GEN = CKM_NCIPHER + 0x24 /* no params */
	CKM_NC_SHA256_HMAC_KEY_GEN = CKM_NCIPHER + 0x25 /* no params */
	CKM_NC_SHA384_HMAC_KEY_GEN = CKM_NCIPHER + 0x26 /* no params */
	CKM_NC_SHA512_HMAC_KEY_GEN = CKM_NCIPHER + 0x27 /* no params */
)

// Vendor specific range for Mozilla NSS.
const (
	NSSCK_VENDOR_NSS   = 0x4E534350
	CKO_NSS            = CKO_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKK_NSS            = CKK_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKC_NSS            = CKC_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKA_NSS            = CKA_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKA_TRUST          = CKA_NSS + 0x2000
	CKM_NSS            = CKM_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKR_NSS            = CKM_VENDOR_DEFINED | NSSCK_VENDOR_NSS
	CKT_VENDOR_DEFINED = 0x80000000
	CKT_NSS            = CKT_VENDOR_DEFINED | NSSCK_VENDOR_NSS
)

// Vendor specific values for Mozilla NSS.
const (
	CKO_NSS_CRL                               = CKO_NSS + 1
	CKO_NSS_SMIME                             = CKO_NSS + 2
	CKO_NSS_TRUST                             = CKO_NSS + 3
	CKO_NSS_BUILTIN_ROOT_LIST                 = CKO_NSS + 4
	CKO_NSS_NEWSLOT                           = CKO_NSS + 5
	CKO_NSS_DELSLOT                           = CKO_NSS + 6
	CKK_NSS_PKCS8                             = CKK_NSS + 1
	CKK_NSS_JPAKE_ROUND1                      = CKK_NSS + 2
	CKK_NSS_JPAKE_ROUND2                      = CKK_NSS + 3
	CKK_NSS_CHACHA20                          = CKK_NSS + 4
	CKA_NSS_URL                               = CKA_NSS + 1
	CKA_NSS_EMAIL                             = CKA_NSS + 2
	CKA_NSS_SMIME_INFO                        = CKA_NSS + 3
	CKA_NSS_SMIME_TIMESTAMP                   = CKA_NSS + 4
	CKA_NSS_PKCS8_SALT                        = CKA_NSS + 5
	CKA_NSS_PASSWORD_CHECK                    = CKA_NSS + 6
	CKA_NSS_EXPIRES                           = CKA_NSS + 7
	CKA_NSS_KRL                               = CKA_NSS + 8
	CKA_NSS_PQG_COUNTER                       = CKA_NSS + 20
	CKA_NSS_PQG_SEED                          = CKA_NSS + 21
	CKA_NSS_PQG_H                             = CKA_NSS + 22
	CKA_NSS_PQG_SEED_BITS                     = CKA_NSS + 23
	CKA_NSS_MODULE_SPEC                       = CKA_NSS + 24
	CKA_NSS_OVERRIDE_EXTENSIONS               = CKA_NSS + 25
	CKA_NSS_JPAKE_SIGNERID                    = CKA_NSS + 26
	CKA_NSS_JPAKE_PEERID                      = CKA_NSS + 27
	CKA_NSS_JPAKE_GX1                         = CKA_NSS + 28
	CKA_NSS_JPAKE_GX2                         = CKA_NSS + 29
	CKA_NSS_JPAKE_GX3                         = CKA_NSS + 30
	CKA_NSS_JPAKE_GX4                         = CKA_NSS + 31
	CKA_NSS_JPAKE_X2                          = CKA_NSS + 32
	CKA_NSS_JPAKE_X2S                         = CKA_NSS + 33
	CKA_NSS_MOZILLA_CA_POLICY                 = CKA_NSS + 34
	CKA_TRUST_DIGITAL_SIGNATURE               = CKA_TRUST + 1
	CKA_TRUST_NON_REPUDIATION                 = CKA_TRUST + 2
	CKA_TRUST_KEY_ENCIPHERMENT                = CKA_TRUST + 3
	CKA_TRUST_DATA_ENCIPHERMENT               = CKA_TRUST + 4
	CKA_TRUST_KEY_AGREEMENT                   = CKA_TRUST + 5
	CKA_TRUST_KEY_CERT_SIGN                   = CKA_TRUST + 6
	CKA_TRUST_CRL_SIGN                        = CKA_TRUST + 7
	CKA_TRUST_SERVER_AUTH                     = CKA_TRUST + 8
	CKA_TRUST_CLIENT_AUTH                     = CKA_TRUST + 9
	CKA_TRUST_CODE_SIGNING                    = CKA_TRUST + 10
	CKA_TRUST_EMAIL_PROTECTION                = CKA_TRUST + 11
	CKA_TRUST_IPSEC_END_SYSTEM                = CKA_TRUST + 12
	CKA_TRUST_IPSEC_TUNNEL                    = CKA_TRUST + 13
	CKA_TRUST_IPSEC_USER                      = CKA_TRUST + 14
	CKA_TRUST_TIME_STAMPING                   = CKA_TRUST + 15
	CKA_TRUST_STEP_UP_APPROVED                = CKA_TRUST + 16
	CKA_CERT_SHA1_HASH                        = CKA_TRUST + 100
	CKA_CERT_MD5_HASH                         = CKA_TRUST + 101
	CKM_NSS_AES_KEY_WRAP                      = CKM_NSS + 1
	CKM_NSS_AES_KEY_WRAP_PAD                  = CKM_NSS + 2
	CKM_NSS_HKDF_SHA1                         = CKM_NSS + 3
	CKM_NSS_HKDF_SHA256                       = CKM_NSS + 4
	CKM_NSS_HKDF_SHA384                       = CKM_NSS + 5
	CKM_NSS_HKDF_SHA512                       = CKM_NSS + 6
	CKM_NSS_JPAKE_ROUND1_SHA1                 = CKM_NSS + 7
	CKM_NSS_JPAKE_ROUND1_SHA256               = CKM_NSS + 8
	CKM_NSS_JPAKE_ROUND1_SHA384               = CKM_NSS + 9
	CKM_NSS_JPAKE_ROUND1_SHA512               = CKM_NSS + 10
	CKM_NSS_JPAKE_ROUND2_SHA1                 = CKM_NSS + 11
	CKM_NSS_JPAKE_ROUND2_SHA256               = CKM_NSS + 12
	CKM_NSS_JPAKE_ROUND2_SHA384               = CKM_NSS + 13
	CKM_NSS_JPAKE_ROUND2_SHA512               = CKM_NSS + 14
	CKM_NSS_JPAKE_FINAL_SHA1                  = CKM_NSS + 15
	CKM_NSS_JPAKE_FINAL_SHA256                = CKM_NSS + 16
	CKM_NSS_JPAKE_FINAL_SHA384                = CKM_NSS + 17
	CKM_NSS_JPAKE_FINAL_SHA512                = CKM_NSS + 18
	CKM_NSS_HMAC_CONSTANT_TIME                = CKM_NSS + 19
	CKM_NSS_SSL3_MAC_CONSTANT_TIME            = CKM_NSS + 20
	CKM_NSS_TLS_PRF_GENERAL_SHA256            = CKM_NSS + 21
	CKM_NSS_TLS_MASTER_KEY_DERIVE_SHA256      = CKM_NSS + 22
	CKM_NSS_TLS_KEY_AND_MAC_DERIVE_SHA256     = CKM_NSS + 23
	CKM_NSS_TLS_MASTER_KEY_DERIVE_DH_SHA256   = CKM_NSS + 24
	CKM_NSS_TLS_EXTENDED_MASTER_KEY_DERIVE    = CKM_NSS + 25
	CKM_NSS_TLS_EXTENDED_MASTER_KEY_DERIVE_DH = CKM_NSS + 26
	CKM_NSS_CHACHA20_KEY_GEN                  = CKM_NSS + 27
	CKM_NSS_CHACHA20_POLY1305                 = CKM_NSS + 28
	CKM_NSS_PKCS12_PBE_SHA224_HMAC_KEY_GEN    = CKM_NSS + 29
	CKM_NSS_PKCS12_PBE_SHA256_HMAC_KEY_GEN    = CKM_NSS + 30
	CKM_NSS_PKCS12_PBE_SHA384_HMAC_KEY_GEN    = CKM_NSS + 31
	CKM_NSS_PKCS12_PBE_SHA512_HMAC_KEY_GEN    = CKM_NSS + 32
	CKR_NSS_CERTDB_FAILED                     = CKR_NSS + 1
	CKR_NSS_KEYDB_FAILED                      = CKR_NSS + 2
	CKT_NSS_TRUSTED                           = CKT_NSS + 1
	CKT_NSS_TRUSTED_DELEGATOR                 = CKT_NSS + 2
	CKT_NSS_MUST_VERIFY_TRUST                 = CKT_NSS + 3
	CKT_NSS_NOT_TRUSTED                       = CKT_NSS + 10
	CKT_NSS_TRUST_UNKNOWN                     = CKT_NSS + 5
)
