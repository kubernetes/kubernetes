/* Copyright (c) OASIS Open 2016. All Rights Reserved./
 * /Distributed under the terms of the OASIS IPR Policy,
 * [http://www.oasis-open.org/policies-guidelines/ipr], AS-IS, WITHOUT ANY
 * IMPLIED OR EXPRESS WARRANTY; there is no warranty of MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE or NONINFRINGEMENT of the rights of others.
 */
        
/* Latest version of the specification:
 * http://docs.oasis-open.org/pkcs11/pkcs11-base/v2.40/pkcs11-base-v2.40.html
 */

/* This header file contains pretty much everything about all the
 * Cryptoki function prototypes.  Because this information is
 * used for more than just declaring function prototypes, the
 * order of the functions appearing herein is important, and
 * should not be altered.
 */

/* General-purpose */

/* C_Initialize initializes the Cryptoki library. */
CK_PKCS11_FUNCTION_INFO(C_Initialize)
#ifdef CK_NEED_ARG_LIST
(
  CK_VOID_PTR   pInitArgs  /* if this is not NULL_PTR, it gets
                            * cast to CK_C_INITIALIZE_ARGS_PTR
                            * and dereferenced
                            */
);
#endif


/* C_Finalize indicates that an application is done with the
 * Cryptoki library.
 */
CK_PKCS11_FUNCTION_INFO(C_Finalize)
#ifdef CK_NEED_ARG_LIST
(
  CK_VOID_PTR   pReserved  /* reserved.  Should be NULL_PTR */
);
#endif


/* C_GetInfo returns general information about Cryptoki. */
CK_PKCS11_FUNCTION_INFO(C_GetInfo)
#ifdef CK_NEED_ARG_LIST
(
  CK_INFO_PTR   pInfo  /* location that receives information */
);
#endif


/* C_GetFunctionList returns the function list. */
CK_PKCS11_FUNCTION_INFO(C_GetFunctionList)
#ifdef CK_NEED_ARG_LIST
(
  CK_FUNCTION_LIST_PTR_PTR ppFunctionList  /* receives pointer to
                                            * function list
                                            */
);
#endif



/* Slot and token management */

/* C_GetSlotList obtains a list of slots in the system. */
CK_PKCS11_FUNCTION_INFO(C_GetSlotList)
#ifdef CK_NEED_ARG_LIST
(
  CK_BBOOL       tokenPresent,  /* only slots with tokens */
  CK_SLOT_ID_PTR pSlotList,     /* receives array of slot IDs */
  CK_ULONG_PTR   pulCount       /* receives number of slots */
);
#endif


/* C_GetSlotInfo obtains information about a particular slot in
 * the system.
 */
CK_PKCS11_FUNCTION_INFO(C_GetSlotInfo)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID       slotID,  /* the ID of the slot */
  CK_SLOT_INFO_PTR pInfo    /* receives the slot information */
);
#endif


/* C_GetTokenInfo obtains information about a particular token
 * in the system.
 */
CK_PKCS11_FUNCTION_INFO(C_GetTokenInfo)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID        slotID,  /* ID of the token's slot */
  CK_TOKEN_INFO_PTR pInfo    /* receives the token information */
);
#endif


/* C_GetMechanismList obtains a list of mechanism types
 * supported by a token.
 */
CK_PKCS11_FUNCTION_INFO(C_GetMechanismList)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID            slotID,          /* ID of token's slot */
  CK_MECHANISM_TYPE_PTR pMechanismList,  /* gets mech. array */
  CK_ULONG_PTR          pulCount         /* gets # of mechs. */
);
#endif


/* C_GetMechanismInfo obtains information about a particular
 * mechanism possibly supported by a token.
 */
CK_PKCS11_FUNCTION_INFO(C_GetMechanismInfo)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID            slotID,  /* ID of the token's slot */
  CK_MECHANISM_TYPE     type,    /* type of mechanism */
  CK_MECHANISM_INFO_PTR pInfo    /* receives mechanism info */
);
#endif


/* C_InitToken initializes a token. */
CK_PKCS11_FUNCTION_INFO(C_InitToken)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID      slotID,    /* ID of the token's slot */
  CK_UTF8CHAR_PTR pPin,      /* the SO's initial PIN */
  CK_ULONG        ulPinLen,  /* length in bytes of the PIN */
  CK_UTF8CHAR_PTR pLabel     /* 32-byte token label (blank padded) */
);
#endif


/* C_InitPIN initializes the normal user's PIN. */
CK_PKCS11_FUNCTION_INFO(C_InitPIN)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_UTF8CHAR_PTR   pPin,      /* the normal user's PIN */
  CK_ULONG          ulPinLen   /* length in bytes of the PIN */
);
#endif


/* C_SetPIN modifies the PIN of the user who is logged in. */
CK_PKCS11_FUNCTION_INFO(C_SetPIN)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_UTF8CHAR_PTR   pOldPin,   /* the old PIN */
  CK_ULONG          ulOldLen,  /* length of the old PIN */
  CK_UTF8CHAR_PTR   pNewPin,   /* the new PIN */
  CK_ULONG          ulNewLen   /* length of the new PIN */
);
#endif



/* Session management */

/* C_OpenSession opens a session between an application and a
 * token.
 */
CK_PKCS11_FUNCTION_INFO(C_OpenSession)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID            slotID,        /* the slot's ID */
  CK_FLAGS              flags,         /* from CK_SESSION_INFO */
  CK_VOID_PTR           pApplication,  /* passed to callback */
  CK_NOTIFY             Notify,        /* callback function */
  CK_SESSION_HANDLE_PTR phSession      /* gets session handle */
);
#endif


/* C_CloseSession closes a session between an application and a
 * token.
 */
CK_PKCS11_FUNCTION_INFO(C_CloseSession)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession  /* the session's handle */
);
#endif


/* C_CloseAllSessions closes all sessions with a token. */
CK_PKCS11_FUNCTION_INFO(C_CloseAllSessions)
#ifdef CK_NEED_ARG_LIST
(
  CK_SLOT_ID     slotID  /* the token's slot */
);
#endif


/* C_GetSessionInfo obtains information about the session. */
CK_PKCS11_FUNCTION_INFO(C_GetSessionInfo)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE   hSession,  /* the session's handle */
  CK_SESSION_INFO_PTR pInfo      /* receives session info */
);
#endif


/* C_GetOperationState obtains the state of the cryptographic operation
 * in a session.
 */
CK_PKCS11_FUNCTION_INFO(C_GetOperationState)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,             /* session's handle */
  CK_BYTE_PTR       pOperationState,      /* gets state */
  CK_ULONG_PTR      pulOperationStateLen  /* gets state length */
);
#endif


/* C_SetOperationState restores the state of the cryptographic
 * operation in a session.
 */
CK_PKCS11_FUNCTION_INFO(C_SetOperationState)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR      pOperationState,      /* holds state */
  CK_ULONG         ulOperationStateLen,  /* holds state length */
  CK_OBJECT_HANDLE hEncryptionKey,       /* en/decryption key */
  CK_OBJECT_HANDLE hAuthenticationKey    /* sign/verify key */
);
#endif


/* C_Login logs a user into a token. */
CK_PKCS11_FUNCTION_INFO(C_Login)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_USER_TYPE      userType,  /* the user type */
  CK_UTF8CHAR_PTR   pPin,      /* the user's PIN */
  CK_ULONG          ulPinLen   /* the length of the PIN */
);
#endif


/* C_Logout logs a user out from a token. */
CK_PKCS11_FUNCTION_INFO(C_Logout)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession  /* the session's handle */
);
#endif



/* Object management */

/* C_CreateObject creates a new object. */
CK_PKCS11_FUNCTION_INFO(C_CreateObject)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_ATTRIBUTE_PTR  pTemplate,   /* the object's template */
  CK_ULONG          ulCount,     /* attributes in template */
  CK_OBJECT_HANDLE_PTR phObject  /* gets new object's handle. */
);
#endif


/* C_CopyObject copies an object, creating a new object for the
 * copy.
 */
CK_PKCS11_FUNCTION_INFO(C_CopyObject)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE    hSession,    /* the session's handle */
  CK_OBJECT_HANDLE     hObject,     /* the object's handle */
  CK_ATTRIBUTE_PTR     pTemplate,   /* template for new object */
  CK_ULONG             ulCount,     /* attributes in template */
  CK_OBJECT_HANDLE_PTR phNewObject  /* receives handle of copy */
);
#endif


/* C_DestroyObject destroys an object. */
CK_PKCS11_FUNCTION_INFO(C_DestroyObject)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_OBJECT_HANDLE  hObject    /* the object's handle */
);
#endif


/* C_GetObjectSize gets the size of an object in bytes. */
CK_PKCS11_FUNCTION_INFO(C_GetObjectSize)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_OBJECT_HANDLE  hObject,   /* the object's handle */
  CK_ULONG_PTR      pulSize    /* receives size of object */
);
#endif


/* C_GetAttributeValue obtains the value of one or more object
 * attributes.
 */
CK_PKCS11_FUNCTION_INFO(C_GetAttributeValue)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,   /* the session's handle */
  CK_OBJECT_HANDLE  hObject,    /* the object's handle */
  CK_ATTRIBUTE_PTR  pTemplate,  /* specifies attrs; gets vals */
  CK_ULONG          ulCount     /* attributes in template */
);
#endif


/* C_SetAttributeValue modifies the value of one or more object
 * attributes.
 */
CK_PKCS11_FUNCTION_INFO(C_SetAttributeValue)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,   /* the session's handle */
  CK_OBJECT_HANDLE  hObject,    /* the object's handle */
  CK_ATTRIBUTE_PTR  pTemplate,  /* specifies attrs and values */
  CK_ULONG          ulCount     /* attributes in template */
);
#endif


/* C_FindObjectsInit initializes a search for token and session
 * objects that match a template.
 */
CK_PKCS11_FUNCTION_INFO(C_FindObjectsInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,   /* the session's handle */
  CK_ATTRIBUTE_PTR  pTemplate,  /* attribute values to match */
  CK_ULONG          ulCount     /* attrs in search template */
);
#endif


/* C_FindObjects continues a search for token and session
 * objects that match a template, obtaining additional object
 * handles.
 */
CK_PKCS11_FUNCTION_INFO(C_FindObjects)
#ifdef CK_NEED_ARG_LIST
(
 CK_SESSION_HANDLE    hSession,          /* session's handle */
 CK_OBJECT_HANDLE_PTR phObject,          /* gets obj. handles */
 CK_ULONG             ulMaxObjectCount,  /* max handles to get */
 CK_ULONG_PTR         pulObjectCount     /* actual # returned */
);
#endif


/* C_FindObjectsFinal finishes a search for token and session
 * objects.
 */
CK_PKCS11_FUNCTION_INFO(C_FindObjectsFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession  /* the session's handle */
);
#endif



/* Encryption and decryption */

/* C_EncryptInit initializes an encryption operation. */
CK_PKCS11_FUNCTION_INFO(C_EncryptInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,  /* the encryption mechanism */
  CK_OBJECT_HANDLE  hKey         /* handle of encryption key */
);
#endif


/* C_Encrypt encrypts single-part data. */
CK_PKCS11_FUNCTION_INFO(C_Encrypt)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pData,               /* the plaintext data */
  CK_ULONG          ulDataLen,           /* bytes of plaintext */
  CK_BYTE_PTR       pEncryptedData,      /* gets ciphertext */
  CK_ULONG_PTR      pulEncryptedDataLen  /* gets c-text size */
);
#endif


/* C_EncryptUpdate continues a multiple-part encryption
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_EncryptUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,           /* session's handle */
  CK_BYTE_PTR       pPart,              /* the plaintext data */
  CK_ULONG          ulPartLen,          /* plaintext data len */
  CK_BYTE_PTR       pEncryptedPart,     /* gets ciphertext */
  CK_ULONG_PTR      pulEncryptedPartLen /* gets c-text size */
);
#endif


/* C_EncryptFinal finishes a multiple-part encryption
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_EncryptFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,                /* session handle */
  CK_BYTE_PTR       pLastEncryptedPart,      /* last c-text */
  CK_ULONG_PTR      pulLastEncryptedPartLen  /* gets last size */
);
#endif


/* C_DecryptInit initializes a decryption operation. */
CK_PKCS11_FUNCTION_INFO(C_DecryptInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,  /* the decryption mechanism */
  CK_OBJECT_HANDLE  hKey         /* handle of decryption key */
);
#endif


/* C_Decrypt decrypts encrypted data in a single part. */
CK_PKCS11_FUNCTION_INFO(C_Decrypt)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,           /* session's handle */
  CK_BYTE_PTR       pEncryptedData,     /* ciphertext */
  CK_ULONG          ulEncryptedDataLen, /* ciphertext length */
  CK_BYTE_PTR       pData,              /* gets plaintext */
  CK_ULONG_PTR      pulDataLen          /* gets p-text size */
);
#endif


/* C_DecryptUpdate continues a multiple-part decryption
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DecryptUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pEncryptedPart,      /* encrypted data */
  CK_ULONG          ulEncryptedPartLen,  /* input length */
  CK_BYTE_PTR       pPart,               /* gets plaintext */
  CK_ULONG_PTR      pulPartLen           /* p-text size */
);
#endif


/* C_DecryptFinal finishes a multiple-part decryption
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DecryptFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,       /* the session's handle */
  CK_BYTE_PTR       pLastPart,      /* gets plaintext */
  CK_ULONG_PTR      pulLastPartLen  /* p-text size */
);
#endif



/* Message digesting */

/* C_DigestInit initializes a message-digesting operation. */
CK_PKCS11_FUNCTION_INFO(C_DigestInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,   /* the session's handle */
  CK_MECHANISM_PTR  pMechanism  /* the digesting mechanism */
);
#endif


/* C_Digest digests data in a single part. */
CK_PKCS11_FUNCTION_INFO(C_Digest)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,     /* the session's handle */
  CK_BYTE_PTR       pData,        /* data to be digested */
  CK_ULONG          ulDataLen,    /* bytes of data to digest */
  CK_BYTE_PTR       pDigest,      /* gets the message digest */
  CK_ULONG_PTR      pulDigestLen  /* gets digest length */
);
#endif


/* C_DigestUpdate continues a multiple-part message-digesting
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DigestUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_BYTE_PTR       pPart,     /* data to be digested */
  CK_ULONG          ulPartLen  /* bytes of data to be digested */
);
#endif


/* C_DigestKey continues a multi-part message-digesting
 * operation, by digesting the value of a secret key as part of
 * the data already digested.
 */
CK_PKCS11_FUNCTION_INFO(C_DigestKey)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_OBJECT_HANDLE  hKey       /* secret key to digest */
);
#endif


/* C_DigestFinal finishes a multiple-part message-digesting
 * operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DigestFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,     /* the session's handle */
  CK_BYTE_PTR       pDigest,      /* gets the message digest */
  CK_ULONG_PTR      pulDigestLen  /* gets byte count of digest */
);
#endif



/* Signing and MACing */

/* C_SignInit initializes a signature (private key encryption)
 * operation, where the signature is (will be) an appendix to
 * the data, and plaintext cannot be recovered from the
 * signature.
 */
CK_PKCS11_FUNCTION_INFO(C_SignInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,  /* the signature mechanism */
  CK_OBJECT_HANDLE  hKey         /* handle of signature key */
);
#endif


/* C_Sign signs (encrypts with private key) data in a single
 * part, where the signature is (will be) an appendix to the
 * data, and plaintext cannot be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_Sign)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,        /* the session's handle */
  CK_BYTE_PTR       pData,           /* the data to sign */
  CK_ULONG          ulDataLen,       /* count of bytes to sign */
  CK_BYTE_PTR       pSignature,      /* gets the signature */
  CK_ULONG_PTR      pulSignatureLen  /* gets signature length */
);
#endif


/* C_SignUpdate continues a multiple-part signature operation,
 * where the signature is (will be) an appendix to the data,
 * and plaintext cannot be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_SignUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_BYTE_PTR       pPart,     /* the data to sign */
  CK_ULONG          ulPartLen  /* count of bytes to sign */
);
#endif


/* C_SignFinal finishes a multiple-part signature operation,
 * returning the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_SignFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,        /* the session's handle */
  CK_BYTE_PTR       pSignature,      /* gets the signature */
  CK_ULONG_PTR      pulSignatureLen  /* gets signature length */
);
#endif


/* C_SignRecoverInit initializes a signature operation, where
 * the data can be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_SignRecoverInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,   /* the session's handle */
  CK_MECHANISM_PTR  pMechanism, /* the signature mechanism */
  CK_OBJECT_HANDLE  hKey        /* handle of the signature key */
);
#endif


/* C_SignRecover signs data in a single operation, where the
 * data can be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_SignRecover)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,        /* the session's handle */
  CK_BYTE_PTR       pData,           /* the data to sign */
  CK_ULONG          ulDataLen,       /* count of bytes to sign */
  CK_BYTE_PTR       pSignature,      /* gets the signature */
  CK_ULONG_PTR      pulSignatureLen  /* gets signature length */
);
#endif



/* Verifying signatures and MACs */

/* C_VerifyInit initializes a verification operation, where the
 * signature is an appendix to the data, and plaintext cannot
 * cannot be recovered from the signature (e.g. DSA).
 */
CK_PKCS11_FUNCTION_INFO(C_VerifyInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,  /* the verification mechanism */
  CK_OBJECT_HANDLE  hKey         /* verification key */
);
#endif


/* C_Verify verifies a signature in a single-part operation,
 * where the signature is an appendix to the data, and plaintext
 * cannot be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_Verify)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,       /* the session's handle */
  CK_BYTE_PTR       pData,          /* signed data */
  CK_ULONG          ulDataLen,      /* length of signed data */
  CK_BYTE_PTR       pSignature,     /* signature */
  CK_ULONG          ulSignatureLen  /* signature length*/
);
#endif


/* C_VerifyUpdate continues a multiple-part verification
 * operation, where the signature is an appendix to the data,
 * and plaintext cannot be recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_VerifyUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_BYTE_PTR       pPart,     /* signed data */
  CK_ULONG          ulPartLen  /* length of signed data */
);
#endif


/* C_VerifyFinal finishes a multiple-part verification
 * operation, checking the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_VerifyFinal)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,       /* the session's handle */
  CK_BYTE_PTR       pSignature,     /* signature to verify */
  CK_ULONG          ulSignatureLen  /* signature length */
);
#endif


/* C_VerifyRecoverInit initializes a signature verification
 * operation, where the data is recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_VerifyRecoverInit)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,  /* the verification mechanism */
  CK_OBJECT_HANDLE  hKey         /* verification key */
);
#endif


/* C_VerifyRecover verifies a signature in a single-part
 * operation, where the data is recovered from the signature.
 */
CK_PKCS11_FUNCTION_INFO(C_VerifyRecover)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,        /* the session's handle */
  CK_BYTE_PTR       pSignature,      /* signature to verify */
  CK_ULONG          ulSignatureLen,  /* signature length */
  CK_BYTE_PTR       pData,           /* gets signed data */
  CK_ULONG_PTR      pulDataLen       /* gets signed data len */
);
#endif



/* Dual-function cryptographic operations */

/* C_DigestEncryptUpdate continues a multiple-part digesting
 * and encryption operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DigestEncryptUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pPart,               /* the plaintext data */
  CK_ULONG          ulPartLen,           /* plaintext length */
  CK_BYTE_PTR       pEncryptedPart,      /* gets ciphertext */
  CK_ULONG_PTR      pulEncryptedPartLen  /* gets c-text length */
);
#endif


/* C_DecryptDigestUpdate continues a multiple-part decryption and
 * digesting operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DecryptDigestUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pEncryptedPart,      /* ciphertext */
  CK_ULONG          ulEncryptedPartLen,  /* ciphertext length */
  CK_BYTE_PTR       pPart,               /* gets plaintext */
  CK_ULONG_PTR      pulPartLen           /* gets plaintext len */
);
#endif


/* C_SignEncryptUpdate continues a multiple-part signing and
 * encryption operation.
 */
CK_PKCS11_FUNCTION_INFO(C_SignEncryptUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pPart,               /* the plaintext data */
  CK_ULONG          ulPartLen,           /* plaintext length */
  CK_BYTE_PTR       pEncryptedPart,      /* gets ciphertext */
  CK_ULONG_PTR      pulEncryptedPartLen  /* gets c-text length */
);
#endif


/* C_DecryptVerifyUpdate continues a multiple-part decryption and
 * verify operation.
 */
CK_PKCS11_FUNCTION_INFO(C_DecryptVerifyUpdate)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,            /* session's handle */
  CK_BYTE_PTR       pEncryptedPart,      /* ciphertext */
  CK_ULONG          ulEncryptedPartLen,  /* ciphertext length */
  CK_BYTE_PTR       pPart,               /* gets plaintext */
  CK_ULONG_PTR      pulPartLen           /* gets p-text length */
);
#endif



/* Key management */

/* C_GenerateKey generates a secret key, creating a new key
 * object.
 */
CK_PKCS11_FUNCTION_INFO(C_GenerateKey)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE    hSession,    /* the session's handle */
  CK_MECHANISM_PTR     pMechanism,  /* key generation mech. */
  CK_ATTRIBUTE_PTR     pTemplate,   /* template for new key */
  CK_ULONG             ulCount,     /* # of attrs in template */
  CK_OBJECT_HANDLE_PTR phKey        /* gets handle of new key */
);
#endif


/* C_GenerateKeyPair generates a public-key/private-key pair,
 * creating new key objects.
 */
CK_PKCS11_FUNCTION_INFO(C_GenerateKeyPair)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE    hSession,                    /* session handle */
  CK_MECHANISM_PTR     pMechanism,                  /* key-gen mech. */
  CK_ATTRIBUTE_PTR     pPublicKeyTemplate,          /* template for pub. key */
  CK_ULONG             ulPublicKeyAttributeCount,   /* # pub. attrs. */
  CK_ATTRIBUTE_PTR     pPrivateKeyTemplate,         /* template for priv. key */
  CK_ULONG             ulPrivateKeyAttributeCount,  /* # priv.  attrs. */
  CK_OBJECT_HANDLE_PTR phPublicKey,                 /* gets pub. key handle */
  CK_OBJECT_HANDLE_PTR phPrivateKey                 /* gets priv. key handle */
);
#endif


/* C_WrapKey wraps (i.e., encrypts) a key. */
CK_PKCS11_FUNCTION_INFO(C_WrapKey)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,        /* the session's handle */
  CK_MECHANISM_PTR  pMechanism,      /* the wrapping mechanism */
  CK_OBJECT_HANDLE  hWrappingKey,    /* wrapping key */
  CK_OBJECT_HANDLE  hKey,            /* key to be wrapped */
  CK_BYTE_PTR       pWrappedKey,     /* gets wrapped key */
  CK_ULONG_PTR      pulWrappedKeyLen /* gets wrapped key size */
);
#endif


/* C_UnwrapKey unwraps (decrypts) a wrapped key, creating a new
 * key object.
 */
CK_PKCS11_FUNCTION_INFO(C_UnwrapKey)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE    hSession,          /* session's handle */
  CK_MECHANISM_PTR     pMechanism,        /* unwrapping mech. */
  CK_OBJECT_HANDLE     hUnwrappingKey,    /* unwrapping key */
  CK_BYTE_PTR          pWrappedKey,       /* the wrapped key */
  CK_ULONG             ulWrappedKeyLen,   /* wrapped key len */
  CK_ATTRIBUTE_PTR     pTemplate,         /* new key template */
  CK_ULONG             ulAttributeCount,  /* template length */
  CK_OBJECT_HANDLE_PTR phKey              /* gets new handle */
);
#endif


/* C_DeriveKey derives a key from a base key, creating a new key
 * object.
 */
CK_PKCS11_FUNCTION_INFO(C_DeriveKey)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE    hSession,          /* session's handle */
  CK_MECHANISM_PTR     pMechanism,        /* key deriv. mech. */
  CK_OBJECT_HANDLE     hBaseKey,          /* base key */
  CK_ATTRIBUTE_PTR     pTemplate,         /* new key template */
  CK_ULONG             ulAttributeCount,  /* template length */
  CK_OBJECT_HANDLE_PTR phKey              /* gets new handle */
);
#endif



/* Random number generation */

/* C_SeedRandom mixes additional seed material into the token's
 * random number generator.
 */
CK_PKCS11_FUNCTION_INFO(C_SeedRandom)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,  /* the session's handle */
  CK_BYTE_PTR       pSeed,     /* the seed material */
  CK_ULONG          ulSeedLen  /* length of seed material */
);
#endif


/* C_GenerateRandom generates random data. */
CK_PKCS11_FUNCTION_INFO(C_GenerateRandom)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession,    /* the session's handle */
  CK_BYTE_PTR       RandomData,  /* receives the random data */
  CK_ULONG          ulRandomLen  /* # of bytes to generate */
);
#endif



/* Parallel function management */

/* C_GetFunctionStatus is a legacy function; it obtains an
 * updated status of a function running in parallel with an
 * application.
 */
CK_PKCS11_FUNCTION_INFO(C_GetFunctionStatus)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession  /* the session's handle */
);
#endif


/* C_CancelFunction is a legacy function; it cancels a function
 * running in parallel.
 */
CK_PKCS11_FUNCTION_INFO(C_CancelFunction)
#ifdef CK_NEED_ARG_LIST
(
  CK_SESSION_HANDLE hSession  /* the session's handle */
);
#endif


/* C_WaitForSlotEvent waits for a slot event (token insertion,
 * removal, etc.) to occur.
 */
CK_PKCS11_FUNCTION_INFO(C_WaitForSlotEvent)
#ifdef CK_NEED_ARG_LIST
(
  CK_FLAGS flags,        /* blocking/nonblocking flag */
  CK_SLOT_ID_PTR pSlot,  /* location that receives the slot ID */
  CK_VOID_PTR pRserved   /* reserved.  Should be NULL_PTR */
);
#endif

