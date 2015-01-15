package gss

/*
#cgo pkg-config: krb5-gssapi
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <gssapi/gssapi.h>
#include <gssapi/gssapi_generic.h>
#include <gssapi/gssapi_krb5.h>
#include <gssapi/gssapi_ext.h>

static gss_OID_desc nth_oid_in_set(gss_OID_set_desc *oset, unsigned int n)
{
	return oset->elements[n];
}
static gss_buffer_desc nth_buffer_in_set(gss_buffer_set_desc *bset, unsigned int n)
{
	return bset->elements[n];
}
static gss_key_value_element_desc *alloc_n_kvset_elems(unsigned int n)
{
	return malloc(sizeof(gss_key_value_element_desc));
}
static void kv_set(gss_key_value_set_desc *kvset, int i, char *key, char *value)
{
	kvset->elements[i].key = key;
	kvset->elements[i].value = value;
}
static void free_kv_set(gss_key_value_set_desc kvset)
{
	unsigned i;
	for (i = 0; i < kvset.count; i++) {
		free((char *) kvset.elements[i].key);
		free((char *) kvset.elements[i].value);
	}
	free(kvset.elements);
}
static void free_gss_buffer(gss_buffer_desc buffer)
{
	free(buffer.value);
}
static void free_oid(gss_OID oid)
{
	if (oid != NULL) {
		free(oid->elements);
		free(oid);
	}
}
static void free_oid_set(gss_OID_set buffer)
{
	OM_uint32 minor;
	gss_release_oid_set(&minor, &buffer);
}
static void *
copyOid(unsigned char *bytes, int len)
{
	void *ret;

	if (len < 0) {
		return NULL;
	}
	ret = malloc(len);
	if (ret != NULL) {
		memcpy(ret, bytes, len);
	}
	return ret;
}
*/
import "C"
import "unsafe"
import "encoding/asn1"
import "fmt"
import "bytes"

const (
	C_DCE_STYLE           = C.GSS_C_DCE_STYLE
	C_IDENTIFY_FLAG       = C.GSS_C_IDENTIFY_FLAG
	C_EXTENDED_ERROR_FLAG = C.GSS_C_EXTENDED_ERROR_FLAG

	// C_NO_CRED_STORE

	// credUsage values passed to AcquireCred(), AddCred(), StoreCred() and related functions.
	C_BOTH     = C.GSS_C_BOTH
	C_INITIATE = C.GSS_C_INITIATE
	C_ACCEPT   = C.GSS_C_ACCEPT

	// statusType values to be passed to DisplayStatus().
	C_GSS_CODE  = C.GSS_C_GSS_CODE
	C_MECH_CODE = C.GSS_C_MECH_CODE

	//	C_NO_NAME = nil
	//	C_NO_BUFFER = nil
	//	C_NO_OID = nil
	//	C_NO_OID_SET = nil
	//	C_NO_CONTEXT = nil
	//	C_NO_CREDENTIAL = nil
	//	C_NO_CHANNEL_BINDINGS = nil
	//	C_EMPTY_BUFFER = make([]byte, 0)

	//	C_NULL_OID = nil
	//	C_NULL_OID_SET = nil

	C_QOP_DEFAULT = C.GSS_C_QOP_DEFAULT

	// The maximum-allowed lifetime value.
	C_INDEFINITE = C.GSS_C_INDEFINITE

	C_CALLING_ERROR_OFFSET = C.GSS_C_CALLING_ERROR_OFFSET
	C_ROUTINE_ERROR_OFFSET = C.GSS_C_ROUTINE_ERROR_OFFSET
	C_SUPPLEMENTARY_OFFSET = C.GSS_C_SUPPLEMENTARY_OFFSET
	C_CALLING_ERROR_MASK   = C.GSS_C_CALLING_ERROR_MASK
	C_ROUTINE_ERROR_MASK   = C.GSS_C_ROUTINE_ERROR_MASK
	C_SUPPLEMENTARY_MASK   = C.GSS_C_SUPPLEMENTARY_MASK

	// Major result codes.
	S_COMPLETE                = C.GSS_S_COMPLETE
	S_CALL_INACCESSIBLE_READ  = C.GSS_S_CALL_INACCESSIBLE_READ
	S_CALL_INACCESSIBLE_WRITE = C.GSS_S_CALL_INACCESSIBLE_WRITE
	S_CALL_BAD_STRUCTURE      = C.GSS_S_CALL_BAD_STRUCTURE
	S_BAD_MECH                = C.GSS_S_BAD_MECH
	S_BAD_NAME                = C.GSS_S_BAD_NAME
	S_BAD_NAMETYPE            = C.GSS_S_BAD_NAMETYPE
	S_BAD_BINDINGS            = C.GSS_S_BAD_BINDINGS
	S_BAD_STATUS              = C.GSS_S_BAD_STATUS
	S_BAD_SIG                 = C.GSS_S_BAD_SIG
	S_NO_CRED                 = C.GSS_S_NO_CRED
	S_NO_CONTEXT              = C.GSS_S_NO_CONTEXT
	S_DEFECTIVE_TOKEN         = C.GSS_S_DEFECTIVE_TOKEN
	S_DEFECTIVE_CREDENTIAL    = C.GSS_S_DEFECTIVE_CREDENTIAL
	S_CREDENTIALS_EXPIRED     = C.GSS_S_CREDENTIALS_EXPIRED
	S_CONTEXT_EXPIRED         = C.GSS_S_CONTEXT_EXPIRED
	S_FAILURE                 = C.GSS_S_FAILURE
	S_BAD_QOP                 = C.GSS_S_BAD_QOP
	S_UNAUTHORIZED            = C.GSS_S_UNAUTHORIZED
	S_UNAVAILABLE             = C.GSS_S_UNAVAILABLE
	S_DUPLICATE_ELEMENT       = C.GSS_S_DUPLICATE_ELEMENT
	S_NAME_NOT_MN             = C.GSS_S_NAME_NOT_MN
	S_BAD_MECH_ATTR           = C.GSS_S_BAD_MECH_ATTR
	S_CONTINUE_NEEDED         = C.GSS_S_CONTINUE_NEEDED
	S_DUPLICATE_TOKEN         = C.GSS_S_DUPLICATE_TOKEN
	S_OLD_TOKEN               = C.GSS_S_OLD_TOKEN
	S_UNSEQ_TOKEN             = C.GSS_S_UNSEQ_TOKEN
	S_GAP_TOKEN               = C.GSS_S_GAP_TOKEN
	S_CRED_UNAVAIL            = C.GSS_S_CRED_UNAVAIL

	// prfKey values to be passed to PseudoRandom()
	C_PRF_KEY_FULL    = C.GSS_C_PRF_KEY_FULL
	C_PRF_KEY_PARTIAL = C.GSS_C_PRF_KEY_PARTIAL
)

var (
	C_INQ_SSPI_SESSION_KEY  = coidToOid(*C.GSS_C_INQ_SSPI_SESSION_KEY)
	C_ATTR_LOCAL_LOGIN_USER = bufferToString(*C.GSS_C_ATTR_LOCAL_LOGIN_USER)
	C_NT_COMPOSITE_EXPORT   = coidToOid(*C.GSS_C_NT_COMPOSITE_EXPORT)

	// Recognized name types.
	C_NT_USER_NAME                 = coidToOid(*C.GSS_C_NT_USER_NAME)
	C_NT_MACHINE_UID_NAME          = coidToOid(*C.GSS_C_NT_MACHINE_UID_NAME)
	C_NT_STRING_UID_NAME           = coidToOid(*C.GSS_C_NT_STRING_UID_NAME)
	C_NT_HOSTBASED_SERVICE_X       = coidToOid(*C.GSS_C_NT_HOSTBASED_SERVICE_X)
	C_NT_HOSTBASED_SERVICE         = coidToOid(*C.GSS_C_NT_HOSTBASED_SERVICE)
	C_NT_ANONYMOUS                 = coidToOid(*C.GSS_C_NT_ANONYMOUS)
	C_NT_EXPORT_NAME               = coidToOid(*C.GSS_C_NT_EXPORT_NAME)
	KRB5_NT_PRINCIPAL_NAME         = coidToOid(*C.GSS_KRB5_NT_PRINCIPAL_NAME)
	KRB5_NT_HOSTBASED_SERVICE_NAME = coidToOid(*C.GSS_KRB5_NT_HOSTBASED_SERVICE_NAME)
	KRB5_NT_USER_NAME              = coidToOid(*C.GSS_KRB5_NT_USER_NAME)
	KRB5_NT_MACHINE_UID_NAME       = coidToOid(*C.GSS_KRB5_NT_MACHINE_UID_NAME)
	KRB5_NT_STRING_UID_NAME        = coidToOid(*C.GSS_KRB5_NT_STRING_UID_NAME)

	// Recognized mechanism attributes.
	C_MA_MECH_CONCRETE  = coidToOid(*C.GSS_C_MA_MECH_CONCRETE)
	C_MA_MECH_PSEUDO    = coidToOid(*C.GSS_C_MA_MECH_PSEUDO)
	C_MA_MECH_COMPOSITE = coidToOid(*C.GSS_C_MA_MECH_COMPOSITE)
	C_MA_MECH_NEGO      = coidToOid(*C.GSS_C_MA_MECH_NEGO)
	C_MA_MECH_GLUE      = coidToOid(*C.GSS_C_MA_MECH_GLUE)
	C_MA_NOT_MECH       = coidToOid(*C.GSS_C_MA_NOT_MECH)
	C_MA_DEPRECATED     = coidToOid(*C.GSS_C_MA_DEPRECATED)
	C_MA_NOT_DFLT_MECH  = coidToOid(*C.GSS_C_MA_NOT_DFLT_MECH)
	C_MA_ITOK_FRAMED    = coidToOid(*C.GSS_C_MA_ITOK_FRAMED)
	C_MA_AUTH_INIT      = coidToOid(*C.GSS_C_MA_AUTH_INIT)
	C_MA_AUTH_TARG      = coidToOid(*C.GSS_C_MA_AUTH_TARG)
	C_MA_AUTH_INIT_INIT = coidToOid(*C.GSS_C_MA_AUTH_INIT_INIT)
	C_MA_AUTH_TARG_INIT = coidToOid(*C.GSS_C_MA_AUTH_TARG_INIT)
	C_MA_AUTH_INIT_ANON = coidToOid(*C.GSS_C_MA_AUTH_INIT_ANON)
	C_MA_AUTH_TARG_ANON = coidToOid(*C.GSS_C_MA_AUTH_TARG_ANON)
	C_MA_DELEG_CRED     = coidToOid(*C.GSS_C_MA_DELEG_CRED)
	C_MA_INTEG_PROT     = coidToOid(*C.GSS_C_MA_INTEG_PROT)
	C_MA_CONF_PROT      = coidToOid(*C.GSS_C_MA_CONF_PROT)
	C_MA_MIC            = coidToOid(*C.GSS_C_MA_MIC)
	C_MA_WRAP           = coidToOid(*C.GSS_C_MA_WRAP)
	C_MA_PROT_READY     = coidToOid(*C.GSS_C_MA_PROT_READY)
	C_MA_REPLAY_DET     = coidToOid(*C.GSS_C_MA_REPLAY_DET)
	C_MA_OOS_DET        = coidToOid(*C.GSS_C_MA_OOS_DET)
	C_MA_CBINDINGS      = coidToOid(*C.GSS_C_MA_CBINDINGS)
	C_MA_PFS            = coidToOid(*C.GSS_C_MA_PFS)
	C_MA_COMPRESS       = coidToOid(*C.GSS_C_MA_COMPRESS)
	C_MA_CTX_TRANS      = coidToOid(*C.GSS_C_MA_CTX_TRANS)

	// Some mechanisms.
	Mech_krb5          = coidToOid(*C.gss_mech_krb5)
	Mech_krb5_old      = coidToOid(*C.gss_mech_krb5_old)
	Mech_krb5_wrong    = coidToOid(*C.gss_mech_krb5_wrong)
	Mech_iakerb        = coidToOid(*C.gss_mech_iakerb)
	Mech_spnego        = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 2}
	Mech_set_krb5      = coidSetToOids(C.gss_mech_set_krb5)
	Mech_set_krb5_old  = coidSetToOids(C.gss_mech_set_krb5_old)
	Mech_set_krb5_both = coidSetToOids(C.gss_mech_set_krb5_both)

	NT_krb5_name      = coidToOid(*C.gss_nt_krb5_name)
	NT_krb5_principal = coidToOid(*C.gss_nt_krb5_principal)
	//Krb5_gss_oid_array = coidToOid(C.krb5_gss_oid_array)
)

/* CredHandle holds a reference to client or server credentials, or delegated credentials.  It should be released using gss.ReleaseCred() when it's no longer needed. */
type CredHandle C.gss_cred_id_t

/* CredHandle holds a reference to an established or partially-established security context.  It should be released using gss.DeleteSecContext() when it's no longer needed. */
type ContextHandle C.gss_ctx_id_t

/* CredHandle holds a reference to a client or server's name.  It should be released using gss.ReleaseName() when it's no longer needed. */
type InternalName C.gss_name_t

type ChannelBindings struct {
	// These four fields are deprecated.
	//initiatorAddressType uint32
	//acceptorAddressType          uint32
	//initiatorAddress []byte
	//acceptorAddress []byte
	ApplicationData []byte
}

/* Flags describe requested parameters for a context passed to InitSecContext(), or the parameters of an established context as returned by AcceptSecContext() or InquireContext(). */
type Flags struct {
	Deleg, DelegPolicy, Mutual, Replay, Sequence, Anon, Conf, Integ, Trans, ProtReady bool
}

/* bytesToBuffer populates a gss_buffer_t with a borrowed reference to the contents of the slice. */
func bytesToBuffer(data []byte) (cdesc C.gss_buffer_desc) {
	value := unsafe.Pointer(&data[0])
	length := C.size_t(len(data))

	cdesc.value = value
	cdesc.length = length
	return
}

/* bufferToBytes creates a byte array using the contents of the passed-in buffer. */
func bufferToBytes(cdesc C.gss_buffer_desc) (b []byte) {
	length := C.int(cdesc.length)

	b = C.GoBytes(cdesc.value, length)
	return
}

/* buffersToBytes creates an array of byte arrays using the contents of the passed-in buffer set. */
func buffersToBytes(cdesc C.gss_buffer_set_desc) (b [][]byte) {
	count := uint(cdesc.count)
	var i uint

	b = make([][]byte, count)
	for i = 0; i < count; i++ {
		b[i] = bufferToBytes(C.nth_buffer_in_set(&cdesc, C.uint(i)))
	}
	return
}

/* bufferToString creates a string using the contents of the passed-in buffer. */
func bufferToString(cdesc C.gss_buffer_desc) (text string) {
	b := bufferToBytes(cdesc)
	if len(b) > 0 && b[len(b)-1] == 0 {
		b = b[0 : len(b)-1]
	}
	buf := bytes.NewBuffer(b)
	text = buf.String()
	return
}

/* stringToBuffer creates a buffer using the contents of the string. */
func stringToBuffer(text string) (cdesc C.gss_buffer_desc) {
	value := unsafe.Pointer(C.CString(text))
	length := C.size_t(len(text))

	cdesc.value = value
	cdesc.length = length
	return
}

/* buffersToStrings creates a string array using the contents of the passed-in buffer set. */
func buffersToStrings(cdesc C.gss_buffer_set_desc) (s []string) {
	count := uint(cdesc.count)
	var i uint

	s = make([]string, count)
	for i = 0; i < count; i++ {
		s[i] = bufferToString(C.nth_buffer_in_set(&cdesc, C.uint(i)))
	}
	return
}

/* Encode a tag and a length as a DER definite length */
func makeTagAndLength(tag, length int) (l []byte) {
	var count, bits int

	if length <= 127 {
		l = make([]byte, 2)
		l[0] = byte(tag)
		l[1] = byte(length)
		return
	}
	count = 0
	bits = length
	for bits != 0 {
		count++
		bits = bits >> 8
	}
	if count > 126 {
		return nil
	}
	l = make([]byte, 2+count)
	count = 0
	bits = length
	l[0] = byte(tag)
	for bits != 0 {
		l[len(l)-1-count] = byte(bits & 0xff)
		count++
		bits = bits >> 8
	}
	l[1] = byte(count | 0x80)
	return
}

/* Split up a DER item */
func splitTagAndLength(tlv []byte) (class int, constructed bool, tag, length int, value []byte) {
	tbytes := 1
	lbytes := 1

	class = int((tlv[0] & 0xc0) >> 6)
	constructed = (tlv[0] & 0x20) != 0
	tag = int(tlv[0] & 0x1f)
	if tag == 0x1f {
		tag = 0
		for tlv[tbytes]&0x80 != 0 {
			tag = (tag << 7) + int(tlv[tbytes]&0x7f)
			tbytes++
		}
		tag = (tag << 7) + int(tlv[tbytes]&0x7f)
		tbytes++
	}
	if tlv[tbytes]&0x80 == 0 {
		length = int(tlv[tbytes] & 0x7f)
	} else {
		lbytes = int(tlv[tbytes] & 0x7f)
		if lbytes == 0 {
			value = nil
			return
		}
		for count := 0; count < lbytes; count++ {
			length = (length << 8) + int(tlv[tbytes+1+count]&0xff)
		}
		lbytes++
	}
	if len(tlv) != tbytes+lbytes+length {
		value = nil
		return
	}
	value = tlv[(tbytes + lbytes):]
	return
}

/* coidToOid produces an asn1.ObjectIdentifier from the library's preferred bytes-and-length representation, which is just the DER encoding without a tag and length. */
func coidToOid(coid C.gss_OID_desc) (oid asn1.ObjectIdentifier) {
	length := C.int(coid.length)
	b := C.GoBytes(coid.elements, length)

	b = append(makeTagAndLength(6, len(b)), b...)
	asn1.Unmarshal(b, &oid)
	return
}

/* oidToCOid converts an asn1.ObjectIdentifier into an array of encoded bytes without the tag and length, which is how the C library expects them to be structured. */
func oidToCOid(oid asn1.ObjectIdentifier) (coid C.gss_OID) {
	if oid == nil {
		return
	}

	b, _ := asn1.Marshal(oid)
	if b == nil {
		return
	}
	_, _, _, _, v := splitTagAndLength(b)
	if v == nil {
		return
	}
	length := len(v)
	if length == 0 {
		return
	}
	coid = C.gss_OID(C.calloc(1, C.size_t(unsafe.Sizeof(*coid))))
	coid.length = C.OM_uint32(length)
	coid.elements = C.copyOid((*C.uchar)(&v[0]), C.int(length))
	if coid.elements == nil {
		C.free_oid(coid)
		coid = nil
	}
	return
}

/* oidsToCOidSet converts an array of asn1.ObjectIdentifier items into an array of arrays of encoded bytes-and-lengths, which is how the C library expects them to be structured. */
func oidsToCOidSet(oidSet []asn1.ObjectIdentifier) (coids C.gss_OID_set) {
	var major, minor C.OM_uint32
	if oidSet == nil {
		return
	}
	major = C.gss_create_empty_oid_set(&minor, &coids)
	if major != 0 {
		return
	}

	for _, o := range oidSet {
		oid := oidToCOid(o)
		if oid == nil {
			continue
		}
		major = C.gss_add_oid_set_member(&minor, oid, &coids)
		C.free_oid(oid)
		if major != 0 {
			major = C.gss_release_oid_set(&minor, &coids)
			coids = nil
			return
		}
	}
	return
}

/* coidSetToOids produces an array of asn1.ObjectIdentifier items from the library's preferred array-of-bytes-and-lengths representation. */
func coidSetToOids(coids *C.gss_OID_set_desc) (oidSet []asn1.ObjectIdentifier) {
	if coids == nil {
		return nil
	}

	oidSet = make([]asn1.ObjectIdentifier, coids.count)
	if oidSet == nil {
		return
	}

	for o := 0; o < int(coids.count); o++ {
		coid := C.nth_oid_in_set(coids, C.uint(o))
		oidSet[o] = coidToOid(coid)
	}

	return
}

func bindingsToCBindings(bindings *ChannelBindings) (cbindings C.gss_channel_bindings_t) {
	if bindings == nil {
		return nil
	}
	cbindings.application_data = bytesToBuffer(bindings.ApplicationData)
	return
}

func cbindingsToBindings(cbindings C.gss_channel_bindings_t) (bindings *ChannelBindings) {
	if cbindings == nil {
		return nil
	}
	bindings.ApplicationData = bufferToBytes(cbindings.application_data)
	return
}

func credStoreToKVSet(credStore [][2]string) (kvset C.gss_key_value_set_desc) {
	kvset.elements = C.alloc_n_kvset_elems(C.uint(len(credStore)))
	if kvset.elements == nil {
		return
	}
	for i, kv := range credStore {
		C.kv_set(&kvset, C.int(i), C.CString(kv[0]), C.CString(kv[1]))
	}
	kvset.count = C.OM_uint32(len(credStore))
	return
}

/* AcquireCred() obtains credentials to be used to either initiate or accept (or both) a security context as desiredName.  The returned outputCredHandle should be released using gss.ReleaseCred() when it's no longer needed. */
func AcquireCred(desiredName InternalName, lifetimeReq uint32, desiredMechs []asn1.ObjectIdentifier, credUsage uint32) (majorStatus, minorStatus uint32, outputCredHandle CredHandle, actualMechs []asn1.ObjectIdentifier, lifetimeRec uint32) {
	name := C.gss_name_t(desiredName)
	lifetime := C.OM_uint32(lifetimeReq)
	desired := oidsToCOidSet(desiredMechs)
	usage := C.gss_cred_usage_t(credUsage)
	var major, minor C.OM_uint32
	var actual C.gss_OID_set
	var handle C.gss_cred_id_t

	major = C.gss_acquire_cred(&minor, name, lifetime, desired, usage, &handle, &actual, &lifetime)
	C.free_oid_set(desired)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandle = CredHandle(handle)
	actualMechs = coidSetToOids(actual)
	C.free_oid_set(actual)
	lifetimeRec = uint32(lifetime)
	return
}

/* ReleaseCred() releases a credential handle which is no longer needed. */
func ReleaseCred(credHandle CredHandle) (majorStatus, minorStatus uint32) {
	handle := C.gss_cred_id_t(credHandle)
	var major, minor C.OM_uint32

	major = C.gss_release_cred(&minor, &handle)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* InquireCred() reads information about a credential handle, or about the default acceptor credentials if credHandle is nil.  The returned credName should be released using gss.ReleaseName() when it's no longer needed. */
func InquireCred(credHandle CredHandle) (majorStatus, minorStatus uint32, credName InternalName, lifetimeRec, credUsage uint32, mechSet []asn1.ObjectIdentifier) {
	handle := C.gss_cred_id_t(credHandle)
	name := C.gss_name_t(nil)
	var major, minor, lifetime C.OM_uint32
	var usage C.gss_cred_usage_t
	var mechs C.gss_OID_set

	major = C.gss_inquire_cred(&minor, handle, &name, &lifetime, &usage, &mechs)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	credName = InternalName(name)
	lifetimeRec = uint32(lifetime)
	credUsage = uint32(usage)
	mechSet = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	return
}

/* AddCred() obtains credentials specific to a particular mechanism, optionally merging them with already-obtained credentials (if outputCredHandle is not nil) or storing them in an entirely new credential handle. */
func AddCred(credHandle CredHandle, desiredName InternalName, desiredMech asn1.ObjectIdentifier, initiatorTimeReq, acceptorTimeReq, credUsage uint32, outputCredHandle CredHandle) (majorStatus, minorStatus uint32, outputCredHandleRec CredHandle, actualMechs []asn1.ObjectIdentifier, initiatorTimeRec, acceptorTimeRec uint32) {
	handle := C.gss_cred_id_t(credHandle)
	name := C.gss_name_t(desiredName)
	mech := oidToCOid(desiredMech)
	itime := C.OM_uint32(initiatorTimeReq)
	atime := C.OM_uint32(acceptorTimeReq)
	usage := C.gss_cred_usage_t(credUsage)
	ohandle := C.gss_cred_id_t(outputCredHandle)
	var major, minor C.OM_uint32
	var mechs C.gss_OID_set

	major = C.gss_add_cred(&minor, handle, name, mech, usage, itime, atime, &ohandle, &mechs, &itime, &atime)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandleRec = CredHandle(ohandle)
	actualMechs = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	initiatorTimeRec = uint32(itime)
	acceptorTimeRec = uint32(atime)
	return
}

/* InquireCredByMech() obtains information about mechanism-specific credentials.  The returned credName is a mechanism-specific name, and should be released using gss.ReleaseName() when it's no longer needed. */
func InquireCredByMech(credHandle CredHandle, mechType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, credName InternalName, initiatorLifetimeRec, acceptorLifetimeRec, credUsage uint32) {
	handle := C.gss_cred_id_t(credHandle)
	mech := oidToCOid(mechType)
	var major, minor, ilife, alife C.OM_uint32
	var name C.gss_name_t
	var usage C.gss_cred_usage_t

	major = C.gss_inquire_cred_by_mech(&minor, handle, mech, &name, &ilife, &alife, &usage)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	credName = InternalName(name)
	initiatorLifetimeRec = uint32(ilife)
	acceptorLifetimeRec = uint32(alife)
	credUsage = uint32(usage)
	return
}

func flagsToInt(flags Flags) (recFlags C.OM_uint32) {
	if flags.Deleg {
		recFlags |= C.GSS_C_DELEG_FLAG
	}
	if flags.DelegPolicy {
		recFlags |= C.GSS_C_DELEG_POLICY_FLAG
	}
	if flags.Mutual {
		recFlags |= C.GSS_C_MUTUAL_FLAG
	}
	if flags.Replay {
		recFlags |= C.GSS_C_REPLAY_FLAG
	}
	if flags.Sequence {
		recFlags |= C.GSS_C_SEQUENCE_FLAG
	}
	if flags.Anon {
		recFlags |= C.GSS_C_ANON_FLAG
	}
	if flags.Conf {
		recFlags |= C.GSS_C_CONF_FLAG
	}
	if flags.Integ {
		recFlags |= C.GSS_C_INTEG_FLAG
	}
	if flags.Trans {
		recFlags |= C.GSS_C_TRANS_FLAG
	}
	if flags.ProtReady {
		recFlags |= C.GSS_C_PROT_READY_FLAG
	}
	return
}

func flagsToFlags(flags C.OM_uint32) (recFlags Flags) {
	if flags&C.GSS_C_DELEG_FLAG != 0 {
		recFlags.Deleg = true
	}
	if flags&C.GSS_C_DELEG_POLICY_FLAG != 0 {
		recFlags.DelegPolicy = true
	}
	if flags&C.GSS_C_MUTUAL_FLAG != 0 {
		recFlags.Mutual = true
	}
	if flags&C.GSS_C_REPLAY_FLAG != 0 {
		recFlags.Replay = true
	}
	if flags&C.GSS_C_SEQUENCE_FLAG != 0 {
		recFlags.Sequence = true
	}
	if flags&C.GSS_C_ANON_FLAG != 0 {
		recFlags.Anon = true
	}
	if flags&C.GSS_C_CONF_FLAG != 0 {
		recFlags.Conf = true
	}
	if flags&C.GSS_C_INTEG_FLAG != 0 {
		recFlags.Integ = true
	}
	if flags&C.GSS_C_TRANS_FLAG != 0 {
		recFlags.Trans = true
	}
	if flags&C.GSS_C_PROT_READY_FLAG != 0 {
		recFlags.ProtReady = true
	}
	return
}

/* FlagsToRaw returns the integer representation of the flags structure, as would typically be used by C implementations.  It is here mainly to aid in running diagnostics. */
func FlagsToRaw(flags Flags) uint32 {
	return uint32(flagsToInt(flags))
}

/* Initialize a security context with a peer named by targName, optionally specifying a requested GSSAPI mechanism.  If the application expects to use confidentiality or integrity-checking functionality, they should be specified in reqFlags.  If the returned majorStatus is gss.S_CONTINUE_NEEDED, the function should be called again using the same contextHandle, but with a new token obtained from the peer.  This may need to be done an unknown number of times.  Any output tokens produced (including when the returned majorStatus is gss.S_COMPLETE) should be sent to the peer.  The context is successfully set up when the returned majorStatus is gss.S_COMPLETE.  If contextHandle is not nil, it should eventually be freed using gss.DeleteSecContext(). */
func InitSecContext(claimantCredHandle CredHandle, contextHandle *ContextHandle, targName InternalName, mechType asn1.ObjectIdentifier, reqFlags Flags, lifetimeReq uint32, chanBindings *ChannelBindings, inputToken []byte) (majorStatus, minorStatus uint32, mechTypeRec asn1.ObjectIdentifier, outputToken []byte, recFlags Flags, transState, protReadyState bool, lifetimeRec uint32) {
	handle := C.gss_cred_id_t(claimantCredHandle)
	ctx := C.gss_ctx_id_t(*contextHandle)
	name := C.gss_name_t(targName)
	desired := oidToCOid(mechType)
	flags := flagsToInt(reqFlags)
	lifetime := C.OM_uint32(lifetimeReq)
	bindings := bindingsToCBindings(chanBindings)
	var major, minor C.OM_uint32
	var itoken, otoken C.gss_buffer_desc
	var actual C.gss_OID

	if inputToken != nil {
		itoken = bytesToBuffer(inputToken)
	}

	major = C.gss_init_sec_context(&minor, handle, &ctx, name, desired, flags, lifetime, bindings, &itoken, &actual, &otoken, &flags, &lifetime)
	C.free_oid(desired)

	*contextHandle = ContextHandle(ctx)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if actual != nil {
		mechTypeRec = coidToOid(*actual)
		/* actual is read-only, so don't free it */
	}
	if otoken.length > 0 {
		outputToken = bufferToBytes(otoken)
		major = C.gss_release_buffer(&minor, &otoken)
	}
	recFlags = flagsToFlags(flags)
	if flags&C.GSS_C_TRANS_FLAG != 0 {
		transState = true
	}
	if flags&C.GSS_C_PROT_READY_FLAG != 0 {
		protReadyState = true
	}
	lifetimeRec = uint32(lifetime)
	return
}

/* Accept a security context from a peer, using the specified acceptor credentials, or the default acceptor credentials if acceptorCredHandle is nil.  If the returned majorStatus is gss.S_CONTINUE_NEEDED, the function should be called again using the same contextHandle, but with a new token obtained from the peer.  This may need to be done an unknown number of times.  Any output tokens produced (including when the returned majorStatus is gss.S_COMPLETE) should be sent to the peer.  The context is successfully set up when the returned majorStatus is gss.S_COMPLETE.  If contextHandle is not nil, it should eventually be freed using gss.DeleteSecContext().  If srcName is not nil, it should eventually be freed using gss.ReleaseName().  If delegatedCredHandle is not nil, it should also be freed. */
func AcceptSecContext(acceptorCredHandle CredHandle, contextHandle *ContextHandle, chanBindings *ChannelBindings, inputToken []byte) (majorStatus, minorStatus uint32, srcName InternalName, mechType asn1.ObjectIdentifier, recFlags Flags, transState, protReadyState bool, lifetimeRec uint32, delegatedCredHandle CredHandle, outputToken []byte) {
	handle := C.gss_cred_id_t(acceptorCredHandle)
	ctx := C.gss_ctx_id_t(*contextHandle)
	bindings := bindingsToCBindings(chanBindings)
	var major, minor, flags, lifetime C.OM_uint32
	var name C.gss_name_t
	var itoken, otoken C.gss_buffer_desc
	var actual C.gss_OID
	var dhandle C.gss_cred_id_t

	if inputToken != nil {
		itoken = bytesToBuffer(inputToken)
	}

	major = C.gss_accept_sec_context(&minor, &ctx, handle, &itoken, bindings, &name, &actual, &otoken, &flags, &lifetime, &dhandle)
	*contextHandle = ContextHandle(ctx)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	srcName = InternalName(name)
	if actual != nil {
		mechType = coidToOid(*actual)
		/* actual is read-only, so don't free it */
	}
	recFlags = flagsToFlags(flags)
	if flags&C.GSS_C_TRANS_FLAG != 0 {
		transState = true
	}
	if flags&C.GSS_C_PROT_READY_FLAG != 0 {
		protReadyState = true
	}
	lifetimeRec = uint32(lifetime)
	delegatedCredHandle = CredHandle(dhandle)
	if otoken.length > 0 {
		outputToken = bufferToBytes(otoken)
		major = C.gss_release_buffer(&minor, &otoken)
	}
	return
}

/* DeleteSecContext() frees resources associated with a security context which is no longer needed.  If an outputContextToken is produced, the calling application should attempt to send it to the peer to pass to ProcessContextToken(). */
func DeleteSecContext(contextHandle ContextHandle) (majorStatus, minorStatus uint32, outputContextToken []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	var major, minor C.OM_uint32
	var token C.gss_buffer_desc

	major = C.gss_delete_sec_context(&minor, &handle, &token)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if token.value != nil {
		outputContextToken = bufferToBytes(token)
		major = C.gss_release_buffer(&minor, &token)
	}
	return
}

/* ProcessContextToken() processes a context token which was created using gss.DeleteSecContext().  It is not usually used, and is included for backward compatibility. */
func ProcessContextToken(contextHandle ContextHandle, contextToken []byte) (majorStatus, minorStatus uint32) {
	handle := C.gss_ctx_id_t(contextHandle)
	var major, minor C.OM_uint32
	var token C.gss_buffer_desc

	if contextToken != nil {
		token = bytesToBuffer(contextToken)
	}

	major = C.gss_process_context_token(&minor, handle, &token)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* ContextTime() returns the amount of time for which an already-established security context will remain valid. */
func ContextTime(contextHandle ContextHandle) (majorStatus, minorStatus, lifetimeRec uint32) {
	handle := C.gss_ctx_id_t(contextHandle)
	var major, minor, lifetime C.OM_uint32

	major = C.gss_context_time(&minor, handle, &lifetime)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	lifetimeRec = uint32(lifetime)
	return
}

/* OidToStr() converts an OID to a displayable form preferred by the GSSAPI library, which may differ from the default representation returned by oid's String() method. */
func OidToStr(oid asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, text string) {
	id := oidToCOid(oid)
	var major, minor C.OM_uint32
	var s C.gss_buffer_desc

	major = C.gss_oid_to_str(&minor, id, &s)
	C.free_oid(id)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if s.length > 0 {
		text = bufferToString(s)
		major = C.gss_release_buffer(&minor, &s)
	}
	return
}

/* InquireContext() returns information about an already-established security context.  The returned srcName and targName values should be released using gss.ReleaseName(). */
func InquireContext(contextHandle ContextHandle) (majorStatus, minorStatus uint32, srcName, targName InternalName, lifetimeRec uint32, mechType asn1.ObjectIdentifier, recFlags Flags, transState, protReadyState, locallyInitiated, open bool) {
	handle := C.gss_ctx_id_t(contextHandle)
	var major, minor, lifetime C.OM_uint32
	var sname, tname C.gss_name_t
	var mech C.gss_OID
	var localinit, opened C.int
	var flags C.OM_uint32

	major = C.gss_inquire_context(&minor, handle, &sname, &tname, &lifetime, &mech, &flags, &localinit, &opened)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	srcName = InternalName(sname)
	targName = InternalName(tname)
	lifetimeRec = uint32(lifetime)
	if mech != nil {
		mechType = coidToOid(*mech)
		/* mech is read-only, so don't free it */
	}
	recFlags = flagsToFlags(flags)
	if flags&C.GSS_C_TRANS_FLAG != 0 {
		transState = true
	}
	if flags&C.GSS_C_PROT_READY_FLAG != 0 {
		protReadyState = true
	}
	locallyInitiated = (localinit != 0)
	open = (opened != 0)
	return
}

/* WrapSizeLimit() returns the maximum size of plaintext which the underlying mechanism can accept if it must guarantee that wrapped tokens must be less than or equal to outputSize bytes. */
func WrapSizeLimit(contextHandle ContextHandle, confReqFlag bool, qopReq uint32, outputSize uint32) (majorStatus, minorStatus, maxInputSize uint32) {
	handle := C.gss_ctx_id_t(contextHandle)
	qop := C.gss_qop_t(qopReq)
	output := C.OM_uint32(outputSize)
	var conf C.int
	var major, minor, input C.OM_uint32

	if confReqFlag {
		conf = 1
	}

	major = C.gss_wrap_size_limit(&minor, handle, conf, qop, output, &input)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	maxInputSize = uint32(input)
	return
}

/* ExportSecContext() serializes all state data related to an established security context.  Upon return, contextHandle will have become invalid. */
func ExportSecContext(contextHandle ContextHandle) (majorStatus, minorStatus uint32, interProcessToken []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	var token C.gss_buffer_desc
	var major, minor C.OM_uint32

	major = C.gss_export_sec_context(&minor, &handle, &token)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if token.length > 0 {
		interProcessToken = bufferToBytes(token)
		major = C.gss_release_buffer(&minor, &token)
	}
	return
}

/* ImportSecContext() deserializes all state data related to an established security context and reconstructs it.  The returned contextHandle can be used immediately, and should eventually be freed using gss.DeleteSecContext(). */
func ImportSecContext(interprocessToken []byte) (majorStatus, minorStatus uint32, contextHandle ContextHandle) {
	token := bytesToBuffer(interprocessToken)
	var major, minor C.OM_uint32
	var handle C.gss_ctx_id_t

	major = C.gss_import_sec_context(&minor, &token, &handle)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	contextHandle = ContextHandle(handle)
	return
}

/* GetMIC() computes a signature over the passed-in message. */
func GetMIC(contextHandle ContextHandle, qopReq uint32, message []byte) (majorStatus, minorStatus uint32, perMessageToken []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	qop := C.gss_qop_t(qopReq)
	var msg, mic C.gss_buffer_desc
	var major, minor C.OM_uint32

	msg = bytesToBuffer(message)

	major = C.gss_get_mic(&minor, handle, qop, &msg, &mic)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if mic.length > 0 {
		perMessageToken = bufferToBytes(mic)
		major = C.gss_release_buffer(&minor, &mic)
	}
	return
}

/* VerifyMIC() checks a passed-in signature over a passed-in message. */
func VerifyMIC(contextHandle ContextHandle, message, perMessageToken []byte) (majorStatus, minorStatus, qopState uint32) {
	handle := C.gss_ctx_id_t(contextHandle)
	msg := bytesToBuffer(message)
	mic := bytesToBuffer(perMessageToken)
	var major, minor C.OM_uint32
	var qop C.gss_qop_t

	major = C.gss_verify_mic(&minor, handle, &msg, &mic, &qop)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	qopState = uint32(qop)
	return
}

/* Wrap() produces either an integrity-protected or confidential token containing the passed-in inputMessage. */
func Wrap(contextHandle ContextHandle, confReq bool, qopReq uint32, inputMessage []byte) (majorStatus, minorStatus uint32, confState bool, outputMessage []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	qop := C.gss_qop_t(qopReq)
	var major, minor C.OM_uint32
	var msg, wrapped C.gss_buffer_desc
	var conf C.int

	if confReq {
		conf = 1
	}
	msg = bytesToBuffer(inputMessage)

	major = C.gss_wrap(&minor, handle, conf, qop, &msg, &conf, &wrapped)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	confState = (conf != 0)
	if wrapped.length > 0 {
		outputMessage = bufferToBytes(wrapped)
		major = C.gss_release_buffer(&minor, &wrapped)
	}
	return
}

/* Unwrap() accepts an integrity-protected or confidential token and returns the plaintext, along with an indication of whether or not the input token was confidential (encrypted). */
func Unwrap(contextHandle ContextHandle, inputMessage []byte) (majorStatus, minorStatus uint32, confState bool, qopState uint32, outputMessage []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	wrapped := bytesToBuffer(inputMessage)
	var major, minor C.OM_uint32
	var msg C.gss_buffer_desc
	var conf C.int
	var qop C.gss_qop_t

	major = C.gss_unwrap(&minor, handle, &wrapped, &msg, &conf, &qop)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	confState = (conf != 0)
	qopState = uint32(qop)
	if msg.length > 0 {
		outputMessage = bufferToBytes(msg)
		major = C.gss_release_buffer(&minor, &msg)
	}
	return
}

/* DisplayStatus() returns a printable representation of a major (C_GSS_CODE) or mechanism-specific minor (C_MECH_CODE) status code. */
func DisplayStatus(statusValue uint32, statusType int, mechType asn1.ObjectIdentifier) []interface{} {
	value := C.OM_uint32(statusValue)
	stype := C.int(statusType)
	mech := oidToCOid(mechType)
	var major, minor, mctx C.OM_uint32
	var status C.gss_buffer_desc

	major = C.gss_display_status(&minor, value, stype, mech, &mctx, &status)
	C.free_oid(mech)

	majorStatus := uint32(major)
	minorStatus := uint32(minor)
	messageContext := uint32(mctx)
	statusString := ""
	if status.length > 0 {
		statusString = bufferToString(status)
		major = C.gss_release_buffer(&minor, &status)
	}
	return []interface{}{majorStatus, minorStatus, messageContext, statusString}
}

/* IndicateMechs() returns a list of the available security mechanism types. */
func IndicateMechs() (majorStatus, minorStatus uint32, mechSet []asn1.ObjectIdentifier) {
	var major, minor C.OM_uint32
	var mechs C.gss_OID_set

	major = C.gss_indicate_mechs(&minor, &mechs)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	mechSet = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	return
}

/* CompareName() compares two names to see if they refer to the same entity. */
func CompareName(name1, name2 InternalName) (majorStatus, minorStatus uint32, nameEqual bool) {
	n1 := C.gss_name_t(name1)
	n2 := C.gss_name_t(name2)
	var major, minor C.OM_uint32
	var equal C.int

	major = C.gss_compare_name(&minor, n1, n2, &equal)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	nameEqual = (equal != 0)
	return
}

/* DisplayName() returns a printable representation of name, along with the type of name that it represents. */
func DisplayName(name InternalName) (majorStatus, minorStatus uint32, nameString string, nameType asn1.ObjectIdentifier) {
	n := C.gss_name_t(name)
	var major, minor C.OM_uint32
	var dname C.gss_buffer_desc
	var ntype C.gss_OID

	major = C.gss_display_name(&minor, n, &dname, &ntype)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if dname.length > 0 {
		nameString = bufferToString(dname)
		major = C.gss_release_buffer(&minor, &dname)
	}
	if ntype != nil {
		nameType = coidToOid(*ntype)
		/* nameType is read-only, so don't free it */
	}
	return
}

/* ImportName() creates an InternalName from an external representation and name type, which is often gss.C_NT_USER_NAME or gss.C_NT_HOSTBASED_SERVICE.  The returned outputName should eventually be freed by calling gss.ReleaseName(). */
func ImportName(inputName string, nameType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, outputName InternalName) {
	ntype := oidToCOid(nameType)
	var major, minor C.OM_uint32
	var name C.gss_buffer_desc
	var iname C.gss_name_t

	name.length = C.size_t(len(inputName))
	name.value = unsafe.Pointer(C.CString(inputName))

	major = C.gss_import_name(&minor, &name, ntype, &iname)
	C.free_gss_buffer(name)
	C.free_oid(ntype)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputName = InternalName(iname)
	return
}

/* ReleaseName() frees resources associated with an InternalName after it is no longer needed. */
func ReleaseName(inputName InternalName) (majorStatus, minorStatus uint32) {
	name := C.gss_name_t(inputName)
	var major, minor C.OM_uint32

	major = C.gss_release_name(&minor, &name)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* ReleaseBuffer ReleaseOidSet CreateEmptyOidSet AddOidSetMember TestOidSetMember */

/* InquireNamesForMech() returns a list of the name types which can be used with the specified mechanism. */
func InquireNamesForMech(inputMechType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, nameTypeSet []asn1.ObjectIdentifier) {
	mech := oidToCOid(inputMechType)
	var major, minor C.OM_uint32
	var ntypes C.gss_OID_set

	major = C.gss_inquire_names_for_mech(&minor, mech, &ntypes)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	nameTypeSet = coidSetToOids(ntypes)
	C.free_oid_set(ntypes)
	return
}

/* InquireMechsForName() returns a list of the mechanisms with which the provided name can be used. */
func InquireMechsForName(inputName InternalName) (majorStatus, minorStatus uint32, mechTypes []asn1.ObjectIdentifier) {
	name := C.gss_name_t(inputName)
	var major, minor C.OM_uint32
	var mechs C.gss_OID_set

	major = C.gss_inquire_mechs_for_name(&minor, name, &mechs)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	mechTypes = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	return
}

/* CanonicalizeName() returns a copy of inputName that has been canonicalized according to the rules for the specified mechanism.  The returned outputName should eventually be freed using gss.ReleaseName(). */
func CanonicalizeName(inputName InternalName, mechType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, outputName InternalName) {
	name := C.gss_name_t(inputName)
	mech := oidToCOid(mechType)
	var major, minor C.OM_uint32
	var newname C.gss_name_t

	major = C.gss_canonicalize_name(&minor, name, mech, &newname)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputName = InternalName(newname)
	return
}

/* ExportName() returns a flat representation of a mechanism-specific inputName that's suitable for bytewise comparison with other exported names. */
func ExportName(inputName InternalName) (majorStatus, minorStatus uint32, outputName []byte) {
	name := C.gss_name_t(inputName)
	var major, minor C.OM_uint32
	var newname C.gss_buffer_desc

	major = C.gss_export_name(&minor, name, &newname)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if newname.length > 0 {
		outputName = bufferToBytes(newname)
		major = C.gss_release_buffer(&minor, &newname)
	}
	return
}

/* DuplicateName() returns a copy of inputName which will eventually need to be freed using gss.ReleaseName(). */
func DuplicateName(inputName InternalName) (majorStatus, minorStatus uint32, destName InternalName) {
	name := C.gss_name_t(inputName)
	var major, minor C.OM_uint32
	var newname C.gss_name_t

	major = C.gss_duplicate_name(&minor, name, &newname)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	destName = InternalName(newname)
	return
}

/* PseudoRandom() generates some pseudo-random data using the context handle of the desired level of randomness (either gss.C_PRF_KEY_FULL or gss.C_PRF_KEY_PARTIAL) of the desired size. */
func PseudoRandom(contextHandle ContextHandle, prfKey int, prfIn []byte, desiredOutputLen int) (majorStatus, minorStatus uint32, prfOut []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	pkey := C.int(prfKey)
	pin := bytesToBuffer(prfIn)
	desired := C.ssize_t(desiredOutputLen)
	var major, minor C.OM_uint32
	var pout C.gss_buffer_desc

	major = C.gss_pseudo_random(&minor, handle, pkey, &pin, desired, &pout)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if pout.length > 0 {
		prfOut = bufferToBytes(pout)
	}
	return
}

/* StoreCred() stores non-nil credentials (for initiator, acceptor, or both) in the current credential store. */
func StoreCred(credHandle CredHandle, credUsage uint32, desiredMech asn1.ObjectIdentifier, overwriteCred, defCred bool) (majorStatus, minorStatus uint32, elementsStored []asn1.ObjectIdentifier, credUsageStored uint32) {
	handle := C.gss_cred_id_t(credHandle)
	usage := C.gss_cred_usage_t(credUsage)
	mech := oidToCOid(desiredMech)
	var major, minor, overwrite, def C.OM_uint32
	var stored C.gss_OID_set

	if overwriteCred {
		overwrite = 1
	}
	if defCred {
		def = 1
	}

	major = C.gss_store_cred(&minor, handle, usage, mech, overwrite, def, &stored, &usage)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	elementsStored = coidSetToOids(stored)
	C.free_oid_set(stored)
	credUsageStored = uint32(usage)
	return
}

/* SetNegMechs() sets the list of mechanisms which will be negotiated when using credHandle with the SPNEGO mechanism ("1.3.6.1.5.5.2"). */
func SetNegMechs(credHandle CredHandle, mechSet []asn1.ObjectIdentifier) (majorStatus, minorStatus uint32) {
	handle := C.gss_cred_id_t(credHandle)
	mechs := oidsToCOidSet(mechSet)
	var major, minor C.OM_uint32

	major = C.gss_set_neg_mechs(&minor, handle, mechs)
	C.free_oid_set(mechs)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* IndicateMechsByAttrs() returns a list of security mechanisms, each of which matches at least one of the desiredMechAttrs, none of which match any of the exceptMechAttrs, and all of which match all of the criticalMechAttrs. */
func IndicateMechsByAttrs(desiredMechAttrs, exceptMechAttrs, criticalMechAttrs []asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, mechs []asn1.ObjectIdentifier) {
	desired := oidsToCOidSet(desiredMechAttrs)
	except := oidsToCOidSet(exceptMechAttrs)
	critical := oidsToCOidSet(criticalMechAttrs)
	var major, minor C.OM_uint32
	var selected C.gss_OID_set

	major = C.gss_indicate_mechs_by_attrs(&minor, desired, except, critical, &selected)
	C.free_oid_set(desired)
	C.free_oid_set(except)
	C.free_oid_set(critical)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	mechs = coidSetToOids(selected)
	C.free_oid_set(selected)
	return
}

/* Krb5ExtractAuthzDataFromSecContext() returns the raw bytes of a specific Kerberos auth-data type associated with the established security context's client. */
func Krb5ExtractAuthzDataFromSecContext(contextHandle ContextHandle, adType int) (majorStatus, minorStatus uint32, adData []byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	adtype := C.int(adType)
	var major, minor C.OM_uint32
	var addata C.gss_buffer_desc

	major = C.gsskrb5_extract_authz_data_from_sec_context(&minor, handle, adtype, &addata)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if addata.length > 0 {
		adData = bufferToBytes(addata)
	}
	return
}

/* Krb5RegisterAcceptorIdentity() sets the location of the keytab which will be used when acting as an acceptor using Kerberos 5 mechanisms. */
func Krb5RegisterAcceptorIdentity(identity string) uint32 {
	id := C.CString(identity)
	var ret C.OM_uint32

	ret = C.krb5_gss_register_acceptor_identity(id)
	C.free(unsafe.Pointer(id))
	return uint32(ret)
}

/* PNameToUid returns a numeric UID corresponding to the entity named by name. */
func PNameToUid(name InternalName, nmech asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, uid string) {
	iname := C.gss_name_t(name)
	mech := oidToCOid(nmech)
	var major, minor C.OM_uint32
	var id C.uid_t

	major = C.gss_pname_to_uid(&minor, iname, mech, &id)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	uid = ""
	if majorStatus == 0 {
		uid = fmt.Sprintf("%d", uint32(id))
	}
	return
}

/* Localname() returns the name of a local user who is considered to be the same entity as name. */
func Localname(name InternalName, mechType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, localName string) {
	iname := C.gss_name_t(name)
	mech := oidToCOid(mechType)
	var major, minor C.OM_uint32
	var lname C.gss_buffer_desc

	major = C.gss_localname(&major, iname, mech, &lname)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if lname.length > 0 {
		localName = bufferToString(lname)
		major = C.gss_release_buffer(&minor, &lname)
	}
	return
}

/* Userok() checks if the entity named by name is authorized to act as local user username. */
func Userok(name InternalName, username string) (ok bool) {
	iname := C.gss_name_t(name)
	lname := C.CString(username)
	var result C.int

	result = C.gss_userok(iname, lname)
	C.free(unsafe.Pointer(lname))

	ok = (result == 1)
	return
}

/* Userok() checks if the entity named by name is authorized to act as local user user. */
func AuthorizeLocalname(name, user InternalName) (majorStatus, minorStatus uint32) {
	iname := C.gss_name_t(name)
	uname := C.gss_name_t(user)
	var major, minor C.OM_uint32

	major = C.gss_authorize_localname(&minor, iname, uname)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* AcquireCredWithPassword() uses a password to obtain credentials to act as desiredName as an initiator, as an acceptor, or as both.  The returned credHandle should eventually be freed using gss.ReleaseCred(). */
func AcquireCredWithPassword(desiredName InternalName, password []byte, timeReq uint32, desiredMechs []asn1.ObjectIdentifier, credUsage uint32) (majorStatus, minorStatus uint32, credHandle CredHandle, actualMechs []asn1.ObjectIdentifier, timeRec uint32) {
	name := C.gss_name_t(desiredName)
	pwd := bytesToBuffer(password)
	time := C.OM_uint32(timeReq)
	dmechs := oidsToCOidSet(desiredMechs)
	usage := C.gss_cred_usage_t(credUsage)
	var major, minor C.OM_uint32
	var amechs C.gss_OID_set
	var handle C.gss_cred_id_t

	major = C.gss_acquire_cred_with_password(&minor, name, &pwd, time, dmechs, usage, &handle, &amechs, &time)
	C.free_oid_set(dmechs)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	credHandle = CredHandle(handle)
	actualMechs = coidSetToOids(amechs)
	C.free_oid_set(amechs)
	timeRec = uint32(time)
	return
}

/*
func AddCredWithPassword(icred CredHandle, desiredName InternalName, desiredMech asn1.ObjectIdentifier, password []byte, credUsage uint32, initiatorTimeReq, acceptorTimeReq uint32) (majorStatus, minorStatus uint32, ocred CredHandle, actualMechs []asn1.ObjectIdentifier, initiatorTimeRec, acceptorTimeRec uint32) {
	cred := C.gss_cred_id_t(icred)
	name := C.gss_name_t(desiredName)
	dmech := oidToCOid(desiredMech)
	pwd := bytesToBuffer(password)
	usage := C.gss_cred_usage_t(credUsage)
	itime := C.OM_uint32(initiatorTimeReq)
	atime := C.OM_uint32(AcceptorTimeReq)
	var major, minor C.OM_uint32
	var amechs C.gss_OID_set

	major = C.gss_add_cred_with_password(&minor, cred, name, dmech, &pwd, usage, itime, atime, &cred, &amechs, &itime, &atime)
	C.free_oid(dmech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	ocred = CredHandle(cred)
	actualMechs = coidSetToOids(amechs)
	C.free_oid_set(amechs)
	initiatorTimeRec = uint32(itime)
	acceptorTimeRec = uint32(atime)
	return
}
*/

func InquireSecContextByOid(contextHandle ContextHandle, desiredObject asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, dataSet [][]byte) {
	handle := C.gss_ctx_id_t(contextHandle)
	obj := oidToCOid(desiredObject)
	var major, minor C.OM_uint32
	var data C.gss_buffer_set_t

	major = C.gss_inquire_sec_context_by_oid(&minor, handle, obj, &data)
	C.free_oid(obj)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if data != nil {
		dataSet = buffersToBytes(*data)
		major = C.gss_release_buffer_set(&minor, &data)
	}
	return
}

func InquireCredByOid(credHandle CredHandle, desiredObject asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, dataSet [][]byte) {
	handle := C.gss_cred_id_t(credHandle)
	obj := oidToCOid(desiredObject)
	var major, minor C.OM_uint32
	var data C.gss_buffer_set_t

	major = C.gss_inquire_cred_by_oid(&minor, handle, obj, &data)
	C.free_oid(obj)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if data != nil {
		dataSet = buffersToBytes(*data)
		major = C.gss_release_buffer_set(&minor, &data)
	}
	return
}

func SetSecContextOption(contextHandle *ContextHandle, desiredObject asn1.ObjectIdentifier, value []byte) (majorStatus, minorStatus uint32) {
	handle := C.gss_ctx_id_t(*contextHandle)
	obj := oidToCOid(desiredObject)
	val := bytesToBuffer(value)
	var major, minor C.OM_uint32

	major = C.gss_set_sec_context_option(&minor, &handle, obj, &val)
	C.free_oid(obj)

	*contextHandle = ContextHandle(handle)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

func SetCredOption(credHandle *CredHandle, desiredObject asn1.ObjectIdentifier, value []byte) (majorStatus, minorStatus uint32) {
	handle := C.gss_cred_id_t(*credHandle)
	obj := oidToCOid(desiredObject)
	val := bytesToBuffer(value)
	var major, minor C.OM_uint32

	major = C.gss_set_cred_option(&minor, &handle, obj, &val)
	C.free_oid(obj)

	*credHandle = CredHandle(handle)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

func MechInvoke(desiredMech, desiredObject asn1.ObjectIdentifier, value *[]byte) (majorStatus, minorStatus uint32) {
	mech := oidToCOid(desiredMech)
	obj := oidToCOid(desiredObject)
	val := bytesToBuffer(*value)
	var major, minor C.OM_uint32

	major = C.gssspi_mech_invoke(&minor, mech, obj, &val)
	C.free_oid(mech)
	C.free_oid(obj)

	*value = bufferToBytes(val)
	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

func CompleteAuthToken(contextHandle ContextHandle, inputMessage []byte) (majorStatus, minorStatus uint32) {
	handle := C.gss_ctx_id_t(contextHandle)
	msg := bytesToBuffer(inputMessage)
	var major, minor C.OM_uint32

	major = C.gss_complete_auth_token(&minor, handle, &msg)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	return
}

/* AcquireCredImpersonateName() uses impersonatorCredHandle to acquire credentials which can be used to impersonate desiredName and returns a new outputCredHandle. */
func AcquireCredImpersonateName(impersonatorCredHandle CredHandle, desiredName InternalName, timeReq uint32, desiredMechs []asn1.ObjectIdentifier, credUsage uint32) (majorStatus, minorStatus uint32, outputCredHandle CredHandle, actualMechs []asn1.ObjectIdentifier, timeRec uint32) {
	cred := C.gss_cred_id_t(impersonatorCredHandle)
	name := C.gss_name_t(desiredName)
	time := C.OM_uint32(timeReq)
	dmechs := oidsToCOidSet(desiredMechs)
	usage := C.gss_cred_usage_t(credUsage)
	var major, minor C.OM_uint32
	var amechs C.gss_OID_set

	major = C.gss_acquire_cred_impersonate_name(&minor, cred, name, time, dmechs, usage, &cred, &amechs, &time)
	C.free_oid_set(dmechs)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandle = CredHandle(cred)
	actualMechs = coidSetToOids(amechs)
	C.free_oid_set(amechs)
	timeRec = uint32(time)
	return
}

/* AddCredImpersonateName() uses impersonatorCredHandle to acquire credentials which can be used to impersonate desiredName, merging them with outputCredHandle (if non-nil), or creating an entirely new credential handle, returning them in outputCredHandleRec. */
func AddCredImpersonateName(inputCredHandle, impersonatorCredHandle CredHandle, desiredName InternalName, desiredMech asn1.ObjectIdentifier, credUsage, initiatorTimeReq, acceptorTimeReq uint32, outputCredHandle CredHandle) (majorStatus, minorStatus uint32, outputCredHandleRec CredHandle, actualMechs []asn1.ObjectIdentifier, initiatorTimeRec, acceptorTimeRec uint32) {
	cred := C.gss_cred_id_t(inputCredHandle)
	icred := C.gss_cred_id_t(impersonatorCredHandle)
	ocred := C.gss_cred_id_t(outputCredHandle)
	name := C.gss_name_t(desiredName)
	mech := oidToCOid(desiredMech)
	usage := C.gss_cred_usage_t(credUsage)
	itime := C.OM_uint32(initiatorTimeReq)
	atime := C.OM_uint32(acceptorTimeReq)
	var major, minor C.OM_uint32
	var amechs C.gss_OID_set

	major = C.gss_add_cred_impersonate_name(&minor, cred, icred, name, mech, usage, itime, atime, &ocred, &amechs, &itime, &atime)
	C.free_oid(mech)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandleRec = CredHandle(ocred)
	actualMechs = coidSetToOids(amechs)
	C.free_oid_set(amechs)
	initiatorTimeRec = uint32(itime)
	acceptorTimeRec = uint32(atime)
	return
}

func DisplayNameExt(name InternalName, displayAsNameType asn1.ObjectIdentifier) (majorStatus, minorStatus uint32, displayName string) {
	iname := C.gss_name_t(name)
	ntype := oidToCOid(displayAsNameType)
	var major, minor C.OM_uint32
	var dname C.gss_buffer_desc

	major = C.gss_display_name_ext(&minor, iname, ntype, &dname)
	C.free_oid(ntype)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if dname.length > 0 {
		displayName = bufferToString(dname)
		major = C.gss_release_buffer(&minor, &dname)
	}
	return
}

/* InquireName() returns a list of attributes which are known about name. */
func InquireName(name InternalName) (majorStatus, minorStatus uint32, nameIsMN bool, mnMech asn1.ObjectIdentifier, attrs []string) {
	iname := C.gss_name_t(name)
	var major, minor C.OM_uint32
	var ismn C.int
	var oid C.gss_OID
	var buffers C.gss_buffer_set_t

	major = C.gss_inquire_name(&minor, iname, &ismn, &oid, &buffers)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	nameIsMN = (ismn != 0)
	if oid.length > 0 {
		mnMech = coidToOid(*oid)
		major = C.gss_release_oid(&minor, &oid)
	}
	if buffers != nil {
		attrs = buffersToStrings(*buffers)
		major = C.gss_release_buffer_set(&minor, &buffers)
	}
	return
}

/* GetNameAttribute() returns a value for the named attribute which is known about name.  When called for the first time, more should be set to -1.  When the last value of the attribute is returned, more will be set to 0. */
func GetNameAttribute(name InternalName, attr string, more *int) (majorStatus, minorStatus uint32, authenticated, complete bool, value []byte, displayValue string) {
	iname := C.gss_name_t(name)
	abuffer := stringToBuffer(attr)
	moar := C.int(*more)
	var major, minor C.OM_uint32
	var auth, comp C.int
	var val, dval C.gss_buffer_desc

	major = C.gss_get_name_attribute(&minor, iname, &abuffer, &auth, &comp, &val, &dval, &moar)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	major = C.gss_release_buffer(&minor, &abuffer)
	authenticated = (auth != 0)
	complete = (comp != 0)
	if val.length > 0 {
		value = bufferToBytes(val)
		major = C.gss_release_buffer(&minor, &val)
	}
	if dval.length > 0 {
		displayValue = bufferToString(dval)
		major = C.gss_release_buffer(&minor, &dval)
	}
	*more = int(moar)
	return
}

/* SetNameAttribute() adds a named attribute value for name. */
func SetNameAttribute(name InternalName, complete bool, attribute string, value []byte) (majorStatus, minorStatus uint32) {
	iname := C.gss_name_t(name)
	var comp C.int
	attr := stringToBuffer(attribute)
	val := bytesToBuffer(value)
	var major, minor C.OM_uint32

	if complete {
		comp = 1
	}
	major = C.gss_set_name_attribute(&minor, iname, comp, &attr, &val)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	major = C.gss_release_buffer(&minor, &attr)
	return
}

/* DeleteNameAttribute() removes a named attribute for name. */
func DeleteNameAttribute(name InternalName, attribute string) (majorStatus, minorStatus uint32) {
	iname := C.gss_name_t(name)
	attr := stringToBuffer(attribute)
	var major, minor C.OM_uint32

	major = C.gss_delete_name_attribute(&minor, iname, &attr)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	major = C.gss_release_buffer(&minor, &attr)
	return
}

func ExportNameComposite(name InternalName) (majorStatus, minorStatus uint32, compositeName []byte) {
	iname := C.gss_name_t(name)
	var major, minor C.OM_uint32
	var cname C.gss_buffer_desc

	major = C.gss_export_name_composite(&minor, iname, &cname)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if cname.length > 0 {
		compositeName = bufferToBytes(cname)
		major = C.gss_release_buffer(&minor, &cname)
	}
	return
}

/* AcquireCredFrom() obtains credentials to be used to either initiate or accept (or both) a security context as desiredName using information pointed to by the credStore.  The returned outputCredHandle should be released using gss.ReleaseCred() when it's no longer needed. */
func AcquireCredFrom(desiredName InternalName, timeReq uint32, desiredMechs []asn1.ObjectIdentifier, desiredCredUsage uint32, credStore [][2]string) (majorStatus, minorStatus uint32, outputCredHandle CredHandle, actualMechs []asn1.ObjectIdentifier, timeRec uint32) {
	name := C.gss_name_t(desiredName)
	time := C.OM_uint32(timeReq)
	dmechs := oidsToCOidSet(desiredMechs)
	usage := C.gss_cred_usage_t(desiredCredUsage)
	kvset := credStoreToKVSet(credStore)
	var major, minor C.OM_uint32
	var cred C.gss_cred_id_t
	var amechs C.gss_OID_set

	major = C.gss_acquire_cred_from(&minor, name, time, dmechs, usage, &kvset, &cred, &amechs, &time)
	C.free_oid_set(dmechs)
	C.free_kv_set(kvset)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandle = CredHandle(cred)
	actualMechs = coidSetToOids(amechs)
	C.free_oid_set(amechs)
	timeRec = uint32(time)
	return
}

/* AddCredFrom() obtains credentials specific to a particular mechanism using information pointed to by credStore, optionally merging them with already-obtained credentials (if outputCredHandle is not nil) or storing them in a new credential handle which should eventually be freed using gss.ReleaseCred(). */
func AddCredFrom(inputCredHandle CredHandle, desiredName InternalName, desiredMech asn1.ObjectIdentifier, desiredCredUsage, initiatorTimeReq, acceptorTimeReq uint32, outputCredHandle CredHandle, credStore [][2]string) (majorStatus, minorStatus uint32, outputCredHandleRec CredHandle, actualMechs []asn1.ObjectIdentifier, initiatorTimeRec, acceptorTimeRec uint32) {
	icred := C.gss_cred_id_t(inputCredHandle)
	ocred := C.gss_cred_id_t(outputCredHandle)
	name := C.gss_name_t(desiredName)
	mech := oidToCOid(desiredMech)
	usage := C.gss_cred_usage_t(desiredCredUsage)
	itime := C.OM_uint32(initiatorTimeReq)
	atime := C.OM_uint32(acceptorTimeReq)
	kvset := credStoreToKVSet(credStore)
	var major, minor C.OM_uint32
	var mechs C.gss_OID_set

	major = C.gss_add_cred_from(&minor, icred, name, mech, usage, itime, atime, &kvset, &ocred, &mechs, &itime, &atime)
	C.free_oid(mech)
	C.free_kv_set(kvset)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	outputCredHandleRec = CredHandle(ocred)
	actualMechs = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	initiatorTimeRec = uint32(itime)
	acceptorTimeRec = uint32(atime)
	return
}

/* StoreCredInto() stores non-nil credentials (for initiator, acceptor, or both) in locations pointed to by the credential store, or the default location if defaultCred is set. */
func StoreCredInto(inputCredHandle CredHandle, desiredCredUsage uint32, desiredMech asn1.ObjectIdentifier, overwriteCred, defaultCred bool, credStore [][2]string) (majorStatus, minorStatus uint32, elementsStored []asn1.ObjectIdentifier, credUsage uint32) {
	cred := C.gss_cred_id_t(inputCredHandle)
	usage := C.gss_cred_usage_t(desiredCredUsage)
	mech := oidToCOid(desiredMech)
	kvset := credStoreToKVSet(credStore)
	var major, minor, over, def C.OM_uint32
	var mechs C.gss_OID_set

	if overwriteCred {
		over = 1
	}
	if defaultCred {
		def = 1
	}

	major = C.gss_store_cred_into(&minor, cred, usage, mech, over, def, &kvset, &mechs, &usage)
	C.free_oid(mech)
	C.free_kv_set(kvset)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	elementsStored = coidSetToOids(mechs)
	C.free_oid_set(mechs)
	credUsage = uint32(usage)
	return
}

/* ExportCred() serializes the contents of the credential handle into a portable token.  The credHandle is not modified. */
func ExportCred(credHandle CredHandle) (majorStatus, minorStatus uint32, token []byte) {
	handle := C.gss_cred_id_t(credHandle)
	var major, minor C.OM_uint32
	var buffer C.gss_buffer_desc

	major = C.gss_export_cred(&minor, handle, &buffer)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	if buffer.length > 0 {
		token = bufferToBytes(buffer)
		major = C.gss_release_buffer(&minor, &buffer)
	}
	return
}

/* ImportCred() constructs a credential handle using the contents of the passed-in token.  The returned credHandle should eventually be freed using gss.ReleaseCred(). */
func ImportCred(token []byte) (majorStatus, minorStatus uint32, credHandle CredHandle) {
	buffer := bytesToBuffer(token)
	var major, minor C.OM_uint32
	var handle C.gss_cred_id_t

	major = C.gss_import_cred(&minor, &buffer, &handle)

	majorStatus = uint32(major)
	minorStatus = uint32(minor)
	credHandle = CredHandle(handle)
	return
}
