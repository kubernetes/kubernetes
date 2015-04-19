/**
 * X509Certificate
 * 
 * A representation for a X509 Certificate, with
 * methods to parse, verify and sign it.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.cert {
	import com.hurlant.crypto.hash.IHash;
	import com.hurlant.crypto.hash.MD2;
	import com.hurlant.crypto.hash.MD5;
	import com.hurlant.crypto.hash.SHA1;
	import com.hurlant.crypto.rsa.RSAKey;
	import com.hurlant.util.ArrayUtil;
	import com.hurlant.util.Base64;
	import com.hurlant.util.der.ByteString;
	import com.hurlant.util.der.DER;
	import com.hurlant.util.der.OID;
	import com.hurlant.util.der.ObjectIdentifier;
	import com.hurlant.util.der.PEM;
	import com.hurlant.util.der.PrintableString;
	import com.hurlant.util.der.Sequence;
	import com.hurlant.util.der.Type;
	
	import flash.utils.ByteArray;
	
	public class X509Certificate {
		private var _loaded:Boolean;
		private var _param:*;
		private var _obj:Object;
		public function X509Certificate(p:*) {
			_loaded = false;
			_param = p;
			// lazy initialization, to avoid unnecessary parsing of every builtin CA at start-up.
		}
		private function load():void {
			if (_loaded) return;
			var p:* = _param;
			var b:ByteArray;
			if (p is String) {
				b = PEM.readCertIntoArray(p as String);
			} else if (p is ByteArray) {
				b = p;
			}
			if (b!=null) {
				_obj = DER.parse(b, Type.TLS_CERT);
				_loaded = true;
			} else {
				throw new Error("Invalid x509 Certificate parameter: "+p);
			}
		}
		public function isSigned(store:X509CertificateCollection, CAs:X509CertificateCollection, time:Date=null):Boolean {
			load();
			// check timestamps first. cheapest.
			if (time==null) {
				time = new Date;
			}
			var notBefore:Date = getNotBefore();
			var notAfter:Date = getNotAfter();
			if (time.getTime()<notBefore.getTime()) return false; // cert isn't born yet.
			if (time.getTime()>notAfter.getTime()) return false;  // cert died of old age.
			// check signature.
			var subject:String = getIssuerPrincipal();
			// try from CA first, since they're treated better.
			var parent:X509Certificate = CAs.getCertificate(subject);
			var parentIsAuthoritative:Boolean = false;
			if (parent == null) {
				parent = store.getCertificate(subject);
				if (parent == null) {
					return false; // issuer not found
				}
			} else {
				parentIsAuthoritative = true;
			}
			if (parent == this) { // pathological case. avoid infinite loop
				return false; // isSigned() returns false if we're self-signed.
			}
			if (!(parentIsAuthoritative&&parent.isSelfSigned(time)) &&
				!parent.isSigned(store, CAs, time)) {
				return false;
			}
			var key:RSAKey = parent.getPublicKey();
			return verifyCertificate(key);
		}
		public function isSelfSigned(time:Date):Boolean {
			load();
			
			var key:RSAKey = getPublicKey();
			return verifyCertificate(key);
		}
		private function verifyCertificate(key:RSAKey):Boolean {
			var algo:String = getAlgorithmIdentifier();
			var hash:IHash;
			var oid:String;
			switch (algo) {
				case OID.SHA1_WITH_RSA_ENCRYPTION:
					hash = new SHA1;
					oid = OID.SHA1_ALGORITHM;
					break;
				case OID.MD2_WITH_RSA_ENCRYPTION:
					hash = new MD2;
					oid = OID.MD2_ALGORITHM;
					break;
				case OID.MD5_WITH_RSA_ENCRYPTION:
					hash = new MD5;
					oid = OID.MD5_ALGORITHM;
					break;
				default:
					return false;
			}
			var data:ByteArray = _obj.signedCertificate_bin;
			var buf:ByteArray = new ByteArray;
			key.verify(_obj.encrypted, buf, _obj.encrypted.length);
			buf.position=0;
			data = hash.hash(data);
			var obj:Object = DER.parse(buf, Type.RSA_SIGNATURE);
			if (obj.algorithm.algorithmId.toString() != oid) {
				return false; // wrong algorithm
			}
			if (!ArrayUtil.equals(obj.hash, data)) {
				return false; // hashes don't match
			}
			return true;
		}
		
		/**
		 * This isn't used anywhere so far.
		 * It would become useful if we started to offer facilities
		 * to generate and sign X509 certificates.
		 * 
		 * @param key
		 * @param algo
		 * @return 
		 * 
		 */
		private function signCertificate(key:RSAKey, algo:String):ByteArray {
			var hash:IHash;
			var oid:String;
			switch (algo) {
				case OID.SHA1_WITH_RSA_ENCRYPTION:
					hash = new SHA1;
					oid = OID.SHA1_ALGORITHM;
					break;
				case OID.MD2_WITH_RSA_ENCRYPTION:
					hash = new MD2;
					oid = OID.MD2_ALGORITHM;
					break;
				case OID.MD5_WITH_RSA_ENCRYPTION:
					hash = new MD5;
					oid = OID.MD5_ALGORITHM;
					break;
				default:
					return null
			}
			var data:ByteArray = _obj.signedCertificate_bin;
			data = hash.hash(data);
			var seq1:Sequence = new Sequence;
			seq1[0] = new Sequence;
			seq1[0][0] = new ObjectIdentifier(0,0, oid);
			seq1[0][1] = null;
			seq1[1] = new ByteString;
			seq1[1].writeBytes(data);
			data = seq1.toDER();
			var buf:ByteArray = new ByteArray;
			key.sign(data, buf, data.length);
			return buf;
		}
		
		public function getPublicKey():RSAKey {
			load();
			var pk:ByteArray = _obj.signedCertificate.subjectPublicKeyInfo.subjectPublicKey as ByteArray;
			pk.position = 0;
			var rsaKey:Object = DER.parse(pk, [{name:"N"},{name:"E"}]);
			return new RSAKey(rsaKey.N, rsaKey.E.valueOf());
		}
		
		/**
		 * Returns a subject principal, as an opaque base64 string.
		 * This is only used as a hash key for known certificates.
		 * 
		 * Note that this assumes X509 DER-encoded certificates are uniquely encoded,
		 * as we look for exact matches between Issuer and Subject fields.
		 * 
		 */
		public function getSubjectPrincipal():String {
			load();
			return Base64.encodeByteArray(_obj.signedCertificate.subject_bin);
		}
		/**
		 * Returns an issuer principal, as an opaque base64 string.
		 * This is only used to quickly find matching parent certificates.
		 * 
		 * Note that this assumes X509 DER-encoded certificates are uniquely encoded,
		 * as we look for exact matches between Issuer and Subject fields.
		 * 
		 */
		public function getIssuerPrincipal():String {
			load();
			return Base64.encodeByteArray(_obj.signedCertificate.issuer_bin);
		}
		public function getAlgorithmIdentifier():String {
			return _obj.algorithmIdentifier.algorithmId.toString();
		}
		public function getNotBefore():Date {
			return _obj.signedCertificate.validity.notBefore.date;
		}
		public function getNotAfter():Date {
			return _obj.signedCertificate.validity.notAfter.date;
		}
		
		public function getCommonName():String {
			var subject:Sequence = _obj.signedCertificate.subject;
			return (subject.findAttributeValue(OID.COMMON_NAME) as PrintableString).getString();
		}
	}
}