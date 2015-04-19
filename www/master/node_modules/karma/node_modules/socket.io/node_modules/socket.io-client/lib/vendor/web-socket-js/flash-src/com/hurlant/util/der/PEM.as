/**
 * PEM
 * 
 * A class to parse some PEM stuff.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import com.hurlant.crypto.rsa.RSAKey;
	import com.hurlant.math.BigInteger;
	import com.hurlant.util.Base64;
	
	import flash.utils.ByteArray;
	import com.hurlant.util.Hex;
	
	public class PEM
	{
		private static const RSA_PRIVATE_KEY_HEADER:String = "-----BEGIN RSA PRIVATE KEY-----";
		private static const RSA_PRIVATE_KEY_FOOTER:String = "-----END RSA PRIVATE KEY-----";
		private static const RSA_PUBLIC_KEY_HEADER:String = "-----BEGIN PUBLIC KEY-----";
		private static const RSA_PUBLIC_KEY_FOOTER:String = "-----END PUBLIC KEY-----";
		private static const CERTIFICATE_HEADER:String = "-----BEGIN CERTIFICATE-----";
		private static const CERTIFICATE_FOOTER:String = "-----END CERTIFICATE-----";
		
		
		
		/**
		 * 
		 * Read a structure encoded according to
		 * ftp://ftp.rsasecurity.com/pub/pkcs/ascii/pkcs-1v2.asc
		 * section 11.1.2
		 * 
		 * @param str
		 * @return 
		 * 
		 */
		public static function readRSAPrivateKey(str:String):RSAKey {
			var der:ByteArray = extractBinary(RSA_PRIVATE_KEY_HEADER, RSA_PRIVATE_KEY_FOOTER, str);
			if (der==null) return null;
			var obj:* = DER.parse(der);
			if (obj is Array) {
				var arr:Array = obj as Array;
				// arr[0] is Version. should be 0. should be checked. shoulda woulda coulda.
				return new RSAKey(
					arr[1],				// N
					arr[2].valueOf(),	// E
					arr[3],				// D
					arr[4],				// P
					arr[5],				// Q
					arr[6],				// DMP1
					arr[7],				// DMQ1	
					arr[8]);			// IQMP
			} else {
				// dunno
				return null;
			}
		}
		
		
		/**
		 * Read a structure encoded according to some spec somewhere
		 * Also, follows some chunk from
		 * ftp://ftp.rsasecurity.com/pub/pkcs/ascii/pkcs-1v2.asc
		 * section 11.1
		 * 
		 * @param str
		 * @return 
		 * 
		 */
		public static function readRSAPublicKey(str:String):RSAKey {
			var der:ByteArray = extractBinary(RSA_PUBLIC_KEY_HEADER, RSA_PUBLIC_KEY_FOOTER, str);
			if (der==null) return null;
			var obj:* = DER.parse(der);
			if (obj is Array) {
				var arr:Array = obj as Array;
				// arr[0] = [ <some crap that means "rsaEncryption">, null ]; ( apparently, that's an X-509 Algorithm Identifier.
				if (arr[0][0].toString()!=OID.RSA_ENCRYPTION) {
					return null;
				}
				// arr[1] is a ByteArray begging to be parsed as DER
				arr[1].position = 1; // there's a 0x00 byte up front. find out why later. like, read a spec.
				obj = DER.parse(arr[1]);
				if (obj is Array) {
					arr = obj as Array;
					// arr[0] = modulus
					// arr[1] = public expt.
					return new RSAKey(arr[0], arr[1]);
				} else {
					return null;
				}
			} else {
				// dunno
				return null;
			}
		}

		public static function readCertIntoArray(str:String):ByteArray {
			var tmp:ByteArray = extractBinary(CERTIFICATE_HEADER, CERTIFICATE_FOOTER, str);
			return tmp;
		}
		
		private static function extractBinary(header:String, footer:String, str:String):ByteArray {
			var i:int = str.indexOf(header);
			if (i==-1) return null;
			i += header.length;
			var j:int = str.indexOf(footer);
			if (j==-1) return null;
			var b64:String = str.substring(i, j);
			// remove whitesapces.
			b64 = b64.replace(/\s/mg, '');
			// decode
			return Base64.decodeToByteArray(b64);
		}
		
	}
}