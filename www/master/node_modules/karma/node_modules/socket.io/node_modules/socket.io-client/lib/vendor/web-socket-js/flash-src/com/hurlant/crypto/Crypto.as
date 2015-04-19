/**
 * Crypto
 * 
 * An abstraction layer to instanciate our crypto algorithms
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto
{
	import com.hurlant.crypto.hash.HMAC;
	import com.hurlant.crypto.hash.MAC;
	import com.hurlant.crypto.hash.IHash;
	import com.hurlant.crypto.hash.MD2;
	import com.hurlant.crypto.hash.MD5;
	import com.hurlant.crypto.hash.SHA1;
	import com.hurlant.crypto.hash.SHA224;
	import com.hurlant.crypto.hash.SHA256;
	import com.hurlant.crypto.prng.ARC4;
	import com.hurlant.crypto.rsa.RSAKey;
	import com.hurlant.crypto.symmetric.AESKey;
	import com.hurlant.crypto.symmetric.BlowFishKey;
	import com.hurlant.crypto.symmetric.CBCMode;
	import com.hurlant.crypto.symmetric.CFB8Mode;
	import com.hurlant.crypto.symmetric.CFBMode;
	import com.hurlant.crypto.symmetric.CTRMode;
	import com.hurlant.crypto.symmetric.DESKey;
	import com.hurlant.crypto.symmetric.ECBMode;
	import com.hurlant.crypto.symmetric.ICipher;
	import com.hurlant.crypto.symmetric.IMode;
	import com.hurlant.crypto.symmetric.IPad;
	import com.hurlant.crypto.symmetric.ISymmetricKey;
	import com.hurlant.crypto.symmetric.IVMode;
	import com.hurlant.crypto.symmetric.NullPad;
	import com.hurlant.crypto.symmetric.OFBMode;
	import com.hurlant.crypto.symmetric.PKCS5;
	import com.hurlant.crypto.symmetric.SimpleIVMode;
	import com.hurlant.crypto.symmetric.TripleDESKey;
	import com.hurlant.crypto.symmetric.XTeaKey;
	import com.hurlant.util.Base64;
	
	import flash.utils.ByteArray;
	
	/**
	 * A class to make it easy to use the rest of the framework.
	 * As a side-effect, using this class will cause most of the framework
	 * to be linked into your application, which is not always what you want.
	 * 
	 * If you want to optimize your download size, don't use this class.
	 * (But feel free to read it to get ideas on how to get the algorithm you want.)
	 */
	public class Crypto
	{
		private var b64:Base64; // we don't use it, but we want the swc to include it, so cheap trick.
		
		public function Crypto(){
		}
		
		/**
		 * Things that should work, among others:
		 *  "aes"
		 *  "aes-128-ecb"
		 *  "aes-128-cbc"
		 *  "aes-128-cfb"
		 *  "aes-128-cfb8"
		 *  "aes-128-ofb"
		 *  "aes-192-cfb"
		 *  "aes-256-ofb"
		 *  "blowfish-cbc"
		 *  "des-ecb"
		 *  "xtea"
		 *  "xtea-ecb"
		 *  "xtea-cbc"
		 *  "xtea-cfb"
		 *  "xtea-cfb8"
		 *  "xtea-ofb"
		 *  "rc4"
		 *  "simple-aes-cbc"
		 */
		public static function getCipher(name:String, key:ByteArray, pad:IPad=null):ICipher {
			// split name into an array.
			var keys:Array = name.split("-");
			switch (keys[0]) {
				/**
				 * "simple" is a special case. It means:
				 * "If using an IV mode, prepend the IV to the ciphertext"
				 */
				case "simple":
					keys.shift();
					name = keys.join("-");
					var cipher:ICipher = getCipher(name, key, pad);
					if (cipher is IVMode) {
						return new SimpleIVMode(cipher as IVMode);
					} else {
						return cipher;
					}
				/**
				 * we support both "aes-128" and "aes128"
				 * Technically, you could use "aes192-128", but you'd
				 * only be hurting yourself.
				 */
				case "aes":
				case "aes128":
				case "aes192":
				case "aes256":
					keys.shift();
					if (key.length*8==keys[0]) {
						// support for "aes-128-..." and such.
						keys.shift();
					}
					return getMode(keys[0], new AESKey(key), pad);
				break;
				case "bf":
				case "blowfish":
					keys.shift();
					return getMode(keys[0], new BlowFishKey(key), pad);
				/**
				 * des-ede and des-ede3 are both equivalent to des3.
				 * the choice between 2tdes and 3tdes is made based
				 * on the length of the key provided.
				 */
				case "des":
					keys.shift();
					if (keys[0]!="ede" && keys[0]!="ede3") {
						return getMode(keys[0], new DESKey(key), pad);
					}
					if (keys.length==1) {
						keys.push("ecb"); // default mode for 2tdes and 3tdes with openssl enc
					}
					// fall-through to triple des
				case "3des":
				case "des3":
					keys.shift();
					return getMode(keys[0], new TripleDESKey(key), pad);
				case "xtea":
					keys.shift();
					return getMode(keys[0], new XTeaKey(key), pad);
				break;
				/**
				 * Technically, you could say "rc4-128" or whatever,
				 * but really, the length of the key is what counts here.
				 */
				case "rc4":
					keys.shift();
					return new ARC4(key);
				break;
			}
			return null;
		}
		
		/**
		 * Returns the size of a key for a given cipher identifier.
		 */
		public static function getKeySize(name:String):uint {
			var keys:Array = name.split("-");
			switch (keys[0]) {
				case "simple":
					keys.shift();
					return getKeySize(keys.join("-"));
				case "aes128":
					return 16;
				case "aes192":
					return 24;
				case "aes256":
					return 32;
				case "aes":
					keys.shift();
					return parseInt(keys[0])/8;
				case "bf":
				case "blowfish":
					return 16;
				case "des":
					keys.shift();
					switch (keys[0]) {
						case "ede":
							return 16;
						case "ede3":
							return 24;
						default:
							return 8;
					}
				case "3des":
				case "des3":
					return 24;
				case "xtea":
					return 8;
				case "rc4":
					if (parseInt(keys[1])>0) {
						return parseInt(keys[1])/8;
					}
					return 16; // why not.
			}
			return 0; // unknown;
		}
		
		private static function getMode(name:String, alg:ISymmetricKey, padding:IPad=null):IMode {
			switch (name) {
				case "ecb":
					return new ECBMode(alg, padding);
				case "cfb":
					return new CFBMode(alg, padding);
				case "cfb8":
					return new CFB8Mode(alg, padding);
				case "ofb":
					return new OFBMode(alg, padding);
				case "ctr":
					return new CTRMode(alg, padding);
				case "cbc":
				default:
					return new CBCMode(alg, padding);
			}
		}
		
		/**
		 * Things that should work:
		 * "md5"
		 * "sha"
		 * "sha1"
		 * "sha224"
		 * "sha256"
		 */
		public static function getHash(name:String):IHash {
			switch(name) {
				case "md2":
					return new MD2;
				case "md5":
					return new MD5;
				case "sha": // let's hope you didn't mean sha-0
				case "sha1":
					return new SHA1;
				case "sha224":
					return new SHA224;
				case "sha256":
					return new SHA256;
			}
			return null;
		}
		
		/**
		 * Things that should work:
		 * "sha1"
		 * "md5-64"
		 * "hmac-md5-96"
		 * "hmac-sha1-128"
		 * "hmac-sha256-192"
		 * etc.
		 */
		public static function getHMAC(name:String):HMAC {
			var keys:Array = name.split("-");
			if (keys[0]=="hmac") keys.shift();
			var bits:uint = 0;
			if (keys.length>1) {
				bits = parseInt(keys[1]);
			}
			return new HMAC(getHash(keys[0]), bits);
		}
		

		public static function getMAC(name:String):MAC {
			
			var keys:Array = name.split("-");
			if (keys[0]=="mac") keys.shift();
			var bits:uint = 0;
			if (keys.length > 1) {
				bits = parseInt(keys[1]);
			}
			return new MAC(getHash(keys[0]), bits);
		}
				
		
		public static function getPad(name:String):IPad {
			switch(name) {
				case "null":
					return new NullPad;
				case "pkcs5":
				default:
					return new PKCS5;
			}
		}
		
		/** mostly useless.
		 */
		public static function getRSA(E:String, M:String):RSAKey {
			return RSAKey.parsePublicKey(M,E);
		}
	}
}