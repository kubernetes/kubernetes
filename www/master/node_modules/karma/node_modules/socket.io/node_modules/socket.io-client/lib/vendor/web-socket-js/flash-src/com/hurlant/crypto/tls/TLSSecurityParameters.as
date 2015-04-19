/**
 * TLSSecurityParameters
 * 
 * This class encapsulates all the security parameters that get negotiated
 * during the TLS handshake. It also holds all the key derivation methods.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * Patched by Bobby Parker (sh0rtwave@gmail.com)
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import com.hurlant.crypto.hash.MD5;
	import com.hurlant.crypto.hash.SHA1;
	import com.hurlant.crypto.prng.TLSPRF;
	import com.hurlant.util.Hex;
	
	import flash.utils.ByteArray;
	import com.hurlant.crypto.rsa.RSAKey;
	
	public class TLSSecurityParameters implements ISecurityParameters {
		
		// COMPRESSION
		public static const COMPRESSION_NULL:uint = 0;
		
		// This is probably not smart. Revise this to use all settings from TLSConfig, since this shouldn't really know about
		// user settings, those are best handled from the engine at a session level.
		public static var IGNORE_CN_MISMATCH:Boolean = true;
		public static var ENABLE_USER_CLIENT_CERTIFICATE:Boolean = false;
		public static var USER_CERTIFICATE:String;
		
		
		private var cert:ByteArray; // Local Cert
		private var key:RSAKey; // local key
		private var entity:uint; // SERVER | CLIENT
		private var bulkCipher:uint; // BULK_CIPHER_*
		private var cipherType:uint; // STREAM_CIPHER | BLOCK_CIPHER
		private var keySize:uint;
		private var keyMaterialLength:uint;
		private var IVSize:uint;
		private var macAlgorithm:uint; // MAC_*
		private var hashSize:uint;
		private var compression:uint; // COMPRESSION_NULL
		private var masterSecret:ByteArray; // 48 bytes
		private var clientRandom:ByteArray; // 32 bytes
		private var serverRandom:ByteArray; // 32 bytes
		private var ignoreCNMismatch:Boolean = true;
		private var trustAllCerts:Boolean = false;
		private var trustSelfSigned:Boolean = false;
		public static const PROTOCOL_VERSION:uint = 0x0301; 
		private var tlsDebug:Boolean = false;

		
		// not strictly speaking part of this, but yeah.
		public var keyExchange:uint;
		public function TLSSecurityParameters(entity:uint, localCert:ByteArray = null, localKey:RSAKey = null) {
			this.entity = entity;
			reset();
			key = localKey;
			cert = localCert;
		}
		
		public function get version() : uint {
			return PROTOCOL_VERSION;
		}
		
		public function reset():void {
			bulkCipher = BulkCiphers.NULL;
			cipherType = BulkCiphers.BLOCK_CIPHER;
			macAlgorithm = MACs.NULL;
			compression = COMPRESSION_NULL;
			masterSecret = null;
		}
		
		public function getBulkCipher():uint {
			return bulkCipher;
		}
		public function getCipherType():uint {
			return cipherType;
		}
		public function getMacAlgorithm():uint {
			return macAlgorithm;
		}
		
		public function setCipher(cipher:uint):void {
			bulkCipher = CipherSuites.getBulkCipher(cipher);
			cipherType = BulkCiphers.getType(bulkCipher);
			keySize = BulkCiphers.getExpandedKeyBytes(bulkCipher);   // 8
			keyMaterialLength = BulkCiphers.getKeyBytes(bulkCipher); // 5
			IVSize = BulkCiphers.getIVSize(bulkCipher);
			
			keyExchange = CipherSuites.getKeyExchange(cipher);
			
			macAlgorithm = CipherSuites.getMac(cipher);
			hashSize = MACs.getHashSize(macAlgorithm);
		}
		public function setCompression(algo:uint):void {
			compression = algo;
		}
		public function setPreMasterSecret(secret:ByteArray):void {
			// compute master_secret
			var seed:ByteArray = new ByteArray;
			seed.writeBytes(clientRandom, 0, clientRandom.length);
			seed.writeBytes(serverRandom, 0, serverRandom.length);
			var prf:TLSPRF = new TLSPRF(secret, "master secret", seed);
			masterSecret = new ByteArray;
			prf.nextBytes(masterSecret, 48);
			if (tlsDebug)
				trace("Master Secret: " + Hex.fromArray( masterSecret, true ));
		}
		public function setClientRandom(secret:ByteArray):void {
			clientRandom = secret;
		}
		public function setServerRandom(secret:ByteArray):void { 
			serverRandom = secret;
		}
		
		public function get useRSA():Boolean {
			return KeyExchanges.useRSA(keyExchange);
		}
		
		public function computeVerifyData(side:uint, handshakeMessages:ByteArray):ByteArray {
			var seed:ByteArray = new ByteArray;
			var md5:MD5 = new MD5;
			if (tlsDebug)
				trace("Handshake value: " + Hex.fromArray(handshakeMessages, true ));
			seed.writeBytes(md5.hash(handshakeMessages),0,md5.getHashSize());
			var sha:SHA1 = new SHA1;
			seed.writeBytes(sha.hash(handshakeMessages),0,sha.getHashSize());
			if (tlsDebug)
				trace("Seed in: " + Hex.fromArray(seed, true ));
			var prf:TLSPRF = new TLSPRF(masterSecret, (side==TLSEngine.CLIENT) ? "client finished" : "server finished", seed);
			var out:ByteArray = new ByteArray;
			prf.nextBytes(out, 12);
			if (tlsDebug)
				trace("Finished out: " + Hex.fromArray(out, true ));
			out.position = 0;
			return out;
		}
		
		// client side certficate check - This is probably incorrect somehow
		public function computeCertificateVerify( side:uint, handshakeMessages:ByteArray ):ByteArray {
			var seed:ByteArray = new ByteArray;
			var md5:MD5 = new MD5;
			seed.writeBytes(md5.hash(handshakeMessages),0,md5.getHashSize());
			var sha:SHA1 = new SHA1;
			seed.writeBytes(sha.hash(handshakeMessages),0,sha.getHashSize());
			
			// Now that I have my hashes of existing handshake messages (which I'm not sure about the length of yet) then 
			// Sign that with my private key
			seed.position = 0;
			var out:ByteArray = new ByteArray();
			key.sign( seed, out, seed.bytesAvailable);
			out.position = 0;
			return out;	
		}
		
		public function getConnectionStates():Object {
			if (masterSecret != null) {
				var seed:ByteArray = new ByteArray;
				seed.writeBytes(serverRandom, 0, serverRandom.length);
				seed.writeBytes(clientRandom, 0, clientRandom.length);
				var prf:TLSPRF = new TLSPRF(masterSecret, "key expansion", seed);
				
				var client_write_MAC:ByteArray = new ByteArray;
				prf.nextBytes(client_write_MAC, hashSize);
				var server_write_MAC:ByteArray = new ByteArray;
				prf.nextBytes(server_write_MAC, hashSize);
				var client_write_key:ByteArray = new ByteArray;
				prf.nextBytes(client_write_key, keyMaterialLength);
				var server_write_key:ByteArray = new ByteArray;
				prf.nextBytes(server_write_key, keyMaterialLength);
				var client_write_IV:ByteArray = new ByteArray;
				prf.nextBytes(client_write_IV, IVSize);
				var server_write_IV:ByteArray = new ByteArray;
				prf.nextBytes(server_write_IV, IVSize);

				var client_write:TLSConnectionState = new TLSConnectionState(
						bulkCipher, cipherType, macAlgorithm,
						client_write_MAC, client_write_key, client_write_IV);
				var server_write:TLSConnectionState = new TLSConnectionState(
						bulkCipher, cipherType, macAlgorithm,
						server_write_MAC, server_write_key, server_write_IV);
				
				if (entity == TLSEngine.CLIENT) {
					return {read:server_write, write:client_write};
				} else {
					return {read:client_write, write:server_write};
				}

			} else {
				return {read:new TLSConnectionState, write:new TLSConnectionState};
			}
		}
		
	}
}