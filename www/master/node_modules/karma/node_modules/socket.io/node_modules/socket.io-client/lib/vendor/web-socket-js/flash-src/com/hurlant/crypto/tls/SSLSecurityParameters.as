/**
 * TLSSecurityParameters
 * 
 * This class encapsulates all the security parameters that get negotiated
 * during the TLS handshake. It also holds all the key derivation methods.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import com.hurlant.crypto.hash.MD5;
	import com.hurlant.crypto.hash.SHA1;
	import com.hurlant.util.Hex;
	
	import flash.utils.ByteArray;
	
	public class SSLSecurityParameters implements ISecurityParameters {
		
		// COMPRESSION
		public static const COMPRESSION_NULL:uint = 0;
		
		private var entity:uint; // SERVER | CLIENT
		private var bulkCipher:uint; // BULK_CIPHER_*
		private var cipherType:uint; // STREAM_CIPHER | BLOCK_CIPHER
		private var keySize:uint;
		private var keyMaterialLength:uint;
		private var keyBlock:ByteArray;
		private var IVSize:uint;
		private var MAC_length:uint;
		private var macAlgorithm:uint; // MAC_*
		private var hashSize:uint;
		private var compression:uint; // COMPRESSION_NULL
		private var masterSecret:ByteArray; // 48 bytes
		private var clientRandom:ByteArray; // 32 bytes
		private var serverRandom:ByteArray; // 32 bytes
		private var pad_1:ByteArray; // varies
		private var pad_2:ByteArray; // varies
		private var ignoreCNMismatch:Boolean = true;
		private var trustAllCerts:Boolean = false;
		private var trustSelfSigned:Boolean = false;
		public static const PROTOCOL_VERSION:uint = 0x0300;
		
		// not strictly speaking part of this, but yeah.
		public var keyExchange:uint;
		
		public function get version() : uint { 
			return PROTOCOL_VERSION;
		}
		public function SSLSecurityParameters(entity:uint, localCert:ByteArray = null, localKey:ByteArray = null) {
			this.entity = entity;
			reset();
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
			pad_1 = new ByteArray();
			pad_2 = new ByteArray();
			for (var x:int = 0; x < 48; x++) {
				pad_1.writeByte(0x36);
				pad_2.writeByte(0x5c);
			}			
		}
		public function setCompression(algo:uint):void {
			compression = algo;
		}
		
		public function setPreMasterSecret(secret:ByteArray):void {
			/* Warning! Following code may cause madness
				 Tread not here, unless ye be men of valor.
			
			***** Official Prophylactic Comment ******
				(to protect the unwary...this code actually works, that's all you need to know)
			
			This does two things, computes the master secret, and generates the keyBlock
			
			
			To compute the master_secret, the following algorithm is used.
			 for SSL 3, this means
			 master = MD5( premaster + SHA1('A' + premaster + client_random + server_random ) ) +
						MD5( premaster + SHA1('BB' + premaster + client_random + server_random ) ) +
						MD5( premaster + SHA1('CCC' + premaster + client_random + server_random ) )
			*/		
			var tempHashA:ByteArray = new ByteArray(); // temporary hash, gets reused a lot
			var tempHashB:ByteArray = new ByteArray(); // temporary hash, gets reused a lot
			
			var shaHash:ByteArray;
			var mdHash:ByteArray;
			
			var i:int;
			var j:int;
			
			var sha:SHA1 = new SHA1();
			var md:MD5 = new MD5();
					
			var k:ByteArray = new ByteArray();
			
			k.writeBytes(secret);
			k.writeBytes(clientRandom);
			k.writeBytes(serverRandom);
			
			masterSecret = new ByteArray();
			var pad_char:uint = 0x41;
			
			for ( i = 0; i < 3; i++) {
				// SHA portion
				tempHashA.position = 0;
								
				for ( j = 0; j < i + 1; j++) {
					tempHashA.writeByte(pad_char);
				}
				pad_char++;
				
				tempHashA.writeBytes(k);
				shaHash = sha.hash(tempHashA);
				
				// MD5 portion
				tempHashB.position = 0;
				tempHashB.writeBytes(secret); 
				tempHashB.writeBytes(shaHash); 
				mdHash = md.hash(tempHashB);
				
				// copy into my key
				masterSecret.writeBytes(mdHash);
			}
			
			// *************** END MASTER SECRET **************
			
			// More prophylactic comments
			// *************** START KEY BLOCK ****************
			
			// So here, I'm setting up the keyBlock array that I will derive MACs, keys, and IVs from.
			// Rebuild k (hash seed)
			 
			k.position = 0; 
			k.writeBytes(masterSecret);
			k.writeBytes(serverRandom);
			k.writeBytes(clientRandom);
			
			keyBlock = new ByteArray(); 
			
			tempHashA = new ByteArray();
			tempHashB = new ByteArray();
			// now for 16 iterations to get 256 bytes (16 * 16), better to have more than not enough
			pad_char = 0x41;
			for ( i = 0; i < 16; i++) {
				tempHashA.position = 0; 
				
				for ( j = 0; j < i + 1; j++) {
					tempHashA.writeByte(pad_char);
				}
				pad_char++;
				tempHashA.writeBytes(k);
				shaHash = sha.hash(tempHashA);	
				
				tempHashB.position = 0; 
				tempHashB.writeBytes(masterSecret);
				tempHashB.writeBytes(shaHash, 0);
				mdHash = md.hash(tempHashB);
				
				keyBlock.writeBytes(mdHash); 
			}
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
		
		// This is the Finished message
		// if you value your sanity, stay away...far away
		public function computeVerifyData(side:uint, handshakeMessages:ByteArray):ByteArray {
			// for SSL 3.0, this consists of
			// 	finished = md5( masterSecret + pad2 + md5( handshake + sender + masterSecret + pad1 ) ) +
			//			   sha1( masterSecret + pad2 + sha1( handshake + sender + masterSecret + pad1 ) )
			
			// trace("Handshake messages: " + Hex.fromArray(handshakeMessages));
			var sha:SHA1 = new SHA1();
			var md:MD5 = new MD5();
			var k:ByteArray = new ByteArray(); // handshake + sender + masterSecret + pad1
			var j:ByteArray = new ByteArray(); // masterSecret + pad2 + k
			
			var innerKey:ByteArray;
			var outerKey:ByteArray = new ByteArray();
			
			var hashSha:ByteArray;
			var hashMD:ByteArray;
			
			var sideBytes:ByteArray = new ByteArray();
			if (side == TLSEngine.CLIENT) {
			 	sideBytes.writeUnsignedInt(0x434C4E54);
			 } else {
				sideBytes.writeUnsignedInt(0x53525652);
			}
			
			// Do the SHA1 part of the routine first
			masterSecret.position = 0;
			k.writeBytes(handshakeMessages);
			k.writeBytes(sideBytes);
			k.writeBytes(masterSecret);
			k.writeBytes(pad_1, 0, 40); // limited to 40 chars for SHA1
				
			innerKey = sha.hash(k);
			// trace("Inner SHA Key: " + Hex.fromArray(innerKey));
			
			j.writeBytes(masterSecret);
			j.writeBytes(pad_2, 0, 40); // limited to 40 chars for SHA1
			j.writeBytes(innerKey);
			
			hashSha = sha.hash(j);
			// trace("Outer SHA Key: " + Hex.fromArray(hashSha));
			
			// Rebuild k for MD5
			k = new ByteArray();
			
			k.writeBytes(handshakeMessages);
			k.writeBytes(sideBytes);
			k.writeBytes(masterSecret);
			k.writeBytes(pad_1); // Take the whole length of pad_1 & pad_2 for MD5
			
			innerKey = md.hash(k);
			// trace("Inner MD5 Key: " + Hex.fromArray(innerKey));
			
			j = new ByteArray();
			j.writeBytes(masterSecret);
			j.writeBytes(pad_2); // see above re: 48 byte pad
			j.writeBytes(innerKey); 
			
			hashMD = md.hash(j);
			// trace("Outer MD5 Key: " + Hex.fromArray(hashMD));
			
			outerKey.writeBytes(hashMD, 0, hashMD.length);
			outerKey.writeBytes(hashSha, 0, hashSha.length);
			var out:String = Hex.fromArray(outerKey);
			// trace("Finished Message: " + out);
			outerKey.position = 0;
			
			return outerKey;
		
		}
		
		public function computeCertificateVerify( side:uint, handshakeMessages:ByteArray ):ByteArray {
			// TODO: Implement this, but I don't forsee it being necessary at this point in time, since for purposes
			// of the override, I'm only going to use TLS
			return null;  
		}
		
		public function getConnectionStates():Object {
			
			if (masterSecret != null) {
				// so now, I have to derive the actual keys from the keyblock that I generated in setPremasterSecret.
				// for MY purposes, I need RSA-AES 128/256 + SHA
				// so I'm gonna have keylen = 32, minlen = 32, mac_length = 20, iv_length = 16
				// but...I can get this data from the settings returned in the constructor when this object is 
				// It strikes me that TLS does this more elegantly...
				
				var mac_length:int = hashSize as Number;
				var key_length:int = keySize as Number;
				var iv_length:int = IVSize as Number;
				
				var client_write_MAC:ByteArray = new ByteArray();
				var server_write_MAC:ByteArray = new ByteArray();
				var client_write_key:ByteArray = new ByteArray();
				var server_write_key:ByteArray = new ByteArray();
				var client_write_IV:ByteArray = new ByteArray();
				var server_write_IV:ByteArray = new ByteArray();
		
				// Derive the keys from the keyblock
				// Get the MACs first
				keyBlock.position = 0;
				keyBlock.readBytes(client_write_MAC, 0, mac_length);
				keyBlock.readBytes(server_write_MAC, 0, mac_length);
				
				// keyBlock.position is now at MAC_length * 2
				// then get the keys
				keyBlock.readBytes(client_write_key, 0, key_length);
				keyBlock.readBytes(server_write_key, 0, key_length);
				
				// keyBlock.position is now at (MAC_length * 2) + (keySize * 2) 
				// and then the IVs
				keyBlock.readBytes(client_write_IV, 0, iv_length);
				keyBlock.readBytes(server_write_IV, 0, iv_length);
				
				// reset this in case it's needed, for some reason or another, but I doubt it
				keyBlock.position = 0;
				
				var client_write:SSLConnectionState = new SSLConnectionState(
						bulkCipher, cipherType, macAlgorithm,
						client_write_MAC, client_write_key, client_write_IV);
				var server_write:SSLConnectionState = new SSLConnectionState(
						bulkCipher, cipherType, macAlgorithm,
						server_write_MAC, server_write_key, server_write_IV);
				
				if (entity == TLSEngine.CLIENT) {
					return {read:server_write, write:client_write};
				} else {
					return {read:client_write, write:server_write};
				}


			} else {
				return {read:new SSLConnectionState, write:new SSLConnectionState};
			}
		}
		
	}
}