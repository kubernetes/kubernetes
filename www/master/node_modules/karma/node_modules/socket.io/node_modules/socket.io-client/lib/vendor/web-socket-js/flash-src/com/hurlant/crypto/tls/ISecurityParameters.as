/**
 * ISecurityParameters
 * 
 * This class encapsulates all the security parameters that get negotiated
 * during the TLS handshake. It also holds all the key derivation methods.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import flash.utils.ByteArray;
	
	public interface ISecurityParameters {
		function get version() : uint;
		function reset():void;
		function getBulkCipher():uint;
		function getCipherType():uint;
		function getMacAlgorithm():uint;
		function setCipher(cipher:uint):void;
		function setCompression(algo:uint):void;
		function setPreMasterSecret(secret:ByteArray):void;
		function setClientRandom(secret:ByteArray):void;
		function setServerRandom(secret:ByteArray):void;
		function get useRSA():Boolean;
		function computeVerifyData(side:uint, handshakeMessages:ByteArray):ByteArray;
		function computeCertificateVerify( side:uint, handshakeRecords:ByteArray):ByteArray;
		function getConnectionStates():Object;
	}
}